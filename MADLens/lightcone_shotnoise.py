from vmad import autooperator, operator
from vmad.core import stdlib
from vmad.lib import fastpm
from vmad.lib import linalg
from vmad.lib.fastpm import FastPMSimulation, ParticleMesh
import numpy
from vmad.lib.linalg import sum, mul
import scipy 
from mpi4py import MPI
import numpy as np
import MADLens.PGD as PGD
from MADLens.util import save_snapshot, save3Dpower
from nbodykit.lab import FFTPower, ArrayCatalog
import pickle
import os
import errno
import resource
import logging
import sys

def BinarySearch_Left(mylist, items):
    print(mylist, items)
    "finds where to insert elements into a sorted array, this is the equivalent of numpy.searchsorted"
    results =[]
    for item in items:
        if item>=max(mylist):
            results.append(len(mylist))
        elif item<=min(mylist):
            results.append(0)
        else:
            results.append(binarysearch_left(mylist,item, low=0, high=len(mylist)-1))
    return np.asarray(results, dtype=int)

def binarysearch_left(A, value, low, high):
    "left binary search"
    if (high < low):
        return low
    mid = (low + high) //2
    if (A[mid] >= value):
        return binarysearch_left(A, value, low, mid-1)
    else:
        return binarysearch_left(A, value, mid+1, high)


class mod_list(list):
    def __add__(self,other):
        assert(len(other)==len(self))
        return [self[ii]+other[ii] for ii in range(len(self))]


@operator
class list_elem:
    """
    take an item from a list
    """
    ain = {'x' : '*',}
    aout = {'elem' : '*'}

    def apl(node, x, i):
        elem = x[i]
        return dict(elem=elem, x_shape=[numpy.shape(xx) for xx in x])

    def vjp(node, _elem, x_shape, i):
        _x       = []
        for ii in range(len(x_shape)):
            _x.append(numpy.zeros(x_shape[ii],dtype='f8'))
        _x[i][:] = _elem

        return dict(_x=_x)
        
    def jvp(node,x_, x, i):
        elem_ = x_[i]
        return dict(elem_=elem_)



@operator
class list_put:
    """ 
    put an item into a list
    """
    ain = {'x': 'ndarray', 'elem': 'ndarray'}
    aout = {'y': 'ndarray'}

    def apl(node, x, elem, i):
        y    = x
        y[i] = elem
        return dict(y=y, len_x = len(x))

    def vjp(node, _y, len_x, i):
        _elem    = _y[i]
        _x       = mod_list([_y[ii] for ii in range(len_x)])
        _x[i]    = np.zeros_like(_elem)
        return dict(_x=_x, _elem=_elem)

    def jvp(node, x_, elem_, len_x, i):
        deriv    = numpy.ones(len_x)
        deriv[i] = 0
        deriv_   = np.zeros(len_x)
        deriv_[i]= 1
        elem_    = np.asarray(elem_,dtype=object)
        e        = np.asarray([elem_ for ii in range(len_x)],dtype=object)
        y_       = numpy.einsum('i,i...->i...',deriv,x_)+numpy.einsum('i,i...->i...',deriv_,e)
        y_       = mod_list(y_)
        return dict(y_=y_)

@operator
class chi_z:
    """
    go from comsoving distance to redshift
    """
    ain  = {'z' : 'ndarray'}
    aout = {'chi': 'ndarray'}

    def apl(node, z, cosmo):
        return dict(chi = cosmo.comoving_distance(z))
    
    def vjp(node, _chi, z, cosmo):
        res = 1./cosmo.efunc(z)/cosmo.H0*cosmo.C
        return dict(_z = numpy.multiply(res,_chi))
    
    def jvp(node, z_, z, cosmo):
        res = 1./cosmo.efunc(z)/cosmo.H0*cosmo.C
        return dict(chi_ = numpy.multiply(res,z_))
    
    
@operator
class z_chi:
    """
    go from redshift to comoving distance 
    """

    ain  = {'chi' : 'ndarray'}
    aout = {'z': 'ndarray'}

    def apl(node, chi, cosmo, z_chi_int):
        return dict(z = z_chi_int(chi))
    
    def vjp(node, _z, z, cosmo, z_chi_int):
        res = cosmo.efunc(z)*cosmo.H0/cosmo.C
        return dict(_chi = res*_z)
    
    def jvp(node, chi_, z, cosmo, z_chi_int):
        res = cosmo.efunc(z)*cosmo.H0/cosmo.C
        return dict(z_ = res*chi_)
    

def get_PGD_params(B,res,n_steps,pgd_dir):
    """
    loads PGD params from file 
    B:      force resolution parameter
    res:    resolution: Boxsize/Nmesh
    nsteps: number of fastpm steps
    pgd_dir: directory in which PGD parameter files are stored
    """

    pgd_file= os.path.join(pgd_dir,'pgd_params_%d_%d_%d.pkl'%(B,res,n_steps))

    if not os.path.isfile(pgd_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pgd_file)
    else:
        pgd_params = pickle.load(open(pgd_file,'rb'))

    alpha0 = pgd_params['alpha0']
    mu     = pgd_params['mu']
    kl     = pgd_params['kl']
    ks     = pgd_params['ks']
    return kl, ks, alpha0, mu



class ImageGenerator:
    """
    rotates, shifts and stacks simulation boxes to fill the observed volume
    so far only 90 degree rotations and shifts are supported but more transformation can be added
    """

    def __init__(self, pm, ds, vert_num):
        """
        defines rotations and shifts
        pm      : 3D pmesh object
        ds      : maximal distance to source plane
        vert_num: how many times to repeat the box in vertical direction
        """

        self.BoxSize    = pm.BoxSize
        self.chi_source = ds 
        self.vert_num   = np.ceil(vert_num)
        # basis vectors
        x      = np.asarray([1,0,0],dtype=int)
        y      = np.asarray([0,1,0],dtype=int)
        z      = np.asarray([0,0,1],dtype=int)
        # identity shift
        self.I = np.zeros(3)
        
        # shuffle directions, only 90 deg rotations, makes a total of 6
        self.M_matrices = [np.asarray([x,y,z],dtype=int), np.asarray([x,z,y],dtype=int),np.asarray([z,y,x],dtype=int),np.asarray([z,x,y],dtype=int), \
                             np.asarray([y,x,z],dtype=int), np.asarray([y,z,x],dtype=int)]
        
        # shifts (only repeat the box twice in x and y)
        self.xyshifts = [np.asarray([0.5,0.5,0.],dtype=float),np.asarray([-0.5,0.5,0.],dtype=float),np.asarray([-0.5,-0.5,0.],dtype=float),np.asarray([0.5,-0.5,0.],dtype=float)]      
        
        # make sure we cover entire redshift range
        self.len = len(self.M_matrices)
        if pm.comm.rank==0:
            print('rotations available: %d'%self.len)
            print('rotations required: %d'%np.ceil(ds/pm.BoxSize[-1]))
    
        try:
            assert(self.len*pm.BoxSize[-1]>ds)
            if pm.comm.rank==0:
                print('sufficient number of rotations to fill lightcone.')
        except:
            if pm.comm.rank==0:
                print('insufficient number of rotations to fill the lightcone.')
        
        self.x = x
        self.z = z
        
    
    def generate(self, di, df):
        """ Returns a list of rotation matrices and shifts that are applied to the box
        di : distance to inital redshift (before taking fastpm step)
        df : distance to final redshift (after taking fastpm step)
        """
        
        if df>self.chi_source:
            return 0, 0
        else:    
            shift_ini = np.floor(max(di,self.chi_source)/self.BoxSize[-1])
            shift_end = np.floor(df/self.BoxSize[-1])
            M = []
            if self.vert_num==1:
                for ii in np.arange(shift_end,shift_ini+1,dtype=int):
                    M.append((self.M_matrices[ii%len(self.M_matrices)], self.I+ii*self.z))
            elif self.vert_num==2:
                for ii in np.arange(shift_end,shift_ini+1,dtype=int):
                    for jj in range(4):
                        M.append((self.M_matrices[ii%len(self.M_matrices)], self.I+ii*self.z+self.xyshifts[jj]))
            else:
                raise ValueError('vertical number of boxes must be 1 or 2, but is %d'%self.vert_num)

            return M
        
    
    
class WLSimulation(FastPMSimulation):
    def __init__(self, stages, cosmology, pm, params, boxsize2D, logger):
        """
        stages:    1d array of floats. scale factors at which to evaluate fastpm simulation (fastpm steps)
        cosmology: nbodykit cosmology object
        pm:        3D pmesh object
        params:    dictionary, parameters for this run
        boxsize2D: list, fov in radians
        """
        self.logger = logger

        q = pm.generate_uniform_particle_grid(shift=0.)        
        FastPMSimulation.__init__(self, stages, cosmology, pm, params['B'], q)

        # source redshifts and distances
        self.zs      = params['zs_source']
        self.ds      = np.asarray([cosmology.comoving_distance(zs) for zs in self.zs],dtype=float)
        # maximal distance at which particles need to be read out
        self.max_ds  = max(self.ds)
        
        # redshift as a function of comsoving distance for underlying cosmology
        z_int          = np.logspace(-12,np.log10(1500),40000)
        chis           = cosmology.comoving_distance(z_int) #Mpc/h
        self.z_chi_int = scipy.interpolate.interp1d(chis,z_int, kind='linear', bounds_error=False, fill_value='extrapolate')

        #how many times to duplicate the box in x-y to span the observed area (probably not desired for machine learning!)
        self.vert_num  = (max(boxsize2D)*max(self.ds))/pm.BoxSize[-1]
        if pm.comm.rank==0:
            print('number of box replications required to fill field of view: %.1f'%self.vert_num)

        # ImageGenerator defines box rotations and shifts
        self.imgen = ImageGenerator(pm,self.max_ds,self.vert_num)       
        # 2D mesh object for convergence field
        self.mappm = ParticleMesh(BoxSize=boxsize2D, Nmesh=params['Nmesh2D'], comm=pm.comm, np=pm.np, resampler='cic')
        self.mappm.affine.period[...] = 0 # disable the periodicity
        
        self.cosmo  = cosmology
        self.params = params
        if self.params['PGD']:
            self.kl, self.ks, self.alpha0, self.mu = get_PGD_params(params['B'],res=pm.BoxSize[0]/pm.Nmesh[0],n_steps=params['N_steps'],pgd_dir=params['PGD_path'])
          
        # mean 3D particle density
        self.nbar    = pm.comm.allreduce(len(self.q))/pm.BoxSize.prod()
        # 2D mesh area in rad^2 per pixel
        self.A       = self.mappm.BoxSize.prod()/self.mappm.Nmesh.prod()
        # redshift independent prefactor from Poisson equation
        self.factor  = 3./2.*self.cosmo.Omega0_m*(self.cosmo.H0/self.cosmo.C)**2
      
    @autooperator('x->xy, d')
    def rotate(self, x, M, boxshift):
        """
        rotates, shift, and separates particle coordinates into distance and xy position
        x:        particle positions
        M:        rotation matrix
        boxshift: shift vector
        """
        y  = linalg.einsum('ij,kj->ki', (M, x))     
        y  = y + self.pm.BoxSize * boxshift
        d  = linalg.take(y, 2, axis=1)
        xy = linalg.take(y, (0, 1), axis=1)
        return dict(xy=xy, d=d)
    
    @autooperator('d->w')
    def wlen(self, d, ds):
        """
        returns the correctly weighted lensing efficiency kernel
        d:   particle distance (assuming parllel projection)
        ds:  source redshift
        """
        z          = z_chi(d,self.cosmo,self.z_chi_int)
        columndens = self.nbar*self.A*linalg.pow(d,2) #particles/Volume*angular pixel area* distance^2 -> 1/L units
        w          = (ds-d)*d/ds*(1.+z)/columndens #distance
        return w

    @autooperator('xy,w ->map')
    def makemap(self, xy, w):
        """
        paint projected particles to 2D mesh
        xy: particle positions in radians
        w:  weighting = projection kernel
        """
        if (self.mappm.affine.period != 0).any():
            raise RuntimeError("The ParticeMesh object must be non-periodic")
        if self.mappm.ndim != 2:
            raise RuntimeError("The ParticeMesh object must be 2 dimensional. ")
            
        compensation = self.mappm.resampler.get_compensation()
        
        layout       = fastpm.decompose(xy, self.mappm)
        map          = fastpm.paint(xy, w, layout, self.mappm)
        # compensation for cic window
        c            = fastpm.r2c(map)
        c            = fastpm.apply_transfer(c, lambda k : compensation(k, 1.0), kind='circular')
        map          = fastpm.c2r(c)
        
        return map

    @autooperator('dx,p, kmaps, dx_PGD->kmaps')
    def interp(self, dx, p, kmaps, dx_PGD, ax, ap, ai, af):

        di, df = self.cosmo.comoving_distance(1. / numpy.array([ai, af],dtype=float) - 1.)

        for M in self.imgen.generate(di, df):
            # if lower end of box further away than source -> do nothing
            if df>self.max_ds:
                continue
            else:
                M, boxshift = M

                # positions of unevolved particles after rotation
                d_approx = self.rotate.build(M=M, boxshift=boxshift).compute('d', init=dict(x=self.q))
                z_approx = z_chi.apl.impl(node=None,cosmo=self.cosmo,z_chi_int=self.z_chi_int,chi=d_approx)['z']
                a_approx = 1. / (z_approx + 1.)
                
                # move particles to a_approx, then add PGD correction
                
                dx1      = dx + p*self.DriftFactor(a_approx, ax, ap)[:, None] + dx_PGD

                # rotate their positions
                xy, d    = self.rotate((dx1 + self.q)%self.pm.BoxSize, M, boxshift)

                # projection
                xy       = ((xy - self.pm.BoxSize[:2]* 0.5)/linalg.broadcast_to(linalg.reshape(d, (len(self.q),1)), (len(self.q), 2))+self.mappm.BoxSize * 0.5 )
                    
                for ii, ds in enumerate(self.ds):
                    w        = self.wlen(d,ds)
                    mask     = stdlib.eval(d, lambda d, di=di, df=df, ds=ds, d_approx=d_approx: 1.0 * (d_approx < di) * (d_approx >= df) * (d <=ds))
                    kmap     = list_elem(kmaps,ii)
                    kmap_    = self.makemap(xy, w*mask)*self.factor
                    kmap     = linalg.add(kmap_,kmap)
                    kmaps    = list_put(kmaps,kmap,ii)
         
        return kmaps

    @autooperator('kmaps->kmaps')
    def no_interp(self,kmaps,q,ai,af,jj):
        
        di, df = self.cosmo.comoving_distance(1. / numpy.array([ai, af],dtype=object) - 1.)

        #q      = np.random.random(self.q.shape)*self.pm.BoxSize[0]

        for M in self.imgen.generate(di, df):
                # if lower end of box further away than source -> do nothing
            if df>self.max_ds:
                if self.params['logging']:
                    self.logger.info('imgen passed, %d'%jj)
                continue
            else:
                if self.params['logging']:
                    self.logger.info('imgen with projection, %d'%jj)
                M, boxshift = M
                xy, d    = self.rotate(q, M, boxshift)
                d_approx = self.rotate.build(M=M, boxshift=boxshift).compute('d', init=dict(x=q))
            
                xy       = ((xy - self.pm.BoxSize[:2] * 0.5)/linalg.broadcast_to(linalg.reshape(d, (len(self.q),1)), (len(self.q), 2))+self.mappm.BoxSize * 0.5 )

                for ii, ds in enumerate(self.ds):
                    if self.params['logging']: 
                        self.logger.info('projection, %d'%jj)
                    w        = self.wlen(d,ds)
                    mask     = stdlib.eval(d, lambda d, di=di, df=df, ds=ds, d_approx=d_approx : 1.0 * (d_approx < di) * (d_approx >= df) * (d <=ds))
                    kmap_    = self.makemap(xy, w*mask)*self.factor 
                    kmap     = list_elem(kmaps,ii)
                    kmap     = linalg.add(kmap_,kmap)
                    kmaps    = list_put(kmaps,kmap,ii)

        return kmaps
            



    @autooperator('rhok->kmaps')
    def run_interpolated(self, rhok):

        dx, p  = self.firststep(rhok)
        pt     = self.pt
        stages = self.stages
        q      = self.q
        Om0    = pt.Om0

        powers = []
        kmaps  = [self.mappm.create('real', value=0.) for ii in range(len(self.ds))]
        
        f, potk= self.gravity(dx)

        for ai, af in zip(stages[:-1], stages[1:]):
            # central scale factor
            ac = (ai * af) ** 0.5

            # kick
            dp = f * (self.KickFactor(ai, ai, ac) * 1.5 * Om0)
            p  = p + dp

            # drift
            ddx = p * self.DriftFactor(ai, ac, ac)
            dx  = dx + ddx

            if self.params['PGD']:
                alpha    = self.alpha0 * ac **self.mu
                dx_PGD   = PGD.PGD_correction(self.q + dx, alpha, self.kl, self.ks, self.fpm, self.q)
            else:
                dx_PGD = 0.

            #if interpolation is on, only take 'half' and then evolve according to their position
            kmaps = self.interp(dx, p, kmaps, dx_PGD, ac, ac, ai, af)

            # drift
            ddx = p * self.DriftFactor(ac, ac, af)
            dx  = dx + ddx

            # force
            f, potk = self.gravity(dx)

            # kick
            dp = f * (self.KickFactor(ac, af, af) * 1.5 * Om0)
            p  = p + dp
        
        
        return dict(kmaps=kmaps)


    

    @autooperator('rhok->kmaps')
    def run(self, rhok):

        stages = self.stages

        q      = np.random.random(self.q.shape)*self.pm.BoxSize[0]

       
        powers = []
        kmaps  = [self.mappm.create('real', value=0.) for ds in self.ds]
        
        jj = 0 #counting steps for saving snapshots
        for ai, af in zip(stages[:-1], stages[1:]):
            # drift (update positions)   
 
            jj+=1

            kmaps = self.no_interp(kmaps, q, ai, af, jj)


        return kmaps


def run_wl_sim(params, num, cosmo, randseed = 187):
    '''
    params:  dictionary, of run specific settings
    num:     int, number of this run (which run out of #N_maps)
    cosmo:   nbodykit cosmology object, see https://nbodykit.readthedocs.io/en/latest/index.html
    label:   string, label of this run. used in filename if 3D matter distribution is saved
    randseed:random seed for generating initial conditions
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if params['logging']: 
        a_logger = logging.getLogger()
        lformat = logging.Formatter('%(asctime)s - %(message)s')

        output_file_handler = logging.FileHandler("proc%d.log"%rank)
        stdout_handler = logging.StreamHandler(sys.stdout)
        output_file_handler.setFormatter(lformat)

        a_logger.addHandler(output_file_handler)
        a_logger.addHandler(stdout_handler)
    else:
        a_logger = None
    # particle mesh for fastpm simulation
    pm        = fastpm.ParticleMesh(Nmesh=params['Nmesh'], BoxSize=params['BoxSize'], comm=MPI.COMM_WORLD, resampler='cic')
    
    # 2D FOV in radians
    BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]

    np.random.seed(randseed)
    randseeds = np.random.randint(0,1e6,100)

    # generate initial conditions
    cosmo     = cosmo.clone(P_k_max=30)
    rho       = pm.generate_whitenoise(seed=randseeds[num], unitary=False, type='complex')
    #set zero mode to zero
    rho.csetitem([0, 0, 0], 0)
    rho       = rho.c2r()
    if params['logging']:
        logging.info('simulations starts')
    # weak lensing simulation object
    wlsim     = WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params, logger=a_logger)

    #build
    if params['interpolate']:
        model     = wlsim.run_interpolated.build()
    else:
        model     = wlsim.run.build()

    # results
    kmap_vjp,kmap_jvp = [None, None]
    # compute
    if params['forward'] and (params['vjp'] or params['jvp']):
        kmaps, tape = model.compute(vout='kmaps', init=dict(rhok=rho),return_tape=True)
        if params['vjp']:
            vjp         = tape.get_vjp()
            kmap_vjp    = vjp.compute(init=dict(_kmaps=kmaps), vout='_rhok')
        if params['jvp']:
            jvp      = tape.get_jvp()
            kmap_jvp = jvp.compute(init=dict(rhok_=rho), vout='kmaps_')

    if params['forward']:
        kmaps       = model.compute(vout='kmaps', init=dict(rhok=rho))

    return kmaps, [kmap_vjp,kmap_jvp], pm
