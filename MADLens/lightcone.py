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
from nbodykit.lab import FFTPower, ArrayCatalog
import pickle
import os
import errno
import resource


@operator
class list_elem:
    """
    take an item from a list
    """
    ain = {'x' : 'ndarray',}
    aout = {'y' : 'ndarray'}

    def apl(node, x, i):
        y = x[i]
        return dict(y=y)

    def vjp(node, _y, i):
        pass

@operator 
class list_put:
    """ 
    put an item into a list
    """
    ain = {'x': 'ndarray', 'item':'ndarray'}
    aout = {'y': 'ndarray'}

    def apl(node, x, item, i):
        x[i] = item
        y = x
        return dict(y=y)

    def vjp(node, _y, i):
        pass


def save3Dpower(mesh,zi,zf,params):
    path = params['snapshot_path'] 
    power = FFTPower(mesh, mode='1d')
    if mesh.pm.comm.rank==0:
        pickle.dump([zi,zf,power],open(path+'power_z%d_%s.pkl'%(zf*100,label),'wb'))
    return True

def paint2mesh(x,pm,zi,zf,params,dump_mesh=False,interlaced=True,compensated=True):
    cat    = ArrayCatalog({'Position' : x}, BoxSize=pm.BoxSize)
    mesh   = cat.to_mesh(Nmesh=pm.Nmesh, interlaced=interlaced, compensated=compensated, window='cic')
    if dump_mesh:
        save3Dpower(mesh,zi=zi,zf=zf,params=params)
    return True

def get_interp_factors(x_,x,y):
    indices = np.searchsorted(x, x_)
    
    #ensure periodic boundary conditions
    y = np.append(y, y[-1])
    y = np.append(y, y[0])
    
    #ensure that x[indices] is defined for all indices (value is unimportant)
    x = np.append(x, x[-1]+1)
    factors = (y[indices]-y[indices-1])/(x[indices]-x[indices-1])
    return factors

    

#TODO: add support for derivative here
@operator
class DriftFactor:
    """
    Drift Factor for evolution around central redshift 
    """
    ain = {'af':'ndarray'}
    aout= {'drift':'ndarray'}

    def apl(node,af,ai,ac,pt):
        result = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        return dict(drift=result)

    def vjp(node,_drift,ai,ac,pt):
        pass

    def jvp(node,af_,ai,ac,pt):
        pass#factors = get_interpolation_factors(af_,pt.)

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
    

def get_PGD_params(B,res,n_steps):
    """
    loads PGD params from file 
    B:      force resolution parameter
    res:    resolution: Boxsize/Nmesh
    nsteps: number of fastpm steps
    """

    pgd_dir = '../pgd_params/'
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
        self.vert_num   = vert_num
        # basis vectors
        x      = np.asarray([1,0,0])
        y      = np.asarray([0,1,0])
        z      = np.asarray([0,0,1])
        # identity shift
        self.I = np.zeros(3)
        
        # shuffle directions, only 90 deg rotations, makes a total of 6
        self.M_matrices = [np.asarray([x,y,z]), np.asarray([x,z,y]),np.asarray([z,y,x]),np.asarray([z,x,y]), \
                             np.asarray([y,x,z]), np.asarray([y,z,x])]
        
        # shifts (only repeat the box twice in x and y)
        self.xyshifts = [np.asarray([0.5,0.5,0.]),np.asarray([-0.5,0.5,0.]),np.asarray([-0.5,-0.5,0.]),np.asarray([0.5,-0.5,0.])]      
        
        # make sure we cover entire redshift range
        self.len = len(self.M_matrices)
        if self.len*pm.BoxSize[-1]>ds:
            if pm.comm.rank==0:
                print('sufficient number of rotations to fill lightcone')
        else:
            if pm.comm.rank==0:
                print('insufficient number of rotations to fill lightcone')
        
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
    def __init__(self, stages, cosmology, pm, params, boxsize2D):
        """
        stages:    fastpm steps in scale factor
        cosmology: nbodykit comsology object
        pm:        3D pmesh object
        params:    run specifuc parameters
        boxsize2D: fov in radians on to which to project
        """

        q = pm.generate_uniform_particle_grid(shift=0.)        
        FastPMSimulation.__init__(self, stages, cosmology, pm, params['B'], q)

        self.zs      = params['zs_source']
        self.ds      = np.asarray([cosmology.comoving_distance(zs) for zs in self.zs])
        self.max_ds  = max(self.ds)
        
        z_int          = np.logspace(-8,np.log10(1500),10000)
        chis           = cosmology.comoving_distance(z_int) #Mpc/h
        self.z_chi_int = scipy.interpolate.interp1d(chis,z_int, kind=3,bounds_error=False, fill_value=0.)

        #how many times to duplicate the box in x-y to span the observed area (probably not desired for machine learning!)
        self.vert_num  = np.int32(np.ceil((max(boxsize2D)*max(self.ds))/pm.BoxSize[-1]))
        if pm.comm.rank==0:
            print('number of box replications required to fill field of view: %d'%self.vert_num)

        self.imgen = ImageGenerator(pm,self.max_ds,self.vert_num)       
        # 2D mesh for painting field, in radians
        self.mappm = ParticleMesh(BoxSize=boxsize2D, Nmesh=params['Nmesh2D'], comm=pm.comm, np=pm.np, resampler='cic')
        self.mappm.affine.period[...] = 0 # disable the periodicity
        
        self.cosmo = cosmology
        self.params= params
        if self.params['PGD']:
            self.kl, self.ks, self.alpha0, self.mu = get_PGD_params(params['B'],res=pm.BoxSize[0]/pm.Nmesh[0],n_steps=params['N_steps'])
          
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
        cosmo      = self.cosmo
        z          = z_chi(d,cosmo,self.z_chi_int)
        columndens = self.nbar*self.A*(d)**2 #particles/Volume*angular pixel area* distance^2 -> 1/L units
        kernel     = (ds-d)*d/ds*(1.+z)/columndens #distance
        return kernel

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
            
    @autooperator('x->map')
    def make3Dmap(self,x):
        """
        paint particle to 3D map (required if dumping snapshots)
        """
        compensation = self.pm.resampler.get_compensation()
        w = 1./self.nbar
        layout       = fastpm.decompose(x, self.pm)
        map          = fastpm.paint(x, w, layout, self.pm)
        # compensation for cic window
        c            = fastpm.r2c(map)
        c            = fastpm.apply_transfer(c, lambda k : compensation(k, 1.0), kind='circular')
        map          = fastpm.c2r(c)
        return map

    # can we remove p here?
    @autooperator('dx, p, kmaps->kmaps')
    def interp(self, dx, p, kmaps, dx_PGD, ax, ap, ai, af):

        di, df = self.cosmo.comoving_distance(1. / numpy.array([ai, af]) - 1.)

        for M in self.imgen.generate(di, df):
            # if lower end of box further away than source -> do nothing
            if df>self.max_ds:
                continue
            else:
                M, boxshift = M

                # positions of unevolved particles after rotation
                d_approx = self.rotate.build(M=M, boxshift=boxshift).compute('d', init=dict(x=self.q))
                a_approx = 1. / (z_chi(d_approx, self.cosmo, self.z_chi_int) + 1.)

                # move particles to a_approx, then add PGD correction
                dx1      = dx + linalg.einsum("ij,i->ij", [p,DriftFactor(a_approx, ax, ap, self.pt)]) + dx_PGD

                # rotate their positions
                xy, d    = self.rotate((dx1 + self.q)%self.pm.BoxSize, M, boxshift)

                # projection
                xy       = (((xy - self.pm.BoxSize[:2]* 0.5))/ linalg.stack((d,d), axis=-1)+ self.mappm.BoxSize * 0.5 )
                    
                for ii, ds in enumerate(self.ds):
                    w        = self.wlen(d,ds)
                    mask     = stdlib.eval(d, lambda d, di=di, df=df, ds=ds, d_approx=d_approx: 1.0 * (d_approx < di) * (d_approx >= df) * (d <=ds))
                    kmap_    = self.makemap(xy, w*mask)*self.factor
                    kmap     = list_elem(kmaps, ii)
                    kmaps    = list_put(kmaps,kmap_+kmap,ii)

        return kmaps


    @autooperator('rhok->kmaps')
    def run(self, rhok):

        def getrss():
            usage = resource.getrusage(resource.RUSAGE_SELF)
            names=('ru_utime','ru_stime','ru_maxrss','ru_ixrss','ru_idrss') 
            descs=('User time','System time','Max. Resident Set Size','Shared Memory Size','Unshared Memory Size')
            return usage, descs, names

        dx, p  = self.firststep(rhok)
        pt     = self.pt
        stages = self.stages
        q      = self.q
        Om0    = pt.Om0

        powers =[]
        kmaps  = [self.mappm.create('real', value=0.) for ds in self.ds]
        
        f, potk= self.gravity(dx)

        for ai, af in zip(stages[:-1], stages[1:]):
            # central scale factor
            ac = (ai * af) ** 0.5

            if self.params['analyze']:
                if self.pm.comm.rank == 0:
                    usage, descs, names = getrss()
                    for ii in range(len(descs)):
                        stdlib.watchpoint(f, lambda f, ai=ai: print('ai', ai, '%-25s (%-10s) = %s'%(descs[ii], names[ii], getattr(usage, names[ii]))))

            # kick
            dp = f * (self.KickFactor(ai, ai, ac) * 1.5 * Om0)
            p  = p + dp

            # drift
            ddx = p * self.DriftFactor(ai, ac, ac)
            dx  = dx + ddx

            if self.params['PGD']:
                alpha    = self.alpha0 * ac **self.mu
                dx_PGD   = PGD_correction(self.q + dx, alpha, self.kl, self.ks, self.fpm, self.q)
            else:
                dx_PGD = 0.

            #if interpolation is on, only take 'half' and then evolve according to their position
            kmaps = self.interp(dx, p, kmaps, dx_PGD, ac, ac, ai, af)

            # drift
            ddx = p * self.DriftFactor(ac, ac, af)
            dx  = dx + ddx

            zf = 1./af-1.
            zi = 1./ai-1.

            # force
            f, potk = self.gravity(dx)

            # kick
            dp = f * (self.KickFactor(ac, af, af) * 1.5 * Om0)
            p  = p + dp
        
        
        return dict(kmaps=kmaps)


def test_wl(params, num, cosmo, randseed = 187):
    '''
    params:  dictionary of run specific settings
    num:     number of this run (which out of N_maps)
    cosmo:   nbodykit cosmology object, see https://nbodykit.readthedocs.io/en/latest/index.html
    label:   label of this run. used in filename if 3D matter distribution is saved
    randseed:random seed for generating initial conditions
    '''

    pm        = fastpm.ParticleMesh(Nmesh=params['Nmesh'], BoxSize=params['BoxSize'], comm=MPI.COMM_WORLD, resampler='cic')
    BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]

    np.random.seed(randseed)
    randseeds = np.random.randint(0,1e6,100)

    cosmo     = cosmo.clone(P_k_max=30)
    rho       = pm.generate_whitenoise(seed=randseeds[num], unitary=False, type='complex')
    rho       = rho.apply(lambda k, v:(cosmo.get_pklin(k.normp(2) ** 0.5, 0) / pm.BoxSize.prod()) ** 0.5 * v)
    #set zero mode to zero
    rho.csetitem([0, 0, 0], 0)
    rho       = rho.c2r()

    wlsim     = WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params)

    model     = wlsim.run.build()
    kmaps     = model.compute(vout='kmaps', init=dict(rhok=rho.r2c()))
    
    kmaps_deriv = None
    if params['mode']=='backprop': 
        kmap_deriv = model.compute_with_vjp(init=dict(rhok=rho.r2c()), v=dict(_kmap=kmap))

    return kmaps, kmaps_deriv, pm
