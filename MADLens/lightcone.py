from vmad import autooperator, operator
from vmad.core import stdlib
from vmad.core.stdlib.finite_operator import finite_operator
from vmad.core.symbol import Literal, ListPlaceholder
from vmad.lib import fastpm
from vmad.lib import linalg
from vmad.lib.fastpm import FastPMSimulation, ParticleMesh
import numpy
from vmad.lib.linalg import sum, mul, broadcast_to
import scipy 
from mpi4py import MPI
import numpy as np
import MADLens.PGD as PGD
from MADLens.util import save_snapshot, save3Dpower
from MADLens.power import get_Pk_EH, normalize
from nbodykit.lab import FFTPower, ArrayCatalog
from nbodykit.cosmology import Planck15
import pickle
import os
import errno
import resource
import logging
import sys

def get_kedges(x1):
    """
    Compute the k-modes for a given complex pm-object
    """
    dk = 2*np.pi/min(x1.BoxSize)**2
    kmin= dk
    kedges = np.arange(kmin, np.pi*min(x1.Nmesh)/max(x1.BoxSize) + dk/2, dk)
    kedges  = np.append(kedges,2*np.pi*min(x1.Nmesh)/max(x1.BoxSize)+dk)
    kedges  = np.insert(kedges,0,0)
    one     = x1.pm.create(type(x1), value=1).apply(x1._expand_hermitian, kind='index', out=Ellipsis)
    ind     = np.zeros(x1.value.shape, dtype='intp')
    ind     = x1.apply(lambda k, v: np.digitize(k.normp(2) ** 0.5, kedges), out=ind)
    N       = np.bincount(ind.flat, weights=one.real.flat, minlength=len(kedges+1))
    N       = x1.pm.comm.allreduce(N)
    mask    = np.where(N!=0)
    N       = N[mask]
    kedges  = kedges[mask]
    ks      = (kedges[0:-1]+kedges[1:])/2
    return kedges,ks

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
        elem_    = np.asarray(elem_)
        e        = np.asarray([elem_ for ii in range(len_x)])
        y_       = numpy.einsum('i,i...->i...',deriv,x_)+numpy.einsum('i,i...->i...',deriv_,e)
        y_       = mod_list(y_)
        return dict(y_=y_)

def deriv_integral(x, Omega0_m):
    """
    Derivative of the comoving distance integral with respect to matter density
    """
    #Create the denominator of the integral
    E =(Omega0_m * ((1+x)**3 -1)+ 1)**(-3/2)
    diriv_factor = (-1/2) * ((1+x)**3-1)

    return E*diriv_factor

@operator
class chi_z:
    """
    Go from redshift to comoving distance
    """
    ain  = {'Omega0_m': '*'}
    aout = {'chi':'*'}

    def apl(node, Omega0_m, z, cosmo):
        #Calculate the integral from 0->z
        E         = lambda x: (Omega0_m  * ((1+x)**3 -1)+1)**(-1/2)
        Dc , _    = scipy.integrate.quad(E, 0, z)
        return dict(chi = Dc*cosmo.C/cosmo.H0)

    def vjp(node, _chi, Omega0_m, z, cosmo):
        #Return the derivative of the integral WR2 Omega0_m and mult by _chi
        _Omega0_m  =  _chi  *  scipy.integrate.quad(deriv_integral, 0, z, args=Omega0_m)[0]
        #Multiply by hubble distance and return
        return dict(_Omega0_m = _Omega0_m*cosmo.C/cosmo.H0)

    def jvp(node, Omega0_m_, Omega0_m, z, cosmo):
        #Find derivative with respevct to omega_0 and mult by Omega0_m_
        Omega0_m_   = Omega0_m_ * scipy.integrate.quad(deriv_integral, 0, z, args=Omega0_m)[0]

        #Multiply by hubble distance
        return dict(chi_ = Omega0_m_*cosmo.C/cosmo.H0)

@operator
class z_chi:
    """
    go from redshift to comoving distance 
    """

    ain  = {'chi' : 'ndarray', 'Om0':'float'}
    aout = {'z': 'ndarray'}

    def apl(node, chi, Om0, cosmo, z_chi_int, z_chi_int_upper, z_chi_int_lower):
        return dict(z = z_chi_int(chi))
    
    def vjp(node, _z,Om0,chi, z,cosmo, z_chi_int, z_chi_int_upper, z_chi_int_lower):
        res = cosmo.efunc(z)*cosmo.H0/cosmo.C
        sol_upper = z_chi_int_upper(chi)
        sol_lower= z_chi_int_lower(chi)
        res_o = (sol_upper-sol_lower)/2e-3*cosmo.H0/cosmo.C

        return dict(_chi = res*_z, _Om0=res_o*_z)
    
    def jvp(node, chi_, Om0_, z, Om0, chi, cosmo, z_chi_int, z_chi_int_upper, z_chi_int_lower):
        res = cosmo.efunc(z)*cosmo.H0/cosmo.C
        sol_upper = z_chi_int_upper(chi)
        sol_lower= z_chi_int_lower(chi)
        res_o = (sol_upper-sol_lower)/2e-3

        j = res*chi_+ res_o*Om0_*cosmo.H0/cosmo.C

        return dict(z_ = j)

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
            #elif self.vert_num==2:
            #    for ii in np.arange(shift_end,shift_ini+1,dtype=int):
            #        for jj in range(4):
            #            M.append((self.M_matrices[ii%len(self.M_matrices)], self.I+ii*self.z+self.xyshifts[jj]))
            else:
                raise ValueError('vertical number of boxes must be 1, but is %d'%self.vert_num)

            return M
        
    
    
class WLSimulation(FastPMSimulation):
    def __init__(self, stages, cosmology, pm, params, boxsize2D, k_s, kedges, logger):
        """
        stages:    1d array of floats. scale factors at which to evaluate fastpm simulation (fastpm steps)
        cosmology: nbodykit cosmology object
        pm:        3D pmesh object
        params:    dictionary, parameters for this run
        boxsize2D: list, fov in radians
        """
        self.logger = logger

        q = pm.generate_uniform_particle_grid(shift=0.)        
        if params['param_derivs']==True:
            print('taking derivatives wrt Om0 and s8')
            FastPMSimulation_Om0.__init__(self, stages, pm, params['B'], q)
        else:
            print('no para_derivs requested')
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
        
        self.k_s = np.array(k_s)
        self.kedges=np.array(kedges)

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

        #For z_chi fd
        cosmo = Planck15    
        cosmo = cosmo.clone(Omega_ncdm=0)
        cosmo_upper = cosmo.clone(Omega0_cdm=Planck15.Omega0_m+1e-3)
        cosmo_upper = cosmo_upper.match(sigma8=Planck15.sigma8)
        z_int          = np.logspace(-8,np.log10(1500),10000)
        chis           = cosmo_upper.comoving_distance(z_int) #Mpc/h
        self.z_chi_int_upper = scipy.interpolate.interp1d(chis,z_int, kind='linear',bounds_error=False, fill_value='extrapolate')

        cosmo_lower = cosmo.clone(Omega0_cdm=cosmo.Omega0_m-1e-3)
        cosmo_lower = cosmo_lower.match(sigma8=cosmo.sigma8)
        z_int          = np.logspace(-8,np.log10(1500),10000)
        chis           = cosmo_lower.comoving_distance(z_int) #Mpc/h
        self.z_chi_int_lower = scipy.interpolate.interp1d(chis,z_int, kind='linear',bounds_error=False, fill_value='extrapolate')      

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
    
    @autooperator('d, Om0, ds->w')
    def wlen(self, d, Om0, ds):
        """
        returns the correctly weighted lensing efficiency kernel
        d:   particle distance (assuming parllel projection)
        ds:  source redshift
        """
        z          = z_chi(d, Om0, self.cosmo,self.z_chi_int, self.z_chi_int_upper, self.z_chi_int_lower)
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
    @autooperator('rhok,Om0 -> p,dx')
    def first_step_workaround(self, rhok, Om0, stages, q, FactoryCache):
        E  = finite_operator(Om0, lambda Om0, a=stages[0], support=self.support, FactoryCache=FactoryCache:
                               fastpm.firststep_E(Om0, a, support, FactoryCache), epsilon=1e-3, mode='central')
        D1 = finite_operator(Om0, lambda Om0, a=stages[0], support=self.support, FactoryCache=FactoryCache:
                               fastpm.firststep_D1(Om0, a, support, FactoryCache), epsilon=1e-3, mode='central')
        D2 = finite_operator(Om0, lambda Om0, a=stages[0], support=self.support, FactoryCache=FactoryCache:
                               fastpm.firststep_D2(Om0, a, support, FactoryCache), epsilon=1e-3, mode='central')
        f1 = finite_operator(Om0, lambda Om0, a=stages[0], support=self.support, FactoryCache=FactoryCache:
                               fastpm.firststep_f1(Om0, a, support, FactoryCache), epsilon=1e-3, mode='central')
        f2 = finite_operator(Om0, lambda Om0, a=stages[0], support=self.support, FactoryCache=FactoryCache:
                               fastpm.firststep_f2(Om0, a, support, FactoryCache), epsilon=1e-3, mode='central')
        dx1, dx2 = fastpm.lpt(rhok, self.q, self.pm)
        D1 = broadcast_to(D1, self.q.shape)
        D2 = broadcast_to(D2, self.q.shape)
        dx1 = dx1 * D1
        dx2 = dx2 * D2
        E = broadcast_to(E, self.q.shape)
        f1 = broadcast_to(f1, self.q.shape)
        f2 = broadcast_to(f2, self.q.shape)

        p = stages[0]**2*f1*E*dx1 + stages[0]**2*f2*E*dx2
        dx = dx1 + dx2
        return dict(p=p, dx=dx)

    @autooperator('dx,Om0, p, dx_PGD->kmaps')
    def interp(self, dx,Om0, p, dx_PGD, ax, ap, ai, af, FactoryCache):

        # Find distance to fastpm steps 
        di_value, df_value = self.cosmo.comoving_distance(1. / numpy.array([ai, af]) - 1.)
        di = chi_z(Om0, 1. /ai - 1., self.cosmo)
        df = chi_z(Om0, 1. /af - 1., self.cosmo)

        #Calculate source Distance
        ds = Literal(0)
        ds = [chi_z(Om0, z, self.cosmo) for z in self.zs]

        #Initialize empty kmap list of symbols
        zero_map = Literal(self.mappm.create('real',value=0.))
        kmaps = [zero_map for ii in range(len(self.zs))]

        for M in self.imgen.generate(di_value, df_value):
            # if lower end of box further away than source -> do nothing
            if df_value>self.max_ds:
                continue
            else:
                M, boxshift = M

                # positions of unevolved particles after rotation
                d_approx = self.rotate.build(M=M, boxshift=boxshift).compute('d', init=dict(x=self.q))
                z_approx = z_chi.apl.impl(node=None,cosmo=self.cosmo,z_chi_int=self.z_chi_int,chi=d_approx, Om0=Om0,
                                          z_chi_int_upper=self.z_chi_int_upper, z_chi_int_lower=self.z_chi_int_lower)['z']
                a_approx = 1. / (z_approx + 1.)

                # move particles to a_approx, then add PGD correction
                drift = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, a_approx=a_approx, ax=ax, ap=ap:
                                       fastpm.DriftFactor(Om0, support, FactoryCache, a_approx, ax, ap), epsilon=1e-5, mode='central')
                drift = broadcast_to(linalg.reshape(drift, (len(self.q), 1)), self.q.shape)
                dx1      = dx + p*drift + dx_PGD

                # rotate/project their positions
                xy, d    = self.rotate((dx1+self.q)%self.pm.BoxSize, M, boxshift)
                xy       = ((xy - self.pm.BoxSize[:2]* 0.5)/linalg.broadcast_to(linalg.reshape(d, (len(self.q),1)), (len(self.q), 2))+self.mappm.BoxSize * 0.5 )

                for ii, zs in enumerate(self.zs):
                    if self.params['logging']: 
                        self.logger.info('projection, %d'%jj)

                    #Get the right shapes for backprop
                    dsi = ds[ii]
                    dsi = broadcast_to(dsi, stdlib.eval(d, lambda d:d.shape))
                    Om0 = broadcast_to(Om0, stdlib.eval(d, lambda d:d.shape))
                    factor  = 3./2.*Om0*(self.cosmo.H0/self.cosmo.C)**2

                    #Compute lensing efficiency and mask, then project to make map
                    w        = self.wlen(d,Om0,dsi)
                    mask     = stdlib.eval([d, di, df, dsi], lambda args, d_approx=d_approx: 1.0 * (d_approx< args[1]) * (d_approx >= args[2]) * (args[0] <=args[3]))
                    kmap_    = self.makemap(xy,w*mask*factor)
                    kmaps[ii] = kmaps[ii]+kmap_
        return kmaps

    @autooperator('dx, Om0, p, dx_PGD->kmaps')
    def no_interp(self,dx,Om0, p,dx_PGD,ai,af,jj ):
        dx = dx + dx_PGD

        # Find distance to fastpm steps 
        di_value, df_value = self.cosmo.comoving_distance(1. / numpy.array([ai, af]) - 1.)
        di = chi_z(Om0, 1. /ai - 1., self.cosmo)
        df = chi_z(Om0, 1. /af - 1., self.cosmo)

        #Calculate source Distance
        ds = Literal(0)
        ds = [chi_z(Om0, z, self.cosmo) for z in self.zs]

        #Initialize empty map list
        zero_map = Literal(self.mappm.create('real',value=0.)) 
        kmaps = [zero_map for ii in range(len(self.zs))]

        for M in self.imgen.generate(di_value, df_value):
                # if lower end of box further away than source -> do nothing
            if df_value>self.max_ds :
                if self.params['logging']:
                    self.logger.info('imgen passed, %d'%jj)
                continue
            else:
                if self.params['logging']:
                    self.logger.info('imgen with projection, %d'%jj)

                #Find/apply rotations and shifts
                M, boxshift = M
                xy, d    = self.rotate((dx + self.q)%self.pm.BoxSize, M, boxshift)
                _, d_approx = self.rotate(M=M, boxshift=boxshift, x=self.q)
                xy       = ((xy - self.pm.BoxSize[:2] * 0.5)/linalg.broadcast_to(linalg.reshape(d, (len(self.q),1)), (len(self.q), 2))+self.mappm.BoxSize * 0.5 )

                for ii, zs in enumerate(self.zs):
                    if self.params['logging']: 
                        self.logger.info('projection, %d'%jj)

                    #Get the right shapes for backprop
                    dsi = ds[ii]
                    dsi = broadcast_to(dsi, stdlib.eval(d, lambda d:d.shape))
                    Om0 = broadcast_to(Om0, stdlib.eval(d, lambda d:d.shape))
                    factor  = 3./2.*Om0*(self.cosmo.H0/self.cosmo.C)**2

                    #Compute lensing efficiency and project to make map
                    w        = self.wlen(d,Om0,dsi)
                    mask     = stdlib.eval([d, di, df, dsi, d_approx], lambda args: 1.0 * (args[4]< args[1]) * (args[4] >= args[2]) * (args[0] <=args[3]))
                    kmap_    = self.makemap(xy,w*mask*factor)
                    kmaps[ii] = kmaps[ii]+kmap_

        return kmaps



    @autooperator('rho, Om0, sigma8->kmaps')
    def run_interpolated(self, rho, Om0, sigma8):


        rhok = fastpm.r2c(rho)

        # Calculate EH power for initial modes
        norm = finite_operator(Om0, lambda Om0, R=8, tf='EH': normalize(R, Om0, tf), epsilon=1e-3, mode='central')
        norm = (sigma8/norm)**2
        norm = broadcast_to(norm, self.k_s.shape)
        transfer =  get_Pk_EH(Om0, cosmo=self.cosmo, z=0, k=self.k_s)**.5*norm**.5/self.pm.BoxSize.prod()**.5
        digitizer = fastpm.apply_digitized.isotropic_wavenumber(self.k_s)
        rhok= fastpm.apply_digitized(x=rhok, tf=transfer, digitizer=digitizer, kind='wavenumber', mode='amplitude')

        #pt     = self.pt
        stages = self.stages
        q      = self.q
        FactoryCache = fastpm.CosmologyFactory()
        #First step work around
        p, dx = self.first_step_workaround(rhok, Om0, stages, q, FactoryCache)

        zero_map = Literal(self.mappm.create('real', value=0.))
        kmaps  = [zero_map for zs in self.zs]
        f, potk= fastpm.gravity(dx, self.q, self.fpm)

        for ai, af in zip(stages[:-1], stages[1:]):
            # central scale factor
            ac = (ai * af) ** 0.5

            # kick
            kick = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.KickFactor(Om0, support, FactoryCache, ai, ai, ac), epsilon=1e-3, mode='central')
            kick = broadcast_to(kick * 1.5 * Om0, self.q.shape)
            dp = f *kick
            p  = p + dp

            # drift
            drift = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.DriftFactor(Om0, support, FactoryCache, ai, ac, ac), epsilon=1e-3, mode='central')
            drift = broadcast_to(drift, self.q.shape) 
            ddx = p * drift
            dx  = dx + ddx
            if self.params['PGD']:
                alpha    = self.alpha0 * ac **self.mu
                dx_PGD   = PGD.PGD_correction(self.q + dx, alpha, self.kl, self.ks, self.fpm, self.q)
            else:
                dx_PGD = 0.

            #if interpolation is on, only take 'half' and then evolve according to their position
            kmaps_ = self.interp(dx, Om0, p , dx_PGD, ac, ac, ai, af,FactoryCache, kmaps=ListPlaceholder(len(self.zs)))

            for ii in range(len(self.zs)):
                kmaps[ii] = kmaps[ii]+kmaps_[ii]
            # drift
            drift = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.DriftFactor(Om0, support, FactoryCache, ac, ac, af), epsilon=1e-3, mode='central')
            drift = broadcast_to(drift, self.q.shape) 
            ddx = p * drift
            dx  = dx + ddx

            # force (compute force)
            f, potk = fastpm.gravity(dx, self.q, self.fpm)

            # kick
            kick = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.KickFactor(Om0, support, FactoryCache, ac, af, af), epsilon=1e-3, mode='central')
            kick = broadcast_to(kick * 1.5 * Om0, self.q.shape)
            dp = f * kick
            p  = p + dp
        
        
        return dict(kmaps=kmaps)

    @autooperator('rho, Om0, sigma8 ->kmaps')
    def run(self, rho, Om0, sigma8 ):

        rhok = fastpm.r2c(rho)

        # Calculate EH power for initial modes
        norm = finite_operator(Om0, lambda Om0, R=8, tf='EH': normalize(R, Om0, tf), epsilon=1e-3, mode='central')
        norm = (sigma8/norm)**2
        norm = broadcast_to(norm, self.k_s.shape)
        transfer =  get_Pk_EH(Om0, cosmo=self.cosmo, z=0, k=self.k_s)**.5*norm**.5/self.pm.BoxSize.prod()**.5
        digitizer = fastpm.apply_digitized.isotropic_wavenumber(self.k_s)
        rhok= fastpm.apply_digitized(x=rhok, tf=transfer, digitizer=digitizer, kind='wavenumber', mode='amplitude')

        #pt     = self.pt
        stages = self.stages
        q      = self.q
        FactoryCache = fastpm.CosmologyFactory()
        #First step work around
        p, dx = self.first_step_workaround(rhok, Om0, stages, q, FactoryCache)

        zero_map = Literal(self.mappm.create('real', value=0.))
        kmaps  = [zero_map for zs in self.zs]
        f, potk= fastpm.gravity(dx, self.q, self.fpm)
        jj = 0 #counting steps for saving snapshots
        for ai, af in zip(stages[:-1], stages[1:]):
            if self.params['logging']:   
                self.logger.info('fastpm step, %d'%jj)
            # central scale factor
            ac = (ai * af) ** 0.5

            # kick (update momentum)
            kick = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.KickFactor(Om0, support, FactoryCache, ai, ai, ac), epsilon=1e-3, mode='central')
            kick = kick*1.5*Om0
            kick = broadcast_to(kick*1.5*Om0,self.q.shape)
            dp = f * kick
            p  = p + dp

            # drift (update positions)
            drift = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.DriftFactor(Om0, support, FactoryCache, ai, ac, af), epsilon=1e-3, mode='central')
            drift = broadcast_to(drift,self.q.shape) 
            ddx = p * drift
            dx  = dx + ddx

            if self.params['PGD']:
                alpha    = self.alpha0*af**self.mu
                dx_PGD   = PGD.PGD_correction(q+dx, alpha, self.kl, self.ks, self.fpm,q)
            else:
                dx_PGD   = 0.
            if self.params['save3D'] or self.params['save3Dpower']:
                zf       = 1./af-1.
                zi       = 1./ai-1.
                if zi<self.params['zs_source']:
                    pos_raw  = dx+q
                    pos      = pos_raw+dx_PGD
                    stdlib.watchpoint(pos_raw,lambda pos, ii=jj,zi=zi,zf=zf,params=self.params: save_snapshot(pos,ii,zi,zf,params,'raw'))
                    stdlib.watchpoint(pos,lambda pos, ii=jj,zi=zi,zf=zf,params=self.params: save_snapshot(pos,ii,zi,zf,params,'PGD'))

            jj+=1

            kmaps_ = self.no_interp(dx, Om0, p, dx_PGD, ai, af, jj, kmaps=ListPlaceholder(len(self.zs)))#[Symbol('kmaps-%d-%d'%(ii,jj)) for ii in range(len(self.ds))])

            for ii in range(len(self.zs)):
                kmaps[ii] = kmaps[ii]+kmaps_[ii]

            # force (compute force)
            f, potk = fastpm.gravity(dx, self.q, self.fpm)

            # kick (update momentum)
            kick = finite_operator(Om0, lambda Om0, support=self.support, FactoryCache=FactoryCache, ai=ai, af=af, ac=ac:
                                   fastpm.KickFactor(Om0, support, FactoryCache, ac, af, af), epsilon=1e-3, mode='central')
            kick = broadcast_to(kick*1.5*Om0,self.q.shape)
            dp = f * kick
            p  = p + dp


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
    rhok      = pm.generate_whitenoise(seed=randseeds[num], unitary=False, type='complex')

    #set zero mode to zero
    rhok.csetitem([0, 0, 0], 0)

    kedges, k_s = get_kedges(rhok)
    rho = rhok.c2r()
    if params['logging']:
        logging.info('simulations starts')
    # weak lensing simulation object
    wlsim     = WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params, k_s=k_s, kedges=kedges, logger=a_logger)

    #build
    if params['interpolate']:
        model     = wlsim.run_interpolated.build()
    else:
        model     = wlsim.run.build()

    # results
    kmap_vjp,kmap_jvp = [None, None]
    # compute
    if params['forward'] and (params['vjp'] or params['jvp']):
        kmaps, tape       = model.compute(vout='kmaps', init=dict(rho=rho, Om0=cosmo.Omega0_cdm, sigma8=cosmo.sigma8), return_tape=True)
        if params['vjp']:
            vjp         = tape.get_vjp()
            kmap_vjp    = vjp.compute(init=dict(_kmaps=np.ones_like(kmaps[0].value)), vout=['_rho', '_Om0', '_sigma8' ])
        if params['jvp']:
            
            jvp      = tape.get_jvp()
            kmap_jvp = jvp.compute(init=dict(rho_=rho, Om0_=np.array([cosmo.Omega0_cdm]), sigma8_=np.array([cosmo.sigma8])), vout=['kmaps_'])
    else:
        kmaps    = model.compute(vout='kmaps', init=dict(rho=rho, Om0=cosmo.Omega0_cdm, sigma8=cosmo.sigma8))

    return kmaps, kmap_vjp, kmap_jvp, pm
