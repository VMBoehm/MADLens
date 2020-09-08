import numpy as np
from MADLens.testing import BaseVectorTest, BaseScalarTest
from nbodykit.cosmology import Planck15
from MADLens import lightcone
from vmad.lib.fastpm import FastPMSimulation as vmadPMSimulation
from vmad.lib.fastpm import ParticleMesh as vmadParticleMesh
import vmad.lib.fastpm as vmadfastpm
from pmesh.pm import ParticleMesh
import fastpm 
import vmad.lib.linalg as linalg
from mpi4py import MPI
import json
import numpy
import os
import scipy
from pmesh.pm import RealField, ComplexField
from vmad.core import stdlib

def create_bases(x):
    bases = numpy.eye(x.size).reshape([-1] + list(x.shape))
    if isinstance(x, RealField):
        pm = x.pm
        # FIXME: remove this after pmesh 0.1.36
        def create_field(pm, data):
            real = pm.create(type='real')
            real[...] = data
            return real
        return [create_field(pm, i) for i in bases]
    else:
        return [i for i in bases]

class Test_list_elem(BaseVectorTest):
    i = 1
    x = np.ones((2, 2, 2))
    y = x[i]
    def model(self, x):
        elem = lightcone.list_elem(x, i=self.i)
        return elem


class Test_list_put_3d(BaseVectorTest):
    elems = np.ones((5,2,2))*2
    i     = 2
    x_    = np.ones((5,2,2))    
    x_[i] = elems[0]
    x     = np.stack((x_,elems))
    y     = x_

    def model(self, x):
        x_      = linalg.take(x,0,axis=0)
        elems   = linalg.take(x,1,axis=0)
        elem    = linalg.take(elems,0,axis=0)
        res     = lightcone.list_put(x_, elem, self.i)
        return res


class Test_list_put_2d(BaseVectorTest):
    elems = np.ones((5,2))*2
    i     = 2
    x_    = np.ones((5,2))    
    x_[i] = elems[0]
    x     = np.stack((x_,elems))
    y     = x_

    def model(self, x):
        x_      = linalg.take(x,0,axis=0)
        elems   = linalg.take(x,1,axis=0)
        elem    = linalg.take(elems,0,axis=0)
        res     = lightcone.list_put(x_, elem, self.i)
        return res


class Test_chi_z(BaseVectorTest):

    x = np.array([0.5,1.0,1.5,2.])
    cosmo = Planck15
    y = cosmo.comoving_distance(x)

    def model(self, x):
        res = lightcone.chi_z(x,self.cosmo)
        return res


def get_params():
    params={}
    params['Nmesh'] = [4,4,4]
    params['BoxSize'] = [64,64,64]
    params['N_maps'] = 1
    params['Nmesh2D'] = [8,8]
    params['BoxSize2D']=[6.37616/2.]*2
    params['N_steps']=11
    params['custom_cosmo']=False
    params['Omega_m']=0.3089
    params['sigma_8']=0.8158
    params['PGD']=False
    params['B']=2
    params['zs_source']=['0.2']
    params['interpolate']=True
    params['debug']=True
    params['save3D']=False
    params['save3Dpower']= False
    params['PGD_path']=os.path.join('/home/nessa/Documents/codes/MADLens/','pgd_params/')
    return params

def set_up():
    params    = get_params()
    pm        = ParticleMesh(Nmesh=params['Nmesh'], BoxSize=params['BoxSize'],comm=MPI.COMM_WORLD, resampler='cic')
    # generate initial conditions
    cosmo     = Planck15.clone(P_k_max=30)
    x         = pm.generate_uniform_particle_grid(shift=0.1)
    BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]
    sim       = lightcone.WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params)
    kmaps = [sim.mappm.create('real', value=0.) for ii in range(1)]

    return pm, cosmo, x, kmaps, sim.DriftFactor, sim.mappm, sim
   

class Test_rotate(BaseVectorTest):
    
    x = np.arange(3*100).reshape((100,3))                                     
    M = np.ones((3,3))                        
    y = np.einsum('ij,kj->ki', M, x)   

    def model(self,x):
        M = np.ones((3,3))
        y = linalg.einsum('ij, kj->ki', [M,x])     
        return y

class Test_rotate2(BaseVectorTest):

    pm, _, x, _, _, _, _ = set_up()
    boxshift  = 0.5    
    M         = np.asarray([[1,0,0],[0,0,1],[0,1,0]]) 
    y         = np.einsum('ij,kj->ki', M, x)+pm.BoxSize*boxshift
    
    #y = y[:,2]
    y = y[:,0:2]

    def model(self, x):
        y  = linalg.einsum('ij,kj->ki', (self.M, x))     
        y  = y + self.pm.BoxSize * self.boxshift
        d  = linalg.take(y, 2, axis=1)
        xy = linalg.take(y, (0, 1), axis=1)
        return xy#d

class Test_wlen(BaseVectorTest):
    pm, cosmo, x, _, _, _, sim = set_up()
    x               = x[:,2]
    z               = sim.z_chi_int(x)
    ds              = sim.ds[0]
    columndens      = sim.nbar*sim.A*x**2 #particles/Volume*angular pixel area* distance^2 -> 1/L units
    y               = (ds-x)*x/ds*(1.+z)/columndens #distance
    def model(self,x):
        res = self.sim.wlen(x,self.ds)
        return res


class Test_z_chi(BaseVectorTest):

    x              = np.linspace(100,1000)
    z_int          = np.logspace(-8,np.log10(1500),10000)
    chis           = Planck15.comoving_distance(z_int) #Mpc/h
    z_chi_int = scipy.interpolate.interp1d(chis,z_int, kind=3,bounds_error=False, fill_value=0.)
    y              = z_chi_int(x)

    def model(self,x):
        y = lightcone.z_chi(x,Planck15,self.z_chi_int)
        return y
 

class Test_makemap1(BaseScalarTest):
    
    to_scalar = staticmethod(vmadfastpm.to_scalar)

    pm = ParticleMesh(Nmesh=[4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)
    #x  = pm.generate_uniform_particle_grid(shift=0.5)
    #x  = x[:1,:]
    x  = np.array([[3.,3.]],dtype=np.float64) # particle cannot be on grid point
    xx = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    y  = NotImplemented
    w  = np.ones(len(x))
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
            
        compensation = self.pm.resampler.get_compensation()
        layout       = vmadfastpm.decompose(x, self.pm)
        map          = vmadfastpm.paint(x, self.w, layout, self.pm)
        y            = map+self.xx # bias needed to avoid zero derivative
        # compensation for cic window
        c            = vmadfastpm.r2c(y)
        c            = vmadfastpm.apply_transfer(c, lambda k : compensation(k, 1.0), kind='circular')
        map          = vmadfastpm.c2r(c)
         
        return map

class Test_makemap2(BaseScalarTest):
    
    to_scalar = staticmethod(vmadfastpm.to_scalar)

    pm = ParticleMesh(Nmesh=[4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)
    #x  = pm.generate_uniform_particle_grid(shift=0.5)
    #x  = x[:1,:]
    w  = np.array([[3.,3.]],dtype=np.float64) # particle cannot be on grid point
    xx = pm.generate_whitenoise(seed=300, unitary=True, type='real')
    y  = NotImplemented
    x  = np.ones(len(w))
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
            
        compensation = self.pm.resampler.get_compensation()
        layout       = vmadfastpm.decompose(self.w, self.pm)
        map          = vmadfastpm.paint(self.w, x, layout, self.pm)
        y            = map+self.xx # bias needed to avoid zero derivative
        # compensation for cic window
        c            = vmadfastpm.r2c(y)
        c            = vmadfastpm.apply_transfer(c, lambda k : compensation(k, 1.0), kind='circular')
        map          = vmadfastpm.c2r(c)
         
        return map


class Test_interp(BaseScalarTest):

    to_scalar = staticmethod(vmadfastpm.to_scalar)
    
    pm, cosmo, q, kmaps, DriftFactor, mappm ,sim = set_up()
    y  = NotImplemented
    ai = np.asarray([0.5])
    af = np.asarray([1.0])
    ac = (ai * af) ** 0.5
    p  = np.asarray([0.01,0.05,0.02])
    dx_PGD = 0.
    x  = np.asarray([[0.5,3.0,1.2]])
    xx = mappm.generate_whitenoise(seed=300, unitary=True, type='real')
    x_ = create_bases(x)
    epsilon = 1e-3

    def model(self, x):
        di, df = self.cosmo.comoving_distance(1. / numpy.array([self.ai, self.af]) - 1.)
        for M in self.sim.imgen.generate(di, df):
            # if lower end of box further away than source -> do nothing
            if df>self.sim.max_ds:
                continue
            else:
                M, boxshift = M

                # positions of unevolved particles after rotation
                d_approx = self.sim.rotate.build(M=M, boxshift=boxshift).compute('d', init=dict(x=self.q))
                z_approx = lightcone.z_chi.apl.impl(node=None,cosmo=self.cosmo,z_chi_int=self.sim.z_chi_int,chi=d_approx)['z']
                a_approx = 1. / (z_approx + 1.)
                
                # move particles to a_approx, then add PGD correction
                
                dx1      = x + self.p*self.DriftFactor(a_approx, self.ac, self.ac)[:, None] + self.dx_PGD
                # rotate their positions
                xy, d    = self.sim.rotate((dx1 + self.q)%self.pm.BoxSize, M, boxshift)

                # projection
                xy       = (((xy - self.pm.BoxSize[:2]* 0.5))/ linalg.stack((d,d), axis=-1)+ self.sim.mappm.BoxSize * 0.5 )

                for ii, ds in enumerate(self.sim.ds):
                    w        = self.sim.wlen(d,ds)
                    mask     = stdlib.eval(d, lambda d, di=di, df=df, ds=ds, d_approx=d_approx: 1.0 * (d_approx < di) * (d_approx >= df) * (d <=ds))
                    #stdlib.watchpoint(mask, lambda f: print(f))
                    kmap     = lightcone.list_elem(self.kmaps,ii)
                    kmap_    = self.sim.makemap(xy, w*mask)#+self.xx
                    kmap     = linalg.add(kmap_,kmap)
                    self.kmaps = lightcone.list_put(self.kmaps,kmap,ii)
        #stdlib.watchpoint(mask, lambda f: print(f)) 
        #stdlib.watchpoint(d, lambda d, d_approx=d_approx, di=di, df=df, ds=ds: print(d_approx,di,df,ds))  
        kmap_ = lightcone.list_elem(self.kmaps,0)
        #stdlib.watchpoint(kmap_, lambda f: print(f)) 
        return kmap_#lightcone.list_elem(self.kmaps,0)


