import numpy as np
from MADLens.testing import BaseVectorTest
from nbodykit.cosmology import Planck15
from MADLens import lightcone
from vmad.lib.fastpm import FastPMSimulation, ParticleMesh
import vmad.lib.fastpm as fastpm
import vmad.lib.linalg as linalg
from mpi4py import MPI
import json
import numpy
import os
import scipy

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
    params['BoxSize'] = [4,4,4]
    params['N_maps'] = 1
    params['Nmesh2D'] = [8,8]
    params['BoxSize2D']=[6.37616/2.]*2
    params['N_steps']=11
    params['custom_cosmo']=False
    params['Omega_m']=0.3089
    params['sigma_8']=0.8158
    params['PGD']=True
    params['B']=2
    params['zs_source']=['0.02']
    params['interpolate']=True
    params['debug']=True
    params['save3D']=False
    params['save3Dpower']= False
    params['PGD_path']=os.path.join('/home/nessa/Documents/codes/MADLens/','pgd_params/')
    return params

def set_up():
    params    = get_params()
    pm        = fastpm.ParticleMesh(Nmesh=params['Nmesh'], BoxSize=params['BoxSize'],comm=MPI.COMM_WORLD, resampler='cic')
    # generate initial conditions
    cosmo     = Planck15.clone(P_k_max=30)
    x         = pm.generate_uniform_particle_grid(shift=0.1)
    BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]
    sim       = lightcone.WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params)

    return pm, cosmo, x, sim
   

class Test_rotate(BaseVectorTest):
    
    x = np.arange(3*100).reshape((100,3))                                     
    M = np.ones((3,3))                        
    y = np.einsum('ij,kj->ki', M, x)   

    def model(self,x):
        M = np.ones((3,3))
        y = linalg.einsum('ij, kj->ki', [M,x])     
        return y

class Test_rotate2(BaseVectorTest):

    pm, _, x, _ = set_up()
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
    pm, cosmo, x, sim = set_up()
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
 

