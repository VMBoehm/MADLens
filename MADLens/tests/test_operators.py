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

def get_stuff():

    pm        = fastpm.ParticleMesh(Nmesh=[4,4,4], BoxSize=[256,256,256],comm=MPI.COMM_WORLD, resampler='cic')

    # generate initial conditions
    cosmo     = Planck15.clone(P_k_max=30)
    x         = pm.generate_uniform_particle_grid(shift=0.5)
    x         = np.arange(3*100).reshape((100,3))
    M         = np.asarray([[1,0,0],[0,0,1],[0,1,0]]) 
    y         = np.einsum('ij,kj->ki', M, x)

    return x,y
   

class Test_rotate(BaseVectorTest):
    
    x = np.arange(3*100).reshape((100,3))                                     
    M = np.ones((3,3))                        
    y = np.einsum('ij,kj->ki', M, x)   

    def model(self,x):
        M = np.ones((3,3))
        y = linalg.einsum('ij, kj->ki', [M,x])     
        return y






