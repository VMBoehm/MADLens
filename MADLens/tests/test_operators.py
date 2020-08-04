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
    params_file='/home/nessa/Documents/codes/MADLens/runs/88b093c/deriv_run0.json'
    with open(params_file, 'r') as f:
        params = json.load(f)

    pm        = fastpm.ParticleMesh(Nmesh=params['Nmesh'], BoxSize=params['BoxSize'],comm=MPI.COMM_WORLD, resampler='cic')
    BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]

    # generate initial conditions
    cosmo     = Planck15.clone(P_k_max=30)
    rho       = pm.generate_whitenoise(seed=783645, unitary=False, type='complex')
    rho       = rho.apply(lambda k, v:(cosmo.get_pklin(k.normp(2) ** 0.5, 0) / pm.BoxSize.prod()) ** 0.5 * v)
    #set zero mode to zero
    rho.csetitem([0, 0, 0], 0)

    wlsim     = lightcone.WLSimulation(stages = numpy.linspace(0.1, 1.0, params['N_steps'], endpoint=True), cosmology=cosmo, pm=pm, boxsize2D=BoxSize2D, params=params)
    x         = wlsim.q 
    a_ini     = 0.98
    a_fin     = 1.
    d_ini,d_fin = cosmo.comoving_distance(1./np.array([a_ini,a_fin])-1.)
    Ms        = wlsim.imgen.generate(d_ini,d_fin)
    M         = Ms[0]
    y         = np.einsum('ij,kj->ki', (M, x))
    d         = y[:,2]
    xy        = y[:,:2] 

    return x,y,M,wlsim
   

#class Test_rotate(BaseVectorTest):
#    
#    x, y, M, wlsim = get_stuff()
#
#    def model(self,x):
#        res = self.wlsim.rotate(x,self.M, [0.,0.,0.])
#        return res






