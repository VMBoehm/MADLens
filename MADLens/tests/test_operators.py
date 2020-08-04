import numpy as np
from MADLens.testing import BaseVectorTest
from nbodykit.cosmology import Planck15
from MADLens import lightcone

class Test_list_elem(BaseVectorTest):
    i = 1
    x = np.ones((2, 2, 2))
    y = x[i]
    def model(self, x):
        elem = lightcone.list_elem(x, i=self.i)
        return elem

class Test_list_put_3d(BaseVectorTest):
    elem = np.ones((2,2))*2
    i    = 2
    x    = np.asarray([np.ones((2,2)) for ii in range(5)])
    x[i] = elem
    y    = x
    print(y, x)
    def model(self, x):
        res = lightcone.list_put(x, self.elem, self.i)
        return res

class Test_list_put_2d(BaseVectorTest):
    elem = np.ones((2))*2
    i    = 2
    x    = np.asarray([np.ones((2)) for ii in range(5)])
    x[i] = elem
    y    = x
    print(y, x)
    def model(self, x):
        res = lightcone.list_put(x, self.elem, self.i)
        return res



class Test_chi_z(BaseVectorTest):

    x = np.array([0.5,1.0,1.5,2.])
    cosmo = Planck15
    y = cosmo.comoving_distance(x)

    def model(self, x):
        res = lightcone.chi_z(x,self.cosmo)
        return res



#class Test_rotate(BaseVectorTest)        
