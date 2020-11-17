import numpy
from MADLens.testing import BaseVectorTest
from nbodykit.cosmology import LinearPower, Planck15
from vmad.lib import fastpm, linalg
from MADLens import power
import MADLens
import vmad

class Test_get_NWEH_power(BaseVectorTest):
    ks= 1e-1
    redshift = 0
    y =  LinearPower(cosmo=Planck15, redshift=redshift, transfer='NoWiggleEisensteinHu')(ks)
    norm = (Planck15.sigma8/power.normalize(8,Planck15.Omega0_m, 'NWEH'))**2
    y = y/norm
    x = numpy.array([Planck15.Omega0_m])
    
    def model(self, x):
        x = linalg.take(x=x, axis=0, i=0)
        NWEH = power.get_Pk_NWEH(x, cosmo=Planck15, z=self.redshift, k=self.ks)
        return NWEH

class Test_get_EH_power(BaseVectorTest):
    ks= 1
    redshift = 0
    y =  LinearPower(cosmo=Planck15, redshift=redshift, transfer='EisensteinHu')(ks)
    norm = (Planck15.sigma8/power.normalize(8,Planck15.Omega0_m, 'EH'))**2
    y = y/norm
    x = numpy.array([Planck15.Omega0_m])
    
    def model(self, x):
        x = linalg.take(x=x, axis=0, i=0)
        EH = power.get_Pk_EH(x, cosmo=Planck15, z=self.redshift, k=self.ks)
        return EH

class Test_get_omega_z(BaseVectorTest):
    redshift = 0
    y = numpy.array(0.308904)
    x = numpy.array([Planck15.Omega0_m])
    
    def model(self, x):
        x = linalg.take(x=x, axis=0, i=0)
        omega_z = power.get_omega_z(x, z=self.redshift)
        return omega_z


class Test_get_omega_lambda(BaseVectorTest):
    redshift = 0
    y = numpy.array(0.691096)
    x = numpy.array([Planck15.Omega0_m])
    
    def model(self, x):
        x = linalg.take(x=x, axis=0, i=0)
        omega_lambda = power.get_omega_lambda(x, z=self.redshift)
        return omega_lambda



class Test_grow(BaseVectorTest):
    redshift = 0
    y = numpy.array(0.783374)
    x = numpy.array([Planck15.Omega0_m])
    
    def model(self, x):
        x = linalg.take(x=x, axis=0, i=0)
        omega_z = power.get_omega_z(x, z=self.redshift)
        omega_lambda = power.get_omega_lambda(x, z=self.redshift)
        growth = power.grow(omega_z, omega_lambda, z=self.redshift)
        return growth
