import numpy as np
import scipy
import json
import os
from nbodykit.cosmology import Planck15, LinearPower
from nbodykit.lab import *
from vmad.lib.fastpm import ParticleMesh
from pmesh.pm import RealField, ComplexField


def a2z(a):
    """
    scale factor to redshift
    """
    return 1./a-1

def z2a(z):
    """
    redshift to scale factor
    """
    return 1./(1.+z)

def get_Cell(ells,z_source,cosmo,z_chi_int,pm,k_min=None,k_max=None,shotnoise=False):
    """
    computes clkk from halofit Pk for given 
    ells: 1D array, scales at which to compute clkk
    z_source: float, source_redshift
    cosmo: nbodykit cosmology object
    z_chi_int: function, computes redshift for given comoving distance (must be compatible with cosmo)
    pm: ParticleMesh object (resolution of fastpm simulation, that one wants to compare to) 
    k_min: float (optional), minimal k to use in Pk
    k_max: float (optional), maximal k to use in Pk
    shotnoise, bool (optional), if true, add estimate of shotnoise in the simulation to Pk
    """

    if shotnoise:
        #not strictly correct
        n = pm.Nmesh.prod()/pm.BoxSize.prod()
    else:
        n = None

    if k_max == None:     
        k_max      = max(20.*np.pi*(pm.Nmesh.max()/pm.BoxSize.min()),100.)
    if k_min == None:
        k_min      = 2.*np.pi*(1./pm.BoxSize.max())
        
    factor     = 3./2.*cosmo.Omega0_m*(cosmo.H0/cosmo.C)**2 
    cosmo      = cosmo.clone(P_k_max=max(k_max*2.,200), perturb_sampling_stepsize=0.01,nonlinear=True)
    chi_source = cosmo.comoving_distance(z_source)
    chis       = np.linspace(0.5,chi_source,1000)
    
    def W_lens(chis,chimax):
        return chis*(chimax-chis)/chimax 
    
    result = []
    for l_ in ells:
        integrand=[]
        for chi in chis:
            k = l_/chi #in h/Mpc
            z = z_chi_int(chi)
            if (k>k_min)*(k<k_max):
                Pk = cosmo.get_pk(k,z)
                if shotnoise:
                    #not strictly correct for los
                    Pk+=1./(n)
            else:
                Pk = 0.
            # use Limber approximation
            integrand+=[W_lens(chi,chi_source)**2/chi**2*Pk*(1.+z)**2]
        
        result+=[np.trapz(integrand, chis)]
    return np.asarray(result)*factor**2

def get_2Dpower(x1, x2=None,dk= None, kmin=None):
    """
    computes power spectrum of 2D field
    x1: 2D Particle Mesh field (probably also supports 3D, but 3D is supported by nbodykit)
    x2: 2D Particle Mesh field (optional), if given the cross correlation with x1 is computed
    dk: float (optional), binwidth used for power spectrum
    kmin: float (optional), lowest edge of kbins 
    """


    if isinstance(x2,RealField):
        pass
    elif isinstance(x2,ComplexField):
        pass
    else:
        x2 = x1
        
    if isinstance(x1,RealField):
        x1 = x1.r2c()
    if isinstance(x2,RealField):
        x2 = x2.r2c()

    assert(np.all(x1.Nmesh==x2.Nmesh))
    assert(np.all(x1.BoxSize==x2.BoxSize))

    if dk is None:
        dk = 2 * np.pi / min(x1.BoxSize)*2
    if kmin is None:
        kmin = dk
        
    kedges  = np.arange(kmin, np.pi*min(x1.Nmesh)/max(x1.BoxSize) + dk/2, dk)
    kedges  = np.append(kedges,2*np.pi*min(x1.Nmesh)/max(x1.BoxSize)+dk)
    kedges  = np.insert(kedges,0,0)
    ind     = np.zeros(x1.value.shape, dtype='intp')
    ind     = x1.apply(lambda k, v: np.digitize(k.normp(2) ** 0.5, kedges), out=ind)
    weights = (x1 * np.conj(x2)).apply(x1._expand_hermitian, kind='index', out=Ellipsis)
    one     = x1.pm.create(type(x1), value=1).apply(x1._expand_hermitian, kind='index', out=Ellipsis)
    Pk      = np.bincount(ind.flat, weights=weights.real.flat, minlength=len(kedges+1))
    N       = np.bincount(ind.flat, weights=one.real.flat, minlength=len(kedges+1))
    Pk      = x1.pm.comm.allreduce(Pk)*x1.pm.BoxSize.prod()
    N       = x1.pm.comm.allreduce(N)
    mask    = np.where(N!=0)
    Pk      = Pk[mask]
    N       = N[mask]
    kedges  = kedges[mask] 
    ks      = kedges[0:-1]-np.diff(kedges)/2.
    return ks[1:-1], Pk[1:-2]/N[1:-2], N[1:-2]


def save_2Dmap(x,filename):
    """
    dumps a 2D map that is distributed in memory
    """
    x_array = np.concatenate(x.pm.comm.allgather(np.array(x.ravel())))
    if x.pm.comm.rank==0:
        np.save(filename,x_array)
    return True


class Run():
    """
    class that holds results of a single run
    """
    def __init__(self, githash, label, rnum, local_path):
        """
        loads the parameter file of the run
        githash: string, abridged githash of commit under which the run was performed
        label  : string, label of the run
        rnum   : int, number of run under this label and githash
        local_path: string, path under which parameter files have been stored 
        """
        
        #-------------------------------------------------------------#
        params_path  = os.path.join(local_path,'runs',githash)
        params_file  = os.path.join(params_path,label+'%d.json'%rnum)
        with open(params_file, 'r') as f:
            self.params = json.load(f)
            
        path_name   = os.path.join(self.params['results_path'],self.params['label']+'%d/'%rnum)
        self.dirs = {}
        for result in ['cls','maps','snapshots']:
            self.dirs[result] = os.path.join(path_name,result)
        #-------------------------------------------------------------#
            
        cosmo      = Planck15.match(Omega0_m=self.params['Omega_m'])
        self.cosmo = cosmo.match(sigma8=self.params['sigma_8'])
        
        self.pm    = ParticleMesh(Nmesh=self.params['Nmesh'], BoxSize=self.params['BoxSize'], resampler='cic')
    
        BoxSize2D  = [deg/180.*np.pi for deg in self.params['BoxSize2D']]
        self.pm2D  = ParticleMesh(BoxSize=BoxSize2D, Nmesh=self.params['Nmesh2D'],resampler='cic')
        
        z_int      = np.logspace(-8,np.log10(1500),10000)
        chis       = cosmo.comoving_distance(z_int) #Mpc/h
        self.z_chi_int = scipy.interpolate.interp1d(chis,z_int, kind=3,bounds_error=False, fill_value=0.)
        
        self.theory_cls   = {}
        self.measured_cls = {}
        
    def fill_cl_dicts(self):
        """
        fill cl dictionary with results for all source redshifts in this run
        """
        for zs in self.params['zs_source']:
            self.get_measured_cls(zs)
            self.get_theory_cl(self.measured_cls[str(zs)]['L'],zs)
            
        return True
        
    def get_theory_cl(self,bink,z_source):
        """
        compute theory clkk with halofit
        """
        try:
            assert(z_source in self.params['zs_source'])
        except:
            raise ValueError('%.1f not in '%z_source, self.params['zs_source'])
            
        res = get_Cell(cosmo=self.cosmo,ells=bink,z_source=z_source, z_chi_int=self.z_chi_int, pm=self.pm)
        
        self.theory_cls[str(z_source)] = {}
        self.theory_cls[str(z_source)]['L'] = bink
        self.theory_cls[str(z_source)]['clkk'] = res 
        
        return True
    
    def get_measured_cls(self,z_source):
        """
        loads measured clkk for given source redshift 
        z_source: float, source redshift
        """
        try:
            assert(z_source in self.params['zs_source'])
        except:
            raise ValueError('%.1f not in '%z_source, self.params['zs_source'])
            
        clfile = os.path.join(self.dirs['cls'],'mean_cl_zsource%d_averaged_over_%dmaps.npy'%(z_source*10,self.params['N_maps']))
        res    = np.load(clfile)
        
        self.measured_cls[str(z_source)]={}
        for ii, tag in enumerate(['L','clkk','clkk_std','N']):
            self.measured_cls[str(z_source)][tag] = res[ii]
        
        return True
    
    def get_map(self,z_source,num):
        
        """
        z_source: float, source redshift
        num: integer, which of the N_maps to extract
        """
        
        try:
            assert(z_source in self.params['zs_source'])
        except:
            raise ValueError('%.1f not in '%z_source, self.params['zs_source'])
        
        try:
            assert(num<self.params['N_maps'])
        except:
            raise ValueError('%d map was not computed'%num)
        
        map_file  = os.path.join(self.dirs['maps'],'map_decon_zsource%d_map%d_of%d'%(z_source*10,num,self.params['N_maps'])+'.npy')
        kappa_map = np.load(map_file).reshape(*self.pm2D.Nmesh)
        kappa_map = self.pm2D.create(type='real',value=kappa_map)
        
        return kappa_map




