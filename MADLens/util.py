import numpy as np
import scipy
import json
import os
import pickle
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

def get_Cell(ells,z_source,cosmo,z_chi_int,pm,k_min=None,k_max=None,pk=True,SN=None):
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
                if pk:
                    Pk = cosmo.get_pk(k,z)
                elif shotnoise:
                    Pk = SN
            else:
                Pk = 0.
            # use Limber approximation
            integrand+=[W_lens(chi,chi_source)**2/chi**2*Pk*(1.+z)**2]
        
        result+=[np.trapz(integrand, chis)]
    return np.asarray(result,dtype=float)*factor**2

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

def save3Dpower(mesh,ii,zi,zf,params,name):
    power   = FFTPower(mesh, mode='1d')
    Nyquist = np.pi*mesh.Nmesh[0]/mesh.BoxSize[0]
    if mesh.pm.comm.rank==0:
        pickle.dump([zi,zf,power,Nyquist],open(os.path.join(params['snapshot_dir'],'power_%s_%d.pkl'%(name,ii)),'wb'))
    return True

def save_snapshot(pos,ii,zi,zf,params,name):
    cat       = ArrayCatalog({'Position' : pos}, BoxSize=params['BoxSize'])
    mesh      = cat.to_mesh(Nmesh=params['Nmesh']*4, interlaced=True, compensated=True, window='cic')
    #compute shotnoise with cat.weight? Use gather?
    if params['save3D']:
        mesh.save(os.path.join(params['snapshot_dir'],'%s_%d'%(name,ii)))
    if params['save3Dpower']:
        save3Dpower(mesh,ii,zi,zf,params,name)
    return True


def lowpass_transfer(r):
    def filter(k, v):
        k2 = sum(ki ** 2 for ki in k)
        return np.exp(-0.5 * k2 * r**2) * v
    return filter

def get_fov(cosmo,BoxSize,z_source):
    """
    get the field of view (in degrees) for given boxsize and source redshift
    """
    chi_source = cosmo.angular_diameter_distance(z_source)*(1+z_source)
    fov        = BoxSize[0:2]/chi_source/np.pi*180.
    return fov


def downsample_map(x,desired_pixel_num,params):
    fov     = params['BoxSize2D'][0]
    pix_size= fov/params['Nmesh2D'][0]
    num_pix = np.cast['int32'](np.round((fov/(pix_size))))
    new_pm  = ParticleMesh(BoxSize=[fov/180.*np.pi]*2, Nmesh=[num_pix]*2)

    new_map = new_pm.create(type='real',value=x)
    new_pm  = ParticleMesh(BoxSize=[fov/180.*np.pi]*2, Nmesh=[desired_pixel_num]*2, resampler='cic')
    new_map = new_pm.downsample(new_map,resampler='cic',keep_mean=True)
    
    return new_map




class Run():
    """
    class that holds results of a single run
    """
    def __init__(self, githash, label, rnum, local_path, alter_path=None):
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
        if alter_path is None:
            path_name   = os.path.join(self.params['output_path'],self.params['label']+'%d/'%rnum)
        else:
            path_name   = os.path.join(alter_path,self.params['label']+'%d/'%rnum)
        
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
        print('Loading run with BoxSize %d, Resolution %d, SourceRedshift %.2f, PGD %s and interpolation %s.'%(self.params['BoxSize'][0], self.params['Nmesh'][0], self.params['zs_source'][0], str(self.params['PGD']), str(self.params['interpolate'])))
    
        # count how many maps have been dumped
        NN = len(os.listdir(self.dirs['maps']))
        if NN<self.params['N_maps']:
            print('less maps produces than requested. Requested:%d Produced:%d'%(self.params['N_maps'],NN))
        self.N_maps = NN

        self.Nyquist_3D = np.pi*self.pm.Nmesh[0]/self.pm.BoxSize[0]
        self.Nyquist_2D = np.pi*self.pm2D.Nmesh[0]/self.pm2D.BoxSize[0]
        
    
    def fill_cl_dicts(self,downsample=True):
        """
        fill cl dictionary with results for all source redshifts in this run
        """
        for zs in self.params['zs_source']:
            self.get_measured_cls(zs,downsample)
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

        res = get_Cell(cosmo=self.cosmo,ells=bink,z_source=z_source, z_chi_int=self.z_chi_int, pm=self.pm, pk=False, SN=self.pm.BoxSize.prod()/self.pm.Nmesh.prod())
        
        self.theory_cls[str(z_source)]['SN'] = res

        return True
    
    def get_measured_cls(self,z_source,downsample=True):
        """
        loads measured clkk for given source redshift 
        z_source: float, source redshift
        """

        try:
            assert(z_source in self.params['zs_source'])
        except:
            raise ValueError('%.1f not in '%z_source, self.params['zs_source'])
            
        clkks = []
        for num in range(self.N_maps):
            kappa_map = self.get_map(z_source,num)
            if downsample:
                kappa_map=downsample_map(kappa_map,self.params['Nmesh2D'][0]//2,self.params)
            L, clkk, N = get_2Dpower(kappa_map)
            clkks.append(clkk)
            
        self.measured_cls[str(z_source)]={}
        self.measured_cls[str(z_source)]['L'] = L
        self.measured_cls[str(z_source)]['clkk'] = np.mean(clkks, axis=0)
        self.measured_cls[str(z_source)]['clkk_std'] = np.std(clkks, axis=0)
        self.measured_cls[str(z_source)]['N'] = N
        self.measured_cls[str(z_source)]['SN']= self.pm2D.BoxSize.prod()/self.pm2D.Nmesh.prod()
        
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
            assert(num<self.N_maps)
        except:
            raise ValueError('%d map was not computed'%num)
        
        map_file  = os.path.join(self.dirs['maps'],'map_decon_zsource%d_map%d_of%d'%(z_source*10,num,self.params['N_maps'])+'.npy')
        kappa_map = np.load(map_file).reshape(*self.pm2D.Nmesh)
        kappa_map = self.pm2D.create(type='real',value=kappa_map)
        
        return kappa_map

    def get_Pks(self):
        k_max      = max(20.*np.pi*(self.pm.Nmesh.max()/self.pm.BoxSize.min()),100.)
        cosmo      = self.cosmo.clone(P_k_max=max(k_max*2.,200), perturb_sampling_stepsize=0.01,nonlinear=True)
        self.halofit = {}
        self.lin = {}
        self.pks = {}
        self.pks_PGD = {}      
        for pw in os.listdir(self.dirs['snapshots']):
             if 'power' in pw:
                _ ,zf,power = pickle.load(open(os.path.join(self.dirs['snapshots'],pw),'rb'))
                if 'raw' in pw:
                    self.pks[str(zf)]= power
                    self.halofit[str(zf)] = cosmo.get_pk(self.pks[str(zf)].power['k'],zf)
                    self.lin[str(zf)] = self.cosmo.get_pk(self.pks[str(zf)].power['k'],zf)
                else:
                    self.pks_PGD[str(zf)]=power

        return True 
