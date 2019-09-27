import numpy as np
from vmad.lib.fastpm import ParticleMesh
import scipy

def get_Cell(ells,z_source,cosmo,z_chi_int,pm=None,k_min=None,k_max=None,shotnoise=False):
    
    if shotnoise:
        #not strictly correct
        n = pm.Nmesh.prod()/pm.BoxSize.prod()
    else:
        n = None

    if k_max == None:     
        k_max      = 2.*np.pi*(pm.Nmesh.max()/pm.BoxSize.min())
    if k_min == None:
        k_min      = 2.*np.pi*(1./pm.BoxSize.max())
        
    factor     = 3./2.*cosmo.Omega0_m*(cosmo.H0/cosmo.C)**2 
    cosmo      = cosmo.clone(P_k_max=400, perturb_sampling_stepsize=0.01,nonlinear=True)
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





from pmesh.pm import RealField
def get_2Dpower(x,dk= None, kmin=None):

    if isinstance(x,RealField):
        x = x.r2c()
    if dk is None:
        dk = 2 * np.pi / min(x.BoxSize)*2
    if kmin is None:
        kmin = dk
    kedges  = np.arange(kmin, np.pi*min(x.Nmesh)/max(x.BoxSize) + dk/2, dk)
    kedges  = np.append(kedges,2*np.pi*min(x.Nmesh)/max(x.BoxSize)+dk)
    kedges  = np.insert(kedges,0,0)
    ind     = np.zeros(x.value.shape, dtype='intp')
    ind     = x.apply(lambda k, v: np.digitize(k.normp(2) ** 0.5, kedges), out=ind)
    weights = (x * np.conj(x)).apply(x._expand_hermitian, kind='index', out=Ellipsis)
    one     = x.pm.create(type(x), value=1).apply(x._expand_hermitian, kind='index', out=Ellipsis)
    Pk      = np.bincount(ind.flat, weights=weights.real.flat, minlength=len(kedges+1))
    N       = np.bincount(ind.flat, weights=one.real.flat, minlength=len(kedges+1))
    Pk      = x.pm.comm.allreduce(Pk)*x.pm.BoxSize.prod()
    N       = x.pm.comm.allreduce(N)
    mask    = np.where(N!=0)
    Pk      = Pk[mask]
    N       = N[mask]
    kedges  = kedges[mask] 
    ks      = kedges[0:-1]-np.diff(kedges)/2.
    return ks[1:-1], Pk[1:-2]/N[1:-2], N[1:-2]

def save_2Dmap(x,filename):
    x_array = np.concatenate(x.pm.comm.allgather(np.array(x.ravel())))
    if x.pm.comm.rank==0:
        np.save(filename,x_array)
    return True

class Run():
    def __init__(self,num,Nmesh,BoxSize,nsteps,fac,PGD,B,z_source,cosmo, var=True, Nmesh2D =None):
        
        self.num       = num
        self.Nmesh     = Nmesh        
        self.BoxSize   = BoxSize
        self.nsteps    = nsteps
        # source redshift as multiple of boxsize
        self.fac       = fac
        # Biwei's correction
        self.PGD       = PGD
        # force resolution
        self.B         = B
        self.var       = var

        self.z_source  = z_source#z_chi_int(BoxSize[-1]*fac)

        self.cosmo     = cosmo
        self.ds        = cosmo.comoving_distance(z_source)
        
        self.label= '%d_%d_%d_%d_%d'%(Nmesh[0],BoxSize[0],nsteps,B,z_source*10)

        if PGD:
            self.label+='_PGD'
        if not var:
            self.label+='_novar'
        print(var,self.label)
        self.label+='_%d'%num
        
        self.pm    = ParticleMesh(Nmesh=Nmesh, BoxSize=BoxSize)
        if Nmesh2D==None:
            Nmesh2D = Nmesh[:2]
        self.pm2D  = ParticleMesh(BoxSize=BoxSize[:2]/self.ds, Nmesh=Nmesh2D, resampler='cic')
        z_int      = np.logspace(-8,np.log10(1500),10000)
        chis       = cosmo.comoving_distance(z_int) #Mpc/h
        self.z_chi_int = scipy.interpolate.interp1d(chis,z_int, kind=3,bounds_error=False, fill_value=0.)
        
    def get_theory_cl(self,bink):
        self.cl = get_Cell(cosmo=self.cosmo,ells=bink,z_source=self.z_source, z_chi_int=self.z_chi_int, pm=self.pm)
        return self.cl
    
    def get_measured_cls(self,cl_path):
        
        try:
            self.bink,self.binpow,self.binpow_decon = np.load(cl_path+'mean_cl_'+self.label+'.npy')
#         except:
#             self.bink,self.binpow,self.binpow_decon, self.binvar = np.load(cl_path+'mean_cl_'+self.label+'.npy')
        except:
            self.bink,self.binpow,self.binpow_decon, self.binvar, self.N = np.load(cl_path+'mean_cl_'+self.label+'.npy')
            
        return self.bink,self.binpow,self.binpow_decon
    
    def get_map(self,map_path,num):
        map_file  = map_path+'map_'+self.label+'_%d'%num+'.npy'
        kappa_map = np.load(map_file).reshape(*self.pm2D.Nmesh)
        kappa_map = self.pm2D.create(type='real',value=kappa_map)
        return kappa_map
