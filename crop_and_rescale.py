import os
import numpy as np
import copy
from MADLens.util import *
import sys

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

def crop_map(mymap,desired_fov,desired_pixel_num,params,cosmo,zs=1):
    fov     = get_fov(cosmo,params['BoxSize'],z_source=zs)
    pix_size= fov/params['Nmesh2D']
    num_pix = np.cast['int32'](np.round((desired_fov/(pix_size))))
    new_pm  = ParticleMesh(BoxSize=[desired_fov/180.*np.pi]*2, Nmesh=num_pix)
    new_map = new_pm.create(type='real',value=mymap.value[:num_pix[0],:num_pix[0]])
    new_pm  = ParticleMesh(BoxSize=[desired_fov/180.*np.pi]*2, Nmesh=[desired_pixel_num]*2, resampler='cic')
    new_map = new_pm.downsample(new_map,resampler='cic',keep_mean=True)
    new_map = new_map.r2c().apply(lowpass_transfer(pix_size[0]/180.*np.pi*4.)).c2r()
    
    return new_map


fov_min,fov_max,Omega_ms,sigma8s, _, _ = pickle.load(open(os.path.join('./run_specs','S_8_small_run_0.pkl'),'rb'))
param_path  = '/global/homes/v/vboehm/codes/MADLens/runs/'
git_hash    = sys.argv[1]
label       = sys.argv[2]
batch       = int(sys.argv[3])

desired_fov = fov_min

params_file = os.path.join(param_path,git_hash,label+'.json')
with open(params_file, 'r') as f:
    params = json.load(f)
map_dir = os.path.join(params['results_path'],'maps',params['label'])
crop_dir= os.path.join(params['results_path'],'cropped')
if not os.path.isdir(crop_dir):
    os.makedirs(crop_dir)
BoxSize2D = [deg/180.*np.pi for deg in params['BoxSize2D']]
pm2D      = ParticleMesh(BoxSize=BoxSize2D, Nmesh=params['Nmesh2D'],resampler='cic')

for ii in np.arange(100):
    map_num   = ii+batch*100
    Omega_m   = Omega_ms[map_num]
    cosmo     = Planck15.match(Omega0_m=Omega_m)
    map_file  = os.path.join(map_dir,'map_decon_zsource%d_cosmo%d'%(params['zs_source'][0]*10,map_num)+'.npy')
    kappa_map = np.load(map_file).reshape(*pm2D.Nmesh)
    kappa_map = pm2D.create(type='real',value=kappa_map)
    kappa_map = crop_map(kappa_map,desired_fov,1024,params,cosmo)
    map_file  = os.path.join(crop_dir,'map_decon_zsource%d_cosmo%d'%(params['zs_source'][0]*10,map_num)+'.npy')
    save_2Dmap(kappa_map,map_file)


