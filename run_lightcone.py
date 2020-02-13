import time
start = time.time()
from MADLens.lightcone import run_wl_sim
from nbodykit.cosmology import Planck15
from MADLens.util import get_2Dpower, save_2Dmap
import numpy as np
import scipy
from mpi4py import MPI
from absl import app
from absl import flags
import os
import json
import subprocess
import pickle
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

FLAGS = flags.FLAGS
flags.DEFINE_string('results_path','/global/cscratch1/sd/vboehm/MADLensOutputs/results/', "path for storing results")
flags.DEFINE_string('PGD_path',os.path.join(os.getcwd(),'pgd_params/'),"path to the PGD parameter files")
flags.DEFINE_integer('N_maps',1,'number of maps to produce at each source redshift')
flags.DEFINE_float('boxsize',256,'size of the simulation box in Mpc/h')
flags.DEFINE_integer('Nmesh',256,'resolution of fastPM mesh')
flags.DEFINE_integer('Nmesh2D',4096, 'resolution of lensing map')
#flags.DEFINE_float('boxsize2D',6.37616,'field of view in degrees (default is optimal for default settings, use FindConfigs.ipynb notebook to find optimal fov for your setting.')
flags.DEFINE_integer('N_steps',11,'number of fastPM steps')
#bounds from KIDS contours, default values from Planck2015
#flags.DEFINE_float('Omega_m',0.3089,'total matter density', lower_bound=0.1, upper_bound=0.5)
#flags.DEFINE_float('sigma_8',0.8158,'amplitude of matter fluctuations', lower_bound=0.4, upper_bound=1.3)
flags.DEFINE_boolean('PGD',False,'whether to use PGD sharpening')
flags.DEFINE_integer('B',2,'force resolution factor')
flags.DEFINE_spaceseplist('zs_source',['1.'],'source redshifts')
flags.DEFINE_boolean('interpolate',True,'whether to interpolate between snapshots')
flags.DEFINE_boolean('debug',True,'debug mode allows to run repeatedly with the same settings')
flags.DEFINE_boolean('save3D',False,'whether to dump the snapshots, requires interp to be set to False')
flags.DEFINE_boolean('save3Dpower', False, 'whether to measure and save the power spectra of the snapshots')
flags.DEFINE_enum('mode', 'forward', ['forward','backprop'],'whether to run the forward model only or include backpropagation')
flags.DEFINE_boolean('analyze', False, 'whether to print out resource usage')
#flags.DEFINE_string('label', 'test_run', 'label of this run')
flags.DEFINE_string('parameter_file','S_8_small_run', 'which parameter file to read cosmology and fov from')
old_print = print

def print(*args):
    if rank==0:
        old_print(args)
    return True

def main(argv):
    del argv

    """ -------------- setting paramaeters ------------------------"""
    params              = FLAGS.flag_values_dict() 
    param_file_num      = int(sys.argv[1])   
    _,fov_max,Omega_ms,sigma8s,_,Pk_interp = pickle.load(open(os.path.join('./run_specs',params['parameter_file']+'_%d.pkl'%param_file_num),'rb'))
    print(len(Pk_interp))
    print(param_file_num)	
    for nn, (Omega_m,sigma_8) in enumerate(zip(Omega_ms,sigma8s)):
        params['Nmesh']     = [FLAGS.Nmesh]*3
        params['BoxSize']   = [FLAGS.boxsize]*3 
        params['Nmesh2D']   = [FLAGS.Nmesh2D]*2 
        params['BoxSize2D'] = [fov_max]*2 
        params['zs_source'] = [float(zs) for zs in FLAGS.zs_source]
        params['Omega_m']   = Omega_m
        params['sigma_8']   = sigma_8
        params['label']     = 'small_run'
        map_num             = param_file_num*100+nn
        print(map_num)
        cosmo = Planck15.match(Omega0_m=Omega_m)
        # hoping that this is taken care of by choosing the right interpolated pk
        #cosmo = cosmo.match(sigma8=sigma_8)
    
        if params['save3D'] or params['save3Dpower']:
            try:
                assert(params['interpolate']==False)
            except:
                raise ValueError('interpolate must be set to False if requesting 3D outouts')
    
        """------- setting output dirs and saving parameters-----------"""
        dirs = {} 
        if rank ==0: 
            cmd    = "git log --pretty=format:'%h' -n 1"
            githash= subprocess.run([cmd], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
        
            results_path = os.path.join(FLAGS.results_path,githash)
            params_path  = os.path.join(os.path.join(os.getcwd()),'runs',githash)
            params['results_path'] = results_path
            if not os.path.isdir(params_path):
                os.makedirs(params_path)
    
    
            dirs['maps'] = os.path.join(results_path,'maps',params['label'])
            if not os.path.isdir(dirs['maps']):
                os.makedirs(dirs['maps'])
            if nn==0:
                params_file = os.path.join(params_path,params['label']+'.json')
                fjson = json.dumps(params)
                f = open(params_file,"w")
                f.write(fjson)
                f.close()
    
        dirs    = comm.bcast(dirs, root=0)
    
        """---------------------------run actual simulations-----------------------------"""
        sims_start = time.time()
            
        for ii in range(FLAGS.N_maps):
            print('progress in percent:', ii/params['N_maps']*100)
            kmaps, kmaps_deriv, pm = run_wl_sim(params,cosmo=cosmo, num=map_num, pk = Pk_interp[nn])
    
            for jj,z_source in enumerate(params['zs_source']):
                kmap    = kmaps[jj]
                mapfile = os.path.join(dirs['maps'],'map_decon_zsource%d_cosmo%d'%(z_source*10,map_num)+'.npy')
                save_2Dmap(kmap,mapfile)
                print('2D map #%d at z_s=%.1f dumped to %s'%(ii,z_source,mapfile))
                #bink,binpow,N = get_2Dpower(kmap)
                #len_k = len(bink)
                #if rank ==0:
                #    if ii==0 and jj==0:
                #        binpows        = np.zeros((len(params['zs_source']),params['N_maps'],len_k))
                #    binpows[jj,ii] = binpow
    
        end = time.time()
    
        print('time taken per sim in sec %d'%((end-sims_start)/(params['N_maps']*len(params['zs_source']))))
        print('time takes before sims in sec %d'%(sims_start-start))
    
        """ ------------------------- save average cls ----------------------------------"""
        #if rank==0:
        #    for jj,z_source in enumerate(params['zs_source']):
        #        binpow       = np.mean(binpows[jj], axis=0)
        #        binpow_std   = np.std(binpows[jj], axis=0)
   # 
   #             clfile       = os.path.join(dirs['cls'],'mean_cl_zsource%d_averaged_over_%dmaps.npy'%(z_source*10,params['N_maps']))
   # 
   #             np.save(clfile,[bink,binpow,binpow_std,N])
   #             print('cls dumped to %s'%clfile)

if __name__ == '__main__':
  app.run(main)

