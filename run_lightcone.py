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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

FLAGS = flags.FLAGS
flags.DEFINE_string('output_path',os.path.join(os.getcwd(),'results/'), "path for storing results")
flags.DEFINE_string('PGD_path',os.path.join(os.getcwd(),'pgd_params/'),"path to the PGD parameter files")
flags.DEFINE_integer('N_maps',1,'number of maps to produce at each source redshift')
flags.DEFINE_float('boxsize',512.,'size of the simulation box in Mpc/h')
flags.DEFINE_integer('Nmesh',1024,'resolution of fastPM mesh')
flags.DEFINE_integer('Nmesh2D',2048, 'resolution of lensing map')
flags.DEFINE_float('boxsize2D',12.752343510260971,'field of view in degrees (default is optimal for default settings, use FindConfigs.ipynb notebook to find optimal fov for your setting.')
flags.DEFINE_integer('N_steps',40,'number of fastPM steps')
#bounds from KIDS contours, default values from Planck2015
flags.DEFINE_bool('custom_cosmo', False, 'custom cosmology? If true, read in values for sigma8 and Omega_m, otherwise use Plmack15 as default') 
flags.DEFINE_float('Omega_m',0.3089,'total matter density', lower_bound=0.1, upper_bound=0.5)
flags.DEFINE_float('sigma_8',0.8158,'amplitude of matter fluctuations', lower_bound=0.4, upper_bound=1.3)
flags.DEFINE_boolean('PGD',False,'whether to use PGD sharpening')
flags.DEFINE_integer('B',2,'force resolution factor')
flags.DEFINE_spaceseplist('zs_source',['1.0'],'source redshifts')
flags.DEFINE_boolean('interpolate',True,'whether to interpolate between snapshots')
flags.DEFINE_boolean('debug',True,'debug mode allows to run repeatedly with the same settings')
flags.DEFINE_boolean('save3D',False,'whether to dump the snapshots, requires interp to be set to False')
flags.DEFINE_boolean('save3Dpower', False, 'whether to measure and save the power spectra of the snapshots')
flags.DEFINE_boolean('vjp', False,'whether to compute the vjp')
flags.DEFINE_boolean('jvp', False, 'whether to compute the jvp')
flags.DEFINE_boolean('forward',True, 'whether to run forward model')
flags.DEFINE_boolean('analyze',False, 'whether to print out resource usage')
flags.DEFINE_string('label', 'transfer_test', 'label of this run')
flags.DEFINE_boolean('logging', 'False', 'whether to log run or not')

def main(argv):
    del argv

    """ -------------- setting paramaeters ------------------------"""
    params              = FLAGS.flag_values_dict() 
    params['Nmesh']     = [FLAGS.Nmesh]*3
    params['BoxSize']   = [FLAGS.boxsize]*3 
    params['Nmesh2D']   = [FLAGS.Nmesh2D]*2 
    params['BoxSize2D'] = [FLAGS.boxsize2D]*2 
    params['zs_source'] = [float(zs) for zs in FLAGS.zs_source]

    if params['custom_cosmo']:
        cosmo = Planck15.match(Omega0_m=FLAGS.Omega_m)
        cosmo = cosmo.match(sigma8=FLAGS.sigma_8)
    else:
        if rank==0:
            print('custom_cosmo is set to False. Using default cosmology.')
        cosmo = Planck15

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
        print('dumping under githash %s'%githash)
        output_path = os.path.join(FLAGS.output_path,githash)
        params_path  = os.path.join(os.path.join(os.getcwd()),'runs',githash)
        params['output_path'] = output_path
        print(params_path, params['output_path'])
        if not os.path.isdir(params_path):
            os.makedirs(params_path)

        # make sure parameter file name is unique and we are not repeating a run
        num_run = 0
        found   = True
        while found:
            path_name   = os.path.join(output_path,params['label']+'%d/'%num_run)
            params_file = os.path.join(params_path,params['label']+'%d.json'%num_run)
            if not os.path.isdir(path_name):
                os.makedirs(path_name)
                found = False
            if not os.path.isfile(params_file):
                found = False
            else:
                with open(params_file, 'r') as f:
                    old_params = json.load(f)
                    if old_params==params and not params['debug']:
                        raise ValueError('run with same settings already exists: %s'%params_file)
                    elif params['debug']:
                        found = False
                    else:
                        num_run+=1

        for result in ['cls','maps','snapshots']:
            print(path_name)
            dirs[result] = os.path.join(path_name,result)
            if not os.path.isdir(dirs[result]):
                os.makedirs(dirs[result])

        fjson = json.dumps(params)
        f = open(params_file,"w")
        f.write(fjson)
        f.close()

    dirs                  = comm.bcast(dirs, root=0)
    params['snapshot_dir']= dirs['snapshots']

    """---------------------------run actual simulations-----------------------------"""
    sims_start = time.time()
        
    for ii in range(FLAGS.N_maps):
        if rank==0:
            print('progress in percent:', ii/params['N_maps']*100)
        kmaps, kmaps_deriv, pm = run_wl_sim(params,cosmo=cosmo, num=ii)

        for jj,z_source in enumerate(params['zs_source']):
            kmap    = kmaps[jj]
            mapfile = os.path.join(dirs['maps'],'map_decon_zsource%d_map%d_of%d'%(z_source*10,ii,params['N_maps'])+'.npy')
            save_2Dmap(kmap,mapfile)
            if rank==0:
                print('2D map #%d at z_s=%.1f dumped to %s'%(ii,z_source,mapfile))
            
    end = time.time()
    if rank==0:
        print('time taken per sim in sec %d'%((end-sims_start)/(params['N_maps']*len(params['zs_source']))))
        print('time takes before sims in sec %d'%(sims_start-start))

if __name__ == '__main__':
  app.run(main)

