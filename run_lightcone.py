import time
start = time.time()
from MADLens.lightcone import test_wl
from nbodykit.cosmology import Planck15
from MADLens.util import get_2Dpower, save_2Dmap
import numpy as np
import scipy
from mpi4py import MPI
from absl import app
from absl import flags
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

FLAGS = flags.FLAGS


flags.DEFINE_string('cl_path',os.path.join(os.getcwd(),'cls/'), "path for storing lensing cls")
flags.DEFINE_string('map_path',os.path.join(os.getcwd(),'maps/'),'path for storing lensing maps')
flags.DEFINE_integer('N_maps',1,'number of maps to produce at each source redshift')
flags.DEFINE_float('BoxSize',256.,'size of the simulation box in Mpc/h')
flags.DEFINE_integer('Nmesh',256,'resolution of fastPM mesh')
flags.DEFINE_integer('Nmesh2D',2048, 'resolution of lensing map')
flags.DEFINE_float('BoxSize2D',6.37617,'field of view in degrees')
flags.DEFINE_integer('n_steps',40,'number of fastPM steps')
#bounds from KIDS contours, default values from Planck2015
flags.DEFINE_float('Omega_m',0.3089,'total matter density', lower_bound=0.1, upper_bound=0.5)
flags.DEFINE_float('sigma_8',0.8158,'amplitude of matter fluctuations', lower_bound=0.4, upper_bound=1.3)
flags.DEFINE_boolean('PGD',True,'whether to use PGD sharpening')
flags.DEFINE_integer('B',2,'force resolution factor')
flags.DEFINE_spaceseplist('zs_source',['1.'],'source redshifts')
flags.DEFINE_boolean('interp',True,'whether to interpolate between snapshots')
flags.DEFINE_boolean('save3D',False,'whether to dump the snapshots, requires interp to be set to False')
flags.DEFINE_string('snapshot_path',os.path.join(os.path.realpath(__file__),'snapshots/'),'path for storing snapshots')
flags.DEFINE_enum('mode', 'forward', ['forward','backprop'],'whether to run the forward model only or include backpropagation')
flags.DEFINE_boolean('analyze', True, 'whether to print out resource usage')

old_print = print
def print(*args):
    if rank==0:
        old_print(args)
    return True


def main(argv):
    del argv

    params              = FLAGS.flag_values_dict() 
    params['Nmesh']     = [FLAGS.Nmesh]*3
    params['BoxSize']   = [FLAGS.BoxSize]*3 
    params['Nmesh2D']   = [FLAGS.Nmesh2D]*2 
    params['BoxSize2D'] = [FLAGS.BoxSize2D]*2 
    params['zs_source'] = [float(zs) for zs in FLAGS.zs_source]

    cosmo = Planck15.match(Omega0_m=FLAGS.Omega_m)
    cosmo = cosmo.match(sigma8=FLAGS.sigma_8)

    label= '3Dmesh%d_2Dmesh%d_bs%d_fov%d_nsteps%d_B%d'%(FLAGS.Nmesh,FLAGS.Nmesh2D,FLAGS.BoxSize,FLAGS.BoxSize2D*100,FLAGS.n_steps,FLAGS.B)
    
    cl_path = os.path.join(FLAGS.cl_path,label)
    map_path= os.path.join(FLAGS.map_path,label)
        
    if not os.path.isdir(cl_path):
        os.makedirs(cl_path)
    if not os.path.isdir(map_path):
        os.makedirs(map_path)
    
    if FLAGS.save3D:
        snapshot_path = os.path.join(FLAGS.snapshot_path,label)
        if not os.path.isdir(snapshot_path):
            os.makedirs(snapshot_path)
        params['snapshot_path'] = snapshot_path

    sims_start = time.time()

    for ii in range(FLAGS.N_maps):
        print('progress in percent:', ii/params['N_maps']*100)
        kmaps, kmaps_deriv, pm = test_wl(params,cosmo=cosmo, num=ii)

        for jj,z_source in enumerate(params['zs_source']):
            kmap    = kmaps[jj]
            mapfile = os.path.join(map_path,'map_decon_zsource%d_map%d_of%d'%(z_source*10,ii,params['N_maps'])+'.npy')
            save_2Dmap(kmap,mapfile)
            print('2D map #%d at z_s=%.1f dumped to %s'%(ii,z_source,mapfile))
            
            bink,binpow,N = get_2Dpower(kmap)
            len_k = len(bink)
            if rank ==0:
                if ii==0 and jj==0:
                    binpows        = np.zeros((len(params['zs_source']),params['N_maps'],len_k))
                    binpows[jj,ii] = binpow

    end = time.time()

    print('time taken per sim in min %d'%((end-sims_start)/(params['N_maps']*len(params['zs_source']))))
    print('time takes before sims in min %d'%(sims_start-start))

    if rank==0:
        for jj,z_source in enumerate(params['zs_source']):
            binpow       = np.mean(binpows[jj], axis=0)
            binpow_std   = np.std(binpows[jj], axis=0)

            clfile       = os.path.join(cl_path,'mean_cl_zsource%d_averaged_over_%dmaps.npy'%(z_source*10,params['N_maps']))

            np.save(clfile,[bink,binpow,binpow_std,N])
            print('cls dumped to %s', clfile)

if __name__ == '__main__':
  app.run(main)

