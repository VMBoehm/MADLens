#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular 
#SBATCH -J small_run0
#SBATCH --mail-user=vboehm@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 17:00:00
#SBATCH --account m3058 

source /global/common/software/m3035/conda-activate.sh 3.7
bcast-pip https://github.com/rainwoodman/vmad/archive/master.zip
bcast-pip https://github.com/bccp/abopt/archive/master.zip
bcast-pip https://github.com/rainwoodman/fastpm-python/archive/master.zip
bcast-pip https://github.com/abseil/abseil-py/archive/master.zip

export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the sims with:
srun -n 32 -c 2 --cpu_bind=cores python -u run_lightcone.py 0


