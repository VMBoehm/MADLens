#!/bin/bash
# source this file before running run_lightcone.py on cori interactive ndoes

source /global/common/software/m3035/conda-activate.sh 3.7
bcast-pip https://github.com/rainwoodman/vmad/archive/master.zip
bcast-pip https://github.com/bccp/abopt/archive/master.zip
bcast-pip https://github.com/rainwoodman/fastpm-python/archive/master.zip

export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export OMP_NUM_THREADS=4

#run the sims with:
#srun -n num_proc -c 2 python -u run_lightcone.py
