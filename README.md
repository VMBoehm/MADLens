# MADLens
a differentiable lensing simulator

## Installation and Usage

### On your personal computer

Download the code, then run
```
pip install -e .
```
which install MADLens together with all its dependencies.

Alternatively, to create a conda environment with the required dependencies, run 
```
conda env create -f MADLens.yml
```

### On a cluster

Download the code.

For an interactive session, allocate the desired number of nodes, then run 
```
source PrepareInteractiveJob.sh
```
then run 
```
srun -n <number of tasks> -u python run_lightcone.py <flags>
```

For job submission, prepare a job script following PrepareInteractiveJob.sh




