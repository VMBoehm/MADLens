# MADLens
a differentiable lensing simulator ([preprint](https://arxiv.org/abs/2012.07266), [accepted publication](https://www.sciencedirect.com/science/article/abs/pii/S2213133721000445))

## Description
MADLens is a python package for producing non-Gaussian cosmic shear maps at arbitrary source redshifts. MADLens is designed to achieve high accuracy while keeping computational costs as low as possible. A MADLens simulation with only 256^3 particles produces convergence maps whose power agree with theoretical lensing power spectra up to scales of L=10000.
MADlens is based on a highly parallelizable particle-mesh algorithm and employs a sub-evolution scheme in the lensing projection and a machine-learning inspired sharpening step to achieve these high accuracies.
<p float="left">
<img src="/figures/redshift_comp.png" width="400"/> 
  
<img src="/figures/lensing_map.png" width="300"/> 
</p>
  
MADLens is fully differentiable with respect to the initial conditions of the underlying particle-mesh simulations and a number of cosmological parameters. These properties allow MADLens to be used as a forward model in Bayesian inference algorithms that require optimization or derivative-aided sampling. Another use case for MADLens is the production of large, high resolution simulation sets as they are required for training novel deep-learning-based lensing analysis tools.

## Installation and running the code

### On your personal computer

Download the code, then run
```
pip install -e .
```
which installs MADLens together with all its dependencies.

Alternatively, to create a conda environment with the required dependencies, run 
```
conda env create -f MADLens.yml
```
Run the code with

```
python run_lightcone.py
```

### On a cluster (NERSC)

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


## Settings

To display all parameters run

``` 
python run_lightcone.py --helpfull
```

to set a parameter, either change its default in [run_lightcone.py](https://github.com/VMBoehm/MADLens/blob/master/run_lightcone.py) or pass them as

```
python run_lightcone.py --parameter_name parameter_value
```
Here's a non-exhaustive list of parameters that can be set by the user:

Parameter |  Description | Typical Value(s) |
----------|--------------|------------------|
BoxSize | side length of the simulation box | 128-1024 Mpc/h |
Nmesh | resolution of the particle-mesh simulation | 64^3-512^3 |
B | force resolution factor | 2 |
Nsteps | number of steps in the FastPM simulation | 11-40 |
N_maps | number of output maps | >1 |
Nmesh2D | resolution of the convergence map | 256^2-2048^2 |
BoxSize2D | size of the convergence map in degrees | 2.5-22 degrees |
zs\_source | list of source redshifts | 0.3-2.0 |
Omega\_m | total matter density | 0.32 |
sigma\_8 | amplitude of matter fluctuations | 0.82 |
PGD | whether to use PGD enhancement or not | True/False |
interpolation | whether to use the sub-evolution scheme | True/False | 

### Particle Gradient Descent

We provide PGD parameters for a limited amount of configurations. Before using PGD, you have to dump those parameters into parameter files by running the notebook
[PGD_params.ipynb](https://github.com/VMBoehm/MADLens/blob/master/notebooks/PGD_params.ipynb)

## Forward Model and Gradient Computation

See [lightcone.py](https://github.com/VMBoehm/MADLens/blob/0a4d491a2c81f8a46eb350a23ab1456fe8654b86/MADLens/lightcone.py#L462) for examples of how to compute forward passes, vector-Jacobian and Jacobian-vector products.

For example,
```
kmaps, tape = model.compute(vout='kmaps', init=dict(rho=rho),return_tape=True)
```
evolves initial conditions *rho* into lensing maps *kmaps* (runs the forward model), while saving operations to the tape.

Once the tape has been created,
```
vjp         = tape.get_vjp()
kmap_vjp    = vjp.compute(init=dict(_kmaps=vector), vout='_rho')
```
computes the vector Jacobian product against a vector of the same shape as *kmaps*.

## Parameter Gradients

To use the MADLens version that support parameter derivatives, checkout the *param_derivs* branch.

## Contributors

Vanessa Böhm, Yu Feng, Max Lee, Biwei Dai 

## Packages

- Automatic Differentiation based on [VMAD](https://github.com/rainwoodman/vmad)
- Particle Mesh Simulation based on [FastPM](https://github.com/rainwoodman/fastpm-python)

other packages MADlens relies on:

- [Nbodykit](https://nbodykit.readthedocs.io/en/latest/index.html)
- [pmesh](https://github.com/rainwoodman/pmesh)



