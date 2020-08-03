from setuptools import setup

setup(name='MADLens',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='a differentiable lensing simulator',
      url='http://github.com/VMBoehm/MADLens',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='GNU GPLv3',
      packages=['MADLens', 'MADLens.tests'],
      install_requires=['numpy', 'nbodykit', 'dask[array]', 'vmad', 'abopt', 'absl-py','fastpm','mpi4py','cython'],
      )
