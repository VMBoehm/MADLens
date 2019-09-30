from setuptools import setup

setup(name='MADLens',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='a differentiable lensing simulator',
      url='http://github.com/VMBoehm/MADLens',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='GNU GPLv3',
      packages=['MADLens'],
      install_requires=['numpy','cython','mpi4py','nbodykit',
                        'vmad',
                        'abopt','absl-py','fastpm'],
      #dependency_links=['https://github.com/rainwoodman/fastpm-python','http://github.com/bccp/nbodykit']
      )
