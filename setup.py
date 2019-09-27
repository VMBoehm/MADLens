from setuptools import setup

setup(name='MADLens',
      version='0.1',
      description='a differentiable lensing simulator',
      url='http://github.com/VMBoehm/MADLens',
      author='Vanessa Martina Böhm',
      author_email='vboehm@berkeley.edu',
      license='GNU GPLv3',
      packages=['MADLens'],
      dependency_links=['http://github.com/bccp/nbodykit', 'https://github.com/rainwoodman/vmad', 'https://github.com/bccp/abopt', 'https://github.com/rainwoodman/fastpm-python']
      )
