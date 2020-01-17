import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from MADLens.util import *

run_dir = '../runs'

githashes = os.listdir(run_dir)
for githash in githashes:
    print(githash)
    path = os.path.join(run_dir, githash)
    print(os.listdir(path))

my_run0 = Run(githash='176524b', label='myrun', rnum=3, local_path='../', cell_type='van')
# this function loads all available clkk (clkk for all source redhifts) and computes their theory counterparts
# individual clkk can be loaded with get_measured_cls/get_theory_cls
my_run0.fill_cl_dicts()

# load a single map kappa map at a specific source redhsift into memory
mymap = my_run0.get_map(z_source=my_run0.params['zs_source'][0],num=0)

ll = len(my_run0.params['zs_source'])
plt.figure(figsize=(ll * 5, 4))
for ii, zs in enumerate(my_run0.params['zs_source']):
    plt.subplot(1, ll, ii + 1)
    plt.title('$z_s$=%.1f' % zs, fontsize=12)
    plt.loglog(my_run0.measured_cls[str(zs)]['L'],
               my_run0.measured_cls[str(zs)]['L']**2 *
               my_run0.measured_cls[str(zs)]['clkk'],
               label='simulation')
    plt.semilogx(my_run0.theory_cls[str(zs)]['L'],
                 my_run0.theory_cls[str(zs)]['L']**2 *
                 my_run0.theory_cls[str(zs)]['clkk'],
                 label='theory')
    plt.legend(fontsize=12)
    # if ii == 0:
    #     # plt.ylabel('$C_L^{\kappa \kappa}$', fontsize=14)
    # plt.xlabel('L', fontsize=12)
# plt.xlim(200,10000)
# plt.ylim(1e-4,5e-3)
plt.show()