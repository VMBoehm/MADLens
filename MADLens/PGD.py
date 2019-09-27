import numpy as np
from vmad.lib import fastpm, linalg
from vmad import autooperator
from vmad.core import stdlib

def PGDkernel(kl, ks):
    def kernel(k):
        k2 = k.normp(2)
        bad = k2 == 0
        k2[bad] = 1
        kl2 = kl**2
        ks4 = ks**4
        v = - np.exp(-kl2 / k2) * np.exp(-k2**2 / ks4) / k2
        v[bad] = 0
        return v
    return kernel

@autooperator('X->S')
def PGD_correction(X, alpha, kl, ks, pm, q):

    layout = fastpm.decompose(X, pm)
    xl     = fastpm.exchange(X, layout)

    rho    = fastpm.paint(xl, 1.0, None, pm)
    fac    = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(q)) 
    rho    = rho * fac

    rhok   = fastpm.r2c(rho)

    p      = fastpm.apply_transfer(rhok, PGDkernel(kl, ks))

    r1 = []
    for d in range(pm.ndim):
        dx1_c = fastpm.apply_transfer(p, fastpm.fourier_space_neg_gradient(d, pm, order=1))
        dx1_r = fastpm.c2r(dx1_c)
        dx1l  = fastpm.readout(dx1_r, xl, None)
        dx1   = fastpm.gather(dx1l, layout)
        r1.append(dx1)

    S = linalg.stack(r1, axis=-1)

    S = S * alpha

    return S 
