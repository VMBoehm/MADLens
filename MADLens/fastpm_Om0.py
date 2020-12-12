from vmad import operator, autooperator
from vmad.core.stdlib import watchpoint, eval
from vmad.lib import linalg
from vmad.lib.fastpm import *

from fastpm.background import MatterDominated
from pmesh.pm import ParticleMesh
import numpy

class CosmologyFactory():

    from fastpm.background import MatterDominated

    def __init__(self):
        """
        Cosmology cacheing class to avoid recomputation of partile mesh
        for differing cosmology
        """
        self.cosmo_cache = dict()

    def get_cosmology(self, Om0, a):
        if numpy.array(Om0).ndim != 0:
            Om0 = Om0.flatten()[0]
        cosmo_id = hash((Om0))
        if cosmo_id in self.cosmo_cache:
            return self.cosmo_cache[cosmo_id]
        pt = CosmologyFactory.MatterDominated(Om0,a=a)
        self.cosmo_cache[cosmo_id] = pt
        return pt


@autooperator('dx->f,potk')
def gravity(dx, q, pm):
    x = q + dx
    layout = decompose(x, pm)

    xl = exchange(x, layout)
    rho = paint(xl, 1.0, None, pm)

    # convert to 1 + delta
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(q))

    rho = rho * fac
    rhok = r2c(rho)

    p = apply_transfer(rhok, fourier_space_laplace)

    r1 = []
    for d in range(pm.ndim):
        dx1_c = apply_transfer(p, fourier_space_neg_gradient(d, pm, order=1))
        dx1_r = c2r(dx1_c)
        dx1l = readout(dx1_r, xl, None)
        dx1 = gather(dx1l, layout)
        r1.append(dx1)

    f = linalg.stack(r1, axis=-1)
    return dict(f=f, potk=p)

def KickFactor(Om0, support, FactoryCache, ai, ac, af):
    pt = FactoryCache.get_cosmology(Om0, a=support)
    return 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)

def DriftFactor(Om0, support, FactoryCache, ai, ac, af):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    return 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)


def firststep_E(Om0, a, support, FactoryCache):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    E         = pt.E(a)
    return E
def firststep_D1(Om0, a, support, FactoryCache):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    D1        = pt.D1(a)
    return D1
def firststep_D2(Om0, a, support, FactoryCache):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    D2        = pt.D2(a)
    return D2
def firststep_f1(Om0, a, support, FactoryCache):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    f1        = pt.f1(a)
    return f1
def firststep_f2(Om0, a, support, FactoryCache):
    pt        = FactoryCache.get_cosmology(Om0, a=support)
    f2        = pt.f2(a)
    return f2
