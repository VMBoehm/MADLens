import numpy as np
from vmad.core.operator import operator

def _finite_diff(psi, func, epsilon, mode='forward'):
    """
    Find the finite differencing of a scalar function based off of an input psieter.
    Result is the same shape as psi.

    Parameters
    ----------
    psi:     parameter to difference with respect to
    func:    function for finite difference; must return a scalar.
    epsilon: amount of forward stepping in differencing, if a vector, must be the
             same shape as psi.

    Returns
    -------
    forward differencing solution to function with respect to input psi.

    """
    if mode=='forward':
        k1, k2, k3=1, 0, 1

    elif mode=='backward':
        k1, k2, k3=-1, 0, -1

    elif mode=='central':
        k1, k2, k3=1/2, -1/2, 1
    r = []
    it = np.nditer([ psi, epsilon], flags=['multi_index'])
    for  psi1, epsilon1 in it:
        basis = np.zeros_like(psi, dtype='f8')
        basis[it.multi_index] = 1
        r.append( k3 * (func(psi + k1 * epsilon1 * basis)  - func(psi + k2 * epsilon1 * basis)) / epsilon1)
    return np.array(r).reshape(list(it.operands[0].shape)+ list(np.array(r).shape[1:]))

@operator
class finite_operator:
    """
    Converts a function to a numerical differentiated operator.

    """
    ain = {'psi': '*'}
    aout = {'y':'*'}

    def apl(node, psi, func, epsilon, mode='central'):
        f = func(psi)
        return dict(y=f)

    def vjp(node, _y, psi, func, epsilon, mode='central'):
        delta = _finite_diff(psi, lambda x: func(x), epsilon, mode=mode)
        if delta.ndim ==0:
            delta = np.broadcast_to(delta, 1)
        _psi = np.einsum('i...,...->...', delta, _y)
        return dict(_psi=_psi)

    def jvp(node, psi_, psi, func, epsilon, mode='central'):
        delta = _finite_diff(psi, lambda x: func(x), epsilon, mode=mode)
        if delta.ndim==0:
            delta = np.broadcast_to(delta, 1)
        if psi_.ndim==0:
            psi_ = np.broadcast_to(psi_, 1)
        return np.einsum('i... ,i...->...', delta, psi_)
