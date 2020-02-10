from vmad import Builder, autooperator, operator
import numpy as np
from nbodykit.cosmology import Planck15
from vmad.core.stdlib.operators import mul, div, add, sub
from vmad.core.stdlib.operators import pow as power
from vmad.lib.unary import log


@autooperator('omega0_m->delta_h')
def amplitude(omega0_m, n):
    factor_1 = mul(1.94e-5, power(omega0_m, sub(-.785, mul(.05, log(omega0_m)))))
    factor_2 = np.e**((n - 1) + 1.97 * (n - 1)**2)
    
    return dict(delta_h = mul(factor_1, factor_2))

@autooperator('omega_z, omega_lambda->growth')
def grow(omega_z, omega_lambda, z):
    pre_fac = mul((1+z)**-1, div(mul(5, omega_z), 2))
    inside = power(
        add(sub(power(omega_z, 4/7), omega_lambda), 
            mul(add(1, div(omega_z,2)), 
                add(1, div(omega_lambda,70)))), 
        -1)
    return dict(growth=mul(pre_fac, inside))

@autooperator('omega0_m->omega_z')
def get_omega_z(omega0_m, z):
    omega0_l = sub(1, omega0_m)
    num      = mul(omega0_m, (1+z)**3)
    denom    = add(omega0_l, mul(omega0_m, (1+z)**3))
    return dict(omega_z = div(num, denom))

@autooperator('omega0_m->omega_lambda')
def get_omega_lambda(omega0_m, z):
    omega0_l = sub(1, omega0_m)
    denom = add(omega0_l, mul(omega0_m, (1+z)**3))
    return dict(omega_lambda = div(omega0_l, denom))



@autooperator('Omega0_m->Pk')
def get_pklin(Omega0_m, Omega0_b, h, Tcmb0, C, H0, n, z, k):

    Obh2 = Omega0_b * h**2
    Omh2 = mul(Omega0_m, power(h, 2))
    f_baryon = div(Omega0_b, Omega0_m)
    
    theta_cmb = Tcmb0 / 2.7

    k_eq = mul(mul(0.0746, Omh2), power(theta_cmb, -2))  # units of 1/Mpc

    sound_horizon = div(mul(h * 44.5, log(div(9.83, Omh2))),
                        np.sqrt(1 + 10 * Obh2**0.75))  # in Mpc/h


    alpha_gamma = sub(1, add(mul(mul(0.328, log(431*Omh2)), f_baryon), \
                        mul(mul(0.38, log(22.3*Omh2)), power(f_baryon, 2))))
    #zeromode=1 conserves mean
    k = k.normp(p=2,zeromode=1)
    k = k * h  # in 1/Mpc now
    ks = mul(k, div(sound_horizon, h))
    q = div(k, mul(13.41, k_eq))

    gamma_eff = mul(
        Omh2, (alpha_gamma +
               div(sub(1, alpha_gamma), add(1, power(mul(0.43, ks), 4)))))
    
    q_eff = mul(q, div(Omh2, gamma_eff))

    L0 = log(add(mul(2, np.e), mul(1.8, q_eff)))
    C0 = add(14.2, div(731.0, add(1, mul(62.5, q_eff))))

    T = div(L0, add(L0, mul(C0, power(q_eff, 2))))

    ### ADD GROWTH###
    omega_zs = get_omega_z(Omega0_m, z)
    omega_lambdas = get_omega_lambda(Omega0_m, z)
    growth_z = grow(omega_zs, omega_lambdas, z)

    ###ADD AMPLITUDE###
    delta_h = amplitude(Omega0_m, n)

    ###ADD FACTOR###
    factor = 2 * np.pi**2 / (k)**3 * (C * k / H0)**(3 + n)

    Pk = div(
        mul(mul(mul(power(delta_h, 2), power(T, 2)), power(growth_z, 2)),
            factor), h**3)

    return dict(Pk=Pk)
