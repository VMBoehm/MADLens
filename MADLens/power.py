from vmad import Builder, autooperator, operator
import numpy as np
from nbodykit.cosmology import Planck15
from vmad.core.stdlib.operators import mul, div, add, sub
from vmad.core.stdlib.operators import pow as power
from vmad.lib.unary import log
#need sinc, exp, sqrt
from vmad.lib.unary import sinc, exp


def normalize(r,sigma_8,cosmo,transfer='EH', kmin=1e-5, kmax=1e1):
    r"""
    The mass fluctuation within a sphere of radius ``r``, in
    units of :math:`h^{-1} Mpc` at ``redshift``.

    This returns :math:`\sigma`, where

    .. math::

        \sigma^2 = \int_0^\infty \frac{k^3 P(k,z)}{2\pi^2} W^2_T(kr) \frac{dk}{k},

    where :math:`W_T(x) = 3/x^3 (\mathrm{sin}x - x\mathrm{cos}x)` is
    a top-hat filter in Fourier space.

    The value of this function with ``r=8`` returns
    :attr:`sigma8`, within numerical precision.

    Parameters
    ----------
    r : float, array_like
        the scale to compute the mass fluctation over, in units of
        :math:`h^{-1} Mpc`
    kmin : float, optional
        the lower bound for the integral, in units of :math:`\mathrm{Mpc/h}`
    kmax : float, optional
        the upper bound for the integral, in units of :math:`\mathrm{Mpc/h}`
    """
    import mcfit
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    if transfer != 'EH' and transfer !='NWEH':
        raise ValueError('transfer must be EH or NWEH')
    k = np.logspace(np.log10(kmin), np.log10(kmax), 1024)
    if transfer=='EH':
        p= get_Pk_EH.build(cosmo=cosmo, z=0, k=k)
    else:
        p= get_Pk_NWEH.build(cosmo=cosmo, z=0, k=k)

    Pk= p.compute(init = dict(Omega0_m=cosmo.Omega0_m),vout=['Pk'], return_tape=False)
    R, sigmasq = mcfit.TophatVar(k, lowring=True)(Pk, extrap=True)

    return (sigma_8/spline(R, sigmasq)(r)**.5)**2


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
def get_Pk_NWEH(Omega0_m, cosmo, z, k):

    Obh2 =cosmo.Omega0_b * cosmo.h**2
    Omh2 = mul(Omega0_m, power(cosmo.h, 2))
    f_baryon = div(cosmo.Omega0_b, Omega0_m)

    theta_cmb = cosmo.Tcmb0 / 2.7

    k_eq = mul(mul(0.0746, Omh2), power(theta_cmb, -2))  # units of 1/Mpc

    sound_horizon = div(mul(cosmo.h * 44.5, log(div(9.83, Omh2))),
                        np.sqrt(1 + 10 * Obh2**0.75))  # in Mpc/h


    alpha_gamma = sub(1, add(mul(mul(0.328, log(431*Omh2)), f_baryon), \
                        mul(mul(0.38, log(22.3*Omh2)), power(f_baryon, 2))))
    #zeromode=1 conserves mean
    #k = k.normp(p=2,zeromode=1)
    k = k * cosmo.h  # in 1/Mpc now
    ks = mul(k, div(sound_horizon, cosmo.h))
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

    omega_z0 = get_omega_z(Omega0_m, 0)
    omega_lambda0 = get_omega_lambda(Omega0_m, 0)
    growth_0 = grow(omega_z0, omega_lambda0, z)


    ###ADD AMPLITUDE###
    amplitude = (k/cosmo.h)**cosmo.n_s
    ###ADD FACTOR###
    Pk =  mul(mul( power(T, 2), div(power(growth_z, 2), power(growth_0,2))), amplitude)

    return dict(Pk=Pk)

@autooperator('a,b,q->c')
def f(a,b,q):
    return dict(c =div( a , add(a ,mul( b,power(q,2)))))

@autooperator('Omega0_m->Pk')
def get_Pk_EH(Omega0_m, cosmo, z, k):

    Obh2 = cosmo.Omega0_b * cosmo.h ** 2
    Omh2 = mul(Omega0_m, power(cosmo.h, 2))
    f_baryon = div(cosmo.Omega0_b, Omega0_m)
    theta_cmb = cosmo.Tcmb0 / 2.7

    # z and wavenumber of equality
    z_eq = mul(mul(2.5e4, Omh2),theta_cmb ** (-4)) # this is 1 + z
    k_eq = mul(mul(0.0746, Omh2),theta_cmb ** (-2)) # units of 1/Mpc

    # sound horizon and k_silk
    z_drag_b1 = mul(mul(0.313, power(Omh2, -0.419)), add(1, mul(0.607,power(Omh2, 0.674))))
    z_drag_b2 = mul(0.238, power(Omh2,0.223))
    z_drag    = mul(div(mul(1291, power(Omh2, 0.251)), add(1., mul(0.659, power(Omh2, 0.828)))), add(1., mul(z_drag_b1, power(Obh2, z_drag_b2))))



    r_drag = div(31.5 * Obh2 * theta_cmb ** -4 * 1000., add(1,z_drag))
    r_eq   = div(31.5 * Obh2 * theta_cmb ** -4 * 1000., z_eq)

    sound_horizon = mul(mul(div(2., mul(3.,k_eq)),power(div(6., r_eq), .5)), \
                    log(div(add(power(add(1,r_drag), .5), power(add(r_drag,r_eq), .5)), add(1, power(r_eq, .5)))))

    k_silk = mul(mul(1.6 * Obh2 ** 0.52, power(Omh2,0.73)),add(1, power(mul(10.4,Omh2), -0.95)))

    # alpha_c
    alpha_c_a1 = mul(power(mul(46.9,Omh2),0.670 ), add(1, power(mul(32.1,Omh2), -0.532)))
    alpha_c_a2 = mul(power(mul(12.0,Omh2), 0.424), add(1, power(mul(45.0,Omh2), -0.582)))
    alpha_c = mul(power(alpha_c_a1, -f_baryon), power(alpha_c_a2 , power(-f_baryon,3)))

    # beta_c
    beta_c_b1 = div(0.944, add(1, power(mul(458,Omh2), -0.708)))
    beta_c_b2 = mul(0.395, power(Omh2, -0.0266))
    beta_c = div(1., add(1 , sub(mul(beta_c_b1, power(sub(1,f_baryon), beta_c_b2)), 1)))

    y = div(z_eq, add(1, z_drag))
    alpha_b_G = mul(y, \
                    add(mul(-6.,power(add(1,y), .5)),
                        mul(add(2.,mul(3.,y)),
                            log(div(add(power(add(1,y), .5),1),sub(power(add(1,y), .5),1))))))
    alpha_b = mul(mul(mul(2.07,  k_eq), sound_horizon), mul(power(add(1,r_drag),-0.75), alpha_b_G))

    beta_node = mul(8.41, power(Omh2, 0.435))
    beta_b    = add(add(0.5, f_baryon), mul(sub(3., mul(2.,f_baryon)),power( add(power(mul(17.2,Omh2), 2), 1 ), .5)))

    k = k * cosmo.h # now in 1/Mpc

    q = div(k, mul(13.41,k_eq))
    ks = mul(k,sound_horizon)

    T_c_ln_beta   = log(add(np.e, mul(mul(1.8,beta_c),q)))
    T_c_ln_nobeta = log(add(np.e, mul(1.8,q)));

    T_c_C_alpha   = add(div(14.2, alpha_c), div(386., add(1, mul(69.9,power( q, 1.08)))))


    T_c_C_noalpha = add(14.2,  div(386., add(1, mul(69.9, power(q, 1.08)))))


    T_c_f = div(1., add(1.,power(div(ks,5.4), 4)))


    T_c = add(mul(T_c_f, f(T_c_ln_beta, T_c_C_noalpha, q)), mul(sub(1,T_c_f), f(T_c_ln_beta, T_c_C_alpha,q)))


    s_tilde = mul(sound_horizon, power(add(1, power(div(beta_node,ks),3)), (-1./3.)))


    ks_tilde = mul(k,s_tilde)


    T_b_T0 = f(T_c_ln_nobeta, T_c_C_noalpha, q)


    T_b_1 = div(T_b_T0, add(1, power(div(ks,5.2),2 )))


    T_b_2 = mul(div(alpha_b, add(1, power(div(beta_b,ks),3 ))), exp(-power(div(k,k_silk), 1.4)))


    T_b = mul(sinc(div(ks_tilde,np.pi)), add(T_b_1, T_b_2))

    T = add(mul(f_baryon,T_b), mul(sub(1,f_baryon),T_c));
    ### ADD GROWTH###
    omega_zs = get_omega_z(Omega0_m, z)
    omega_lambdas = get_omega_lambda(Omega0_m, z)
    growth_z = grow(omega_zs, omega_lambdas, z)
    omega_zs = get_omega_z(Omega0_m, 0)
    omega_lambdas = get_omega_lambda(Omega0_m, 0)
    growth_0 = grow(omega_zs, omega_lambdas, 0)

    factor = (k/cosmo.h)**cosmo.n_s

    Pk = mul(mul(power(T, 2), div(power(growth_z, 2),power(growth_0,2))),factor)


    return dict(Pk=Pk)
