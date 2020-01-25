from vmad import Builder, autooperator, operator
import numpy as np
from nbodykit.cosmology import Planck15
from vmad.core.stdlib.operators import mul, div, add, sub
from vmad.core.stdlib.operators import pow as power




@operator
class log:
    ain = 'x'
    aout = 'y'
    
    def apl(node, x):
        return dict(y = np.log(x))
    def vjp(node, _y, x):
        return dict(_x = 1/x*_y)
    def jvp(node, x_, x):
        return dict(y_ = 1/x*x_)

@operator
class amplitude:
    ain = 'omega0_m'
    aout = 'delta_h'

    def apl(node, omega0_m, n):
        factor_1 = 1.94e-5 * omega0_m**(-.785 - .05 * np.log(omega0_m))
        factor_2 = np.e**((n-1) + 1.97 * (n-1)**2)

        return dict(delta_h=factor_1 * factor_2)

    def vjp(node, _delta_h, omega0_m, n):
        factor_1 = 1.94e-5*(-.785+.05*np.log(omega0_m))*.05/omega0_m\
                    * omega0_m**(-1.785-.05*np.log(omega0_m))

        factor_2 = np.e**((n - 1) + 1.97 * (n - 1)**2)
        return dict(_omega0_m=factor_1 * factor_2 * _delta_h)

    def jvp(node, omega0_m_, omega0_m, n):
        factor_1 = 1.94e-5*(-.785+.05*np.log(omege0_m))*.05/omega0_m \
                    * omega0_m**(-1.785-.05*np.log(omega0_m))
        factor_2 = np.e**((n - 1) + 1.97* (n - 1)**2)

        return dict(delta_h_=factor_1 * factor_2 * omega0_m_)

@operator
class growth:
    ain = 'omega_z', 'omega_lambda'
    aout = 'growth'
    
    def apl(node, omega_z, omega_lambda, z): 
        growth= (1+z)**(-1)*5*omega_z/2*(omega_z**(4/7)-omega_lambda+(1+omega_z/2)*(1+omega_lambda/70))**(-1)
        return dict(growth = growth)   
    
    def vjp(node, _growth, omega_z, omega_lambda, z):
        _omega_z = (1+z)**(-1)*5/2*(omega_z**(4/7)-omega_lambda+(1+omega_z/2)*(1+omega_lambda/70))**(-1)\
                    *(1- omega_z*(4/7*omega_z**(-3/7)+1/2+omega_lambda/140)\
                     *(omega_z**(4/7)-omega_lambda+(1+omega_z/2)*(1+omega_lambda/70))**(-1))
        
        _omega_lambda = 5*omega_z*(-1+1/70+omega_z/140)/(2*(1+z)*(omega_z**(4/7)-omega_lambda+(1+omega_z/2)\
                                                                  *(1+omega_lambda/70))**(-2))
        return dict(_omega_z = _omega_z*_growth, _omega_lambda =_omega_lambda*_growth)
    
    def jvp(node, omega_z_, omega_lambda_, omega_z, omega_lambda, z):
        omega_z_ *= (1+z)**(-1)*5/2*(omega_z**(4/7)-omega_lambda+(1+omega_z/2)*(1+omega_lambda/70))**(-1)\
                    *(1- omega_z*(4/7*omega_z**(-3/7)+1/2+omega_lambda/140)\
                     *(omega_z**(4/7)-omega_lambda+(1+omega_z/2)*(1+omega_lambda/70))**(-1))
        omega_lambda_ *= 5*omega_z*(-1+1/70+omega_z/140)/(2*(1+z)*(omega_z**(4/7)-omega_lambda+(1+omega_z/2)\
                                                                  *(1+omega_lambda/70))**(-2))
        return dict(growth_ = omega_z_+omega_lambda_)

@operator
class omega_z:
    ain = 'omega0_m'
    aout= 'omega_z'
    
    def apl(node, omega0_m, z):
        omega0_l = 1-omega0_m
        return dict(omega_z = omega0_m*(1+z)**3/(omega0_l +omega0_m*(1+z)**3))
    
    def vjp(node, _omega_z, omega0_m, z):
        _omega0_m = (1+z)**3/(1-omega0_m +omega0_m*(1+z)**3)\
                    *(1 - omega0_m*(1+z)**3/(1-omega0_m +omega0_m*(1+z)**3))
        return dict(_omega0_m = _omega0_m*_omega_z)
    def jvp(node, omega0_m_, omega0_m, z):
        omega0_m_ *= (1+z)**3/(omega0_l +omega0_m*(1+z)**3)\
                    *(1 - omega0_m*(1+z)**3/(omega0_l +omega0_m*(1+z)**3))
        return dict(omega_z_ = omega0_m_)

@operator
class omega_lambda:
    ain='omega0_m'
    aout = 'omega_lambda'
    
    def apl(node, omega0_m, z):
        omega0_l = 1-omega0_m
        return dict(omega_lambda = omega0_l/(omega0_l+omega0_m*(1+z)**3))
    
    def vjp(node, _omega_lambda, omega0_m, z):
        _omega0_m = -1/(1-omega0_m+omega0_m*(1+z)**3)*(1+(1-omega0_m*((1+z)**3-1))\
                                                       /(1-omega0_m +omega0_m*(1+z)**3))
        return dict(_omega0_m = _omega0_m *_omega_lambda)
    def jvp(node, omega0_m_, omega0_m, z):
        omega0_m_ *= -1/(1-omega0_m+omega0_m*(1+z)**3)*(1+(1-omega0_m*((1+z)**3-1))\
                                                       /(1-omega0_m +omega0_m*(1+z)**3))
        return dict(omega_lambda_ = omega0_m_)



@autooperator('Omega0_m->Pk')
def get_pklin(Omega0_m, Omega0_b, h, Tcmb0,C, H0, n, z, k):

    Obh2      = Omega0_b*h**2
    Omh2      = mul(Omega0_m, power(h, 2))
    f_baryon  = div(Omega0_b , Omega0_m)
    theta_cmb = Tcmb0/ 2.7
    
    
    # wavenumber of equality
    k_eq = mul(mul(0.0746, Omh2) , power(theta_cmb, -2)) # units of 1/Mpc


    sound_horizon = div(mul(h * 44.5, log(div(9.83,Omh2))), np.sqrt(1 + 10 * Obh2** 0.75)) # in Mpc/h


    alpha_gamma = sub(1, add(mul(mul(0.328, log(431*Omh2)), f_baryon), \
                        mul(mul(0.38, log(22.3*Omh2)), power(f_baryon, 2))))

    k = np.asarray(k) * h # in 1/Mpc now
    ks = mul(k, div(sound_horizon, h))
    q = div(k, mul(13.41,k_eq))

    gamma_eff = mul(Omh2, (alpha_gamma + div(sub(1, alpha_gamma), add(1, power(mul(0.43,ks),4)))))   
    q_eff = mul(q, div(Omh2, gamma_eff))
    
    L0 = log(add(mul(2,np.e) , mul(1.8, q_eff)))
    C0 = add(14.2, div(731.0, mul(add(1, 62.5),q_eff)))

    T = div(L0, add(L0, mul(C0, power(q_eff,2))))
    
    ### ADD GROWTH###
    omega_zs = omega_z(Omega0_m, z)
    omega_lambdas = omega_lambda(Omega0_m, z)
    growth_z = growth(omega_zs, omega_lambdas, z)
    
    ###ADD AMPLITUDE###
    delta_h =  amplitude(Omega0_m, n)
    ###ADD FACTOR###
    factor =2*np.pi**2/k**3 * (C*k/H0)**(3+n)

    
    Pk = div(mul(mul(mul(power(T,2),power(growth_z,2)),power(delta_h, 2)), factor),h**3)
#     Pk = T
    return dict(Pk=Pk)