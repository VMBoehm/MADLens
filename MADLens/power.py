from vmad import Builder, autooperator, operator
import numpy as np
from nbodykit.cosmology import Planck15
from vmad.core.stdlib.operators import mul, div, pow




@operator
class amplitude:
	ain  = 'omega0_m'
	aout = 'delta_h' 
	
	def apl(node, omega0_m, n):
		factor_1 = 1.94e-5 * omega0_m**(-.785 -.05 * np.log(omega0_m))
		factor_2 = np.e**(-.95 * (n - 1) - .169 * (n - 1)**2)
		
		return dict(delta_h = factor_1*factor_2)

	def vjp(node, _delta_h, omega0_m, n):
		factor_1 = 1.94e-5*(-.785+.05*np.log(omega0_m))*.05/omega0_m\
					* omega0_m**(-1.785-.05*np.log(omega0_m))
		factor_2 = np.e**(-.95*(n-1)-.169*(n-1)**2)
		
		return dict(_omega0_m = factor_1*factor_2*_delta_h)
	
	def jvp(node, omega0_m_, omega0_m, n):
		factor_1 = 1.94e-5*(-.785+.05*np.log(omege0_m))*.05/omega0_m \
					* omega0_m**(-1.785-.05*np.log(omega0_m))
		factor_2 = np.e**(-.95*(n-1)-.169*(n-1)**2)
		
		return dict(delta_h_ = factor_1*factor_2*omega0_m_)

@operator
class growth:
	ain = 'omega_z', 'omega_lambda'
	aout = 'growth'
	
	def apl(node, omega_z, omega_lambda, z): 
		growth= (1+z)**(-1)*5*omega_z/2*(omega_z**(4/7)-omega_lambda+\
			    (1+omega_z/2)*(1+omega_lambda/70))**(-1)
		
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

@operator
class T0:
	ain = 'q_eff', 'c'
	aout = 'T_0'
	
	def apl(node, q_eff, c, theta_cmb):
		return dict(T_0 = np.log(2*np.e + 1.8*q_eff)/(np.log(2*np.e+1.8*q_eff)+c*q_eff**2))
	
	def vjp(node, _T_0, q_eff, c, theta_cmb):
		_q_eff = 1/((np.log(2*np.e+1.8*q_eff)+c*q_eff**2)*(2*np.e+1.8*q_eff)) \
			*(1.8-np.log(2*np.e+1.8*q_eff)/(np.log(2*np.e+1.8*q_eff)+c*q_eff**2) \
			  *(1.8+2*c*q_eff*(2*np.e+1.8*q_eff)))
		
		_c = -np.log(2*np.e + 1.8*q_eff)*(q_eff**2)/(np.log(2*np.e+1.8*q_eff)+c*q_eff**2)**2
		print('T0')
		return dict(_q_eff=_q_eff*_T_0, _c = _c*_T_0)
	
	def jvp(node, q_eff_, c_, q_eff, c, theta_cmb):
		q_eff_ *= 1/((np.log(2*np.e+1.8*q_eff)+c*q_eff**2)*(2*np.e+1.8*q_eff)) \
			*(1.8-np.log(2*np.e+1.8*q_eff)/(np.log(2*np.e+1.8*q_eff)+c*q_eff**2) \
			  *(1.8+2*c*q_eff*(2*np.e+1.8*q_eff)))
		
		c_ *= -np.log(2*np.e + 1.8*q_eff)*(q_eff**2)/(np.log(2*np.e+1.8*q_eff)+c*q_eff**2)**2
		
		return dict(T_0_ = q_eff_ + c_)

@operator
class q:
	ain = 'k_eq'
	aout= 'q'
	
	def apl(node, k_eq, k):
		return dict(q = k/(13.41*k_eq))
		#return dict(q = k*theta_cmb**2/(13.81*7.46e-2*(omega0_m*h**2))**(-2))
	
	def vjp(node, _q, k_eq, k):
		_k_eq = -13.41*k/(13.41*k_eq)**2
		print('k_eq')
		return dict(_k_eq = _k_eq*_q)
	
	def jvp(node, k_eq_, k):
		k_eq_ *= -13.41*k/(13.41*k_eq)**2
		
		return dict(q_ = k_eq_)

@operator
class c:
	ain = 'q_eff'
	aout = 'c'
	
	def apl(node, q_eff):
		return dict(c = 14.2 + 731/(1+62.5*q_eff))
	
	def vjp(node, _c, q_eff):
		_q_eff = -731*62.5/(1+62.5*q_eff)**2
		print('c')
		return dict(_q_eff =_q_eff*_c )
	
	def jvp(node, q_eff_, q):
		pass
		q_eff_ *= -731*62.5/(1+62.5*q_eff)**2
		return dict(c_=q_eff_)

@operator
class s:
	ain = 'omega0_m'
	aout = 's'
	
	def apl(node, omega0_m, omega0_b, h):
		s = 44.5 *h* np.log(9.83/(omega0_m*h**2)) / np.sqrt(1+10*(omega0_b*h**2)**(.75))
		print(s)
		return dict(s = s)
	def vjp(node, _s, omega0_m, omega0_b, h):
		_omega0_m = -44.5 *h/ (omega0_m*np.sqrt(1+10*(omega0_b*h**2)**(.75)))
		print('s')
		return dict(_omega0_m = omega0_m * _s)
		
	def jvp(node, omega0_m_, omega0_m, omega0_b, h):
		omega0_m_ *= -44.5 *h/ (omega0_m*np.sqrt(1+10*(omega0_b*h**2)**(.75)))
		
		return dict(s_ = omega0_m_)

@operator
class k_eq:
	ain = 'omega0_m'
	aout = 'k_eq'
	
	def apl(node, omega0_m, theta_cmb, h):
		k_eq = .0746*omega0_m*h**2*theta_cmb**(-2)
		return dict(k_eq = k_eq)
	
	def vjp(node, _k_eq, omega0_m, theta_cmb, h):
		_omega0_m = .0746*h**2*theta_cmb**(-2)
		print('k_eq')
		return dict(_omega0_m = _omega0_m*_k_eq)
		
	def jvp(node, omega0_m_, omega0_m, theta_cmb, h):
		omega0_m_ *= .0746*h**2*theta_cmb**(-2)
		return dict(k_eq_ = omega0_m)

@operator
class gamma:
	ain = 'omega0_m'
	aout = 'gamma'
	
	def apl(node, omega0_m, omega0_b, h):
		f = omega0_b/omega0_m
		gamma = 1-.328*np.log(431*omega0_m*h**2)*f+.38*np.log(22.3*omega0_m*h**2)*f**2
		return dict(gamma= gamma)
	def vjp(node, _gamma, omega0_m, omega0_b, h):
		_omega0_m = -.328*omega0_b/omega0_m**2*(1-np.log(431*omega0_m*h**2)) \
					+.38*omega0_b**2/omega0_m**3*(1-2*np.log(22.3*omega0_m*h**2))
		print('gamma')
		return dict(_omega0_m = _omega0_m*_gamma)
	
	def jvp(node, omega0_m_, omega0_m, omega0_b, h):
		omega0_m_ *= -.328*omega0_b/omega0_m**2*(1-np.log(431*omega0_m*h**2)) \
					+.38*omega0_b**2/omega0_m**3*(1-2*np.log(22.3*omega0_m*h**2))
		return dict(gamma_ = omega0_m_)

@operator
class gamma_eff:
	ain = 'omega0_m', 'gamma', 's'
	aout = 'gamma_eff'
	
	def apl(node, omega0_m, gamma, s, k, h):
		gamma_eff = omega0_m*h**2*(gamma+(1-gamma)/(1+(.43*k*s/h)**4))
		return dict(gamma_eff = gamma_eff)
	
	def vjp(node, _gamma_eff, omega0_m, gamma, s, k, h):
		_omega0_m = h**2*(gamma+(1-gamma)/(1+(.43*k*s/h)**4))
		_gamma = omega0_m*h**2
		_s = -4*omega0_m*h**2*(1-gamma)*(.43*k*s/h)**4/(s*(1+(.43*k*s/h)**4)**2)
		print('gamma_eff')
		return dict(_omega0_m = _omega0_m * _gamma_eff, _gamma = _gamma * _gamma_eff, _s = _s *_gamma_eff)
	
	def jvp(node, omega0_m_, gamma_, s_, omega0_m, gamma, s, k, h):
		omega0_m_ *= h**2*(gamma+(1-gamma)/(1+(.43*k*s/h)**4))
		gamma_ *= omega0_m*h**2
		s_ *= -4*omega0_m*h**2*(1-gamma)*(.43*k*s/h)**4/(s*(1+(.43*k*s/h)**4)**2)
		return dict(gamma_eff_ = omega0_m_+gamma_+s_)
		
@operator
class q_eff:
	ain = 'omega0_m', 'q', 'gamma_eff'
	aout = 'q_eff'
	
	def apl(node, omega0_m, q, gamma_eff, h):
		return dict(q_eff = q*omega0_m*h**2/gamma_eff)
	
	def vjp(node, _q_eff, omega0_m, q, gamma_eff, h):
		_omega0_m = q*h**2/gamma_eff
		_q = omega0_m*h**2/gamma_eff
		_gamma_eff = -q*omega0_m*h**2/gamma_eff**2
		return dict(_omega0_m = _omega0_m *_q_eff, _q =_q*_q_eff, _gamma_eff = _gamma_eff*_q_eff)
	
	def jvp(node, omega0_m_, q_, gamma_eff_, omega0_m, q, gamma_eff, h):
		omega0_m_ *= q*h**2/gamma_eff
		q_ *= omega0_m*h**2/gamma_eff
		gamma_eff_ *= -q*omega0_m*h**2/gamma_eff**2
		return dict(q_eff_ = omega0_m_ +q_ + gamma_eff_)

@autooperator('omega0_m->P')
def power(omega0_m, k,z, n, theta_cmb, cosmo):
	h = cosmo.h
	k = k*h 
	Omega0_b = cosmo.Omega0_b

	factor =2*np.pi**2/k**3 * (cosmo.C*k/cosmo.H0)**(3+n)


	delta_h =  pow(amplitude(omega0_m, n), 2)

	sound_horizon = s(omega0_m, Omega0_b, h)

	gamma_alpha = gamma(omega0_m, Omega0_b, h)

	k_eqs = k_eq(omega0_m, theta_cmb, h)

	gamma_effs = gamma_eff(omega0_m, gamma_alpha, sound_horizon, k, h)

	qs = q(k_eqs, k)

	q_effs = q_eff(omega0_m, qs, gamma_effs, h)

	cs = c(q_effs)

	omega_zs = omega_z(omega0_m, z)
	omega_lambdas = omega_lambda(omega0_m, z)

	transfer = pow(T0(q_effs,cs, theta_cmb),2)

	growth_z = pow(growth(omega_zs, omega_lambdas, z), 2)
	growth_0 = pow(growth(omega_zs, omega_lambdas, 0), 2)

	g_tot = growth_z
	amp = mul(factor, delta_h)
	amp = mul(amp,g_tot)

	P = mul(amp, transfer)
	P = div(mul(P, div(cosmo.Omega0_cdm, omega0_m)),h**3)

	return dict(P=P)

