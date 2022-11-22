import numpy as np

from scipy.special import polygamma

C_0 = (2/3)*polygamma(1,1/3) - (2*np.pi/3)**2
N = 3
z = 0

#====11=================================================
term_1 = 8 - 12*np.log(2)
term_2 = C_0 - 8*np.log(2) + 1 
r_11 = (term_1 +  (term_2*z/2))*(1 - (1/N))

#====2x3================================================
r_22 = -3*C_0 + 4 + 4*np.log(2) + z*(-C_0 + 1 + 4*np.log(2))
r_22 = r_22/(2*N)

r_23 = -3*C_0 + 4*(1+np.log(2)) + z*(-C_0 + 1 + 4*np.log(2))

r_32 = np.log(2) - (3/2) + z*(np.log(2) - (C_0/4))

r_33 = (3/2)*C_0*(N-(1/N)) - 5*N + (2/N)*(1+np.log(2))
r_33 += z*(-C_0/(2*N) - (N/2) + 1/(2*N) + 2*np.log(2)/N)

#====4x5================================================
r_44 = (3/2)*C_0*(N-(1/N)-(1/2)) - 5*(N-(1/N)) + 2*np.log(2)/N + 7 - 4*np.log(2)
r_44 += z*(-C_0*(1/(2*N) + (1/4)) - (N/2) + 1/(2*N) + 2*np.log(2)/N + (1/2))

r_45 = 4*((C_0/8)*((1/N)-(1/2)) - 7/(6*N) + 5*np.log(2)/(6*N) + (7/12) - 2*np.log(2)/3)
r_45 += 4*z*((C_0/16) - 1/(12*N) + np.log(2)/(6*N) + (1/24) - (np.log(2)/3))

r_54 = 6*C_0/N + 9*C_0 - (16/N) + 40*np.log(2)/N + 4 - 32*np.log(2)
r_54 += z*(3*C_0 - (4/N)*(1-2*np.log(2)) - 2 - 16*np.log(2))
r_54 = r_54/4

r_55 = -C_0*(N/2 + 1/(2*N) + (1/4)) +(N/3) - 7/(3*N) + 26*np.log(2)/(3*N) + 3 - 28*np.log(2)/3 
r_55 += z*(-(C_0/2)*((1/N)-(1/2)) + (1/6)*(N-(1/N)) + 10*np.log(2)/(3*N) + (1/2) - 8*np.log(2)/3)

amz = 0.1185;
m_z = 91.1876;
m_c = 1.250;
m_b = 4.200;
m_t = 174.200;

def n_f(mu):
    if mu<m_c:
        return 3
    elif mu>=m_b:
        return 5
    else:
        return 4

from scipy.special import zeta
def Bcoeffs(f):
    b1 = 11 - (2/3)*f
    b2 = 102 - (38/3)*f
    b3 = (2857/2) - (5033/18)*f + (325/54)*(f**2)
    b4 = (149753/6) + 3564*zeta(3) - (1078361/162 + (6508/27)*zeta(3))*f
    b4 += ((50065/162) + (6472/81)*zeta(3))*(f**2) + (1093/729)*(f**3)
    b5 = (8157455 + 4975080*zeta(3) - 705672*zeta(4) - 4609440*zeta(5))/16
    b5 += (-336460813 - 115467936*zeta(3) + 10994940*zeta(4) + 97847640*zeta(5))*f/1944
    b5 += (25960913 + 16764744*zeta(3) - 2273616*zeta(4) - 9162240*zeta(5))*(f**2)/1944
    b5 += (-630559 - 1169328*zeta(3) + 349488*zeta(4) + 298080*zeta(5))*(f**3)/5832
    b5 += (1205 - 5472*zeta(3))*(f**4)/2916
    return np.array([b1, b2, b3, b4, b5])

def betafn(y, n_flav, n_loops=5):
    n_max = len(Bcoeffs(n_flav))
    if n_loops>n_max:
        print(f'Using {n_max} loops (max available)')
        n_loops = n_max

    bs = Bcoeffs(n_flav)
    poly = -2*sum(bs[n]*(y**(n+2)) for n in range(n_loops))
    return poly


def ODE(a,mu,**kwargs):
    y = a/(4*np.pi)
    dy_dmu = betafn(y,n_flav=n_f(mu))/mu 
    da_dmu = 4*np.pi*dy_dmu
    return da_dmu

from scipy.integrate import odeint
amc = odeint(ODE,amz,[m_z,m_c])[-1]

mus = np.linspace(m_c,2.1,20)
alphas = odeint(ODE, amc, mus)

alpha_3 = 4*np.pi*(1-1.00414)/r_11
def R_MSbar(mu, a=alpha_3, **kwargs):
    if a is None:
        alpha = odeint(ODE, amc, [m_c,mu])[-1]
    else:
        alpha = a

    g = alpha/(4*np.pi)
    R = np.zeros(shape=(5,5))
    R[0,0] = r_11
    R[1,1], R[1,2] = r_22, r_23
    R[2,2], R[2,1] = r_33, r_32
    R[3,3], R[3,4] = r_44, r_45
    R[4,4], R[4,3] = r_55, r_54

    R = np.identity(5)-g*R
    return R




























































