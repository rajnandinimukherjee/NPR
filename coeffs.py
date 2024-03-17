from scipy.integrate import quad
from scipy.special import spence
from scipy.integrate import odeint
from scipy.special import polygamma
from scipy.linalg import expm
import numpy as np
import pdb

# ====beta-fn coeffs====================================
from scipy.special import zeta


def Bcoeffs(f, **kwargs):
    # num_colors=3, need to provide num_flavors=f
    b0 = 11 - (2/3)*f
    b1 = 102 - (38/3)*f
    b2 = (2857/2) - (5033/18)*f + (325/54)*(f**2)
    b3 = (149753/6) + 3564*zeta(3) - (1078361/162 + (6508/27)*zeta(3))*f
    b3 += ((50065/162) + (6472/81)*zeta(3))*(f**2) + (1093/729)*(f**3)
    b4 = (8157455 + 4975080*zeta(3) - 705672*zeta(4) - 4609440*zeta(5))/16
    b4 += (-336460813 - 115467936*zeta(3) +
           10994940*zeta(4) + 97847640*zeta(5))*f/1944
    b4 += (25960913 + 16764744*zeta(3) - 2273616 *
           zeta(4) - 9162240*zeta(5))*(f**2)/1944
    b4 += (-630559 - 1169328*zeta(3) + 349488 *
           zeta(4) + 298080*zeta(5))*(f**3)/5832
    b4 += (1205 - 5472*zeta(3))*(f**4)/2916
    return np.array([b0, b1, b2, b3, b4])


# ====computing alphas at mu values===================
amz = 0.1185
gmz = np.sqrt(4*np.pi*amz)
m_z = 91.1876
m_c = 1.250
m_b = 4.200
m_t = 174.200


def n_f(mu):
    if mu < m_c:
        return 3
    elif mu >= m_b:
        return 5
    else:
        return 4


def betafn(g, n_flav, n_loops=5):
    n_max = len(Bcoeffs(n_flav))
    if n_loops > n_max:
        print(f'Using {n_max} loops (max available)')
        n_loops = n_max

    bs = Bcoeffs(n_flav)
    mult = g**2/(16*np.pi**2)
    poly = -sum(g*bs[n]*mult**(n+1) for n in range(n_loops))
    return poly


def ODE(g, mu, f, **kwargs):
    if f == None:
        f = n_f(mu)
    dg_dmu = betafn(g, f)/mu
    return dg_dmu


gmc = odeint(ODE, gmz, [m_z, m_c], args=(None,))[-1]


def g(mu, **kwargs):
    return odeint(ODE, gmc, [m_c, mu], args=(3,))[-1]


# ====C0 function================================================


def C0(u):

    term1 = (-1j+np.sqrt(3))
    term1 = term1/(np.sqrt(3)-1j*np.sqrt(4*u+1))

    term2 = (1j+np.sqrt(3))
    term2 = term2/(np.sqrt(3)-1j*np.sqrt(4*u+1))

    term3 = (-1j+np.sqrt(3))
    term3 = term3/(1j*np.sqrt(4*u+1)+np.sqrt(3))

    term4 = (1j+np.sqrt(3))
    term4 = term4/(1j*np.sqrt(4*u+1)+np.sqrt(3))

    li = spence(term1)-spence(term2)+spence(term3)-spence(term4)
    return -(li*2*1j/np.sqrt(3)).real


def mylog(x):
    if x < 0:
        return np.log(-x)+1j*np.pi
    else:
        return np.log(x)


def C0_int(u):
    if u == 0:
        return 2.34239
    else:
        d1 = (-1+1j*np.sqrt(3))/2
        n1 = (-2-(1/u)-np.sqrt((u**-2)+(4/u)))/2

        def func(y):
            y_ = y/(1-y)
            num = mylog(y_-n1)
            num += mylog(n1*y_-1)
            num += -mylog(n1)
            num += -2*mylog(y_+1)
            num += mylog(u)
            num += -mylog(u+1)
            den = (y+(y-1)*d1)*(y+(y-1)/d1)
            return num/den

        def real_func(y):
            return func(y).real

        def imag_func(y):
            return func(y).imag

        return -quad(real_func, 0, 1)[0]


def alpha_s(mu):
    return (g(mu)**2)/(4*np.pi)


def R_mSMOM_to_MSbar(mu, mbar):
    CF = 4/3
    sq = (mbar/mu)**2 if mbar != 0.0 else 0.0
    mrat = (mbar**2)/(mbar**2 + mu**2)

    cons = -4-(C0(0)/2)+2*C0(sq)
    if mbar != 0.0:
        mass = 1+4*np.log(mrat)-sq*np.log(mrat)
        mass = -sq*mass-3*np.log(sq/mrat)
        cons += mass
    return 1 + (alpha_s(mu)*CF/(4*np.pi))*(cons)


# ====computing RISMOM(gamma-gamma)->MSbar matching factors======
C_0 = (2/3)*polygamma(1, 1/3) - (2*np.pi/3)**2
N = 3
z = 0

term_1 = 8 - 12*np.log(2)
term_2 = C_0 - 8*np.log(2) + 1
r_gg_11 = (term_1 + (term_2*z/2))*(1 - (1/N))
r_gg_22 = -3*C_0 + 4 + 4*np.log(2) + z*(-C_0 + 1 + 4*np.log(2))
r_gg_22 = r_gg_22/(2*N)
r_gg_23 = -3*C_0 + 4*(1+np.log(2)) + z*(-C_0 + 1 + 4*np.log(2))
r_gg_32 = np.log(2) - (3/2) + z*(np.log(2) - (C_0/4))
r_gg_33 = (3/2)*C_0*(N-(1/N)) - 5*N + (2/N)*(1+np.log(2))
r_gg_33 += z*(-C_0/(2*N) - (N/2) + 1/(2*N) + 2*np.log(2)/N)
r_gg_44 = (3/2)*C_0*(N-(1/N)-(1/2)) - 5*(N-(1/N)) + \
    2*np.log(2)/N + 7 - 4*np.log(2)
r_gg_44 += z*(-C_0*(1/(2*N) + (1/4)) - (N/2) + 1/(2*N) + 2*np.log(2)/N + (1/2))
r_gg_45 = 4*((C_0/8)*((1/N)-(1/2)) - 7/(6*N) + 5 *
          np.log(2)/(6*N) + (7/12) - 2*np.log(2)/3)
r_gg_45 += 4*z*((C_0/16) - 1/(12*N) + np.log(2)/(6*N) + (1/24) - (np.log(2)/3))
r_gg_54 = 6*C_0/N + 9*C_0 - (16/N) + 40*np.log(2)/N + 4 - 32*np.log(2)
r_gg_54 += z*(3*C_0 - (4/N)*(1-2*np.log(2)) - 2 - 16*np.log(2))
r_gg_54 = r_gg_54/4
r_gg_55 = -C_0*((N/2) + 1/(2*N) + (1/4)) + (N/3) - 7/(3*N) + \
    26*np.log(2)/(3*N) + 3 - 28*np.log(2)/3
r_gg_55 += z*(-(C_0/2)*((1/N)-(1/2)) + (1/6)*(N-(1/N)) +
           10*np.log(2)/(3*N) + (1/2) - 8*np.log(2)/3)

r_gg_mtx = np.zeros(shape=(5, 5))
r_gg_mtx[0, 0] = r_gg_11
r_gg_mtx[1, 1], r_gg_mtx[1, 2] = r_gg_22, r_gg_23
r_gg_mtx[2, 2], r_gg_mtx[2, 1] = r_gg_33, r_gg_32
r_gg_mtx[3, 3], r_gg_mtx[3, 4] = r_gg_44, r_gg_45
r_gg_mtx[4, 4], r_gg_mtx[4, 3] = r_gg_55, r_gg_54


# ====computing RISMOM(qslash-qslash)->MSbar matching factors======

r_qq_11 = -(9/N) + (12*np.log(2)/N) + 9 - 12*np.log(2) +\
        z*(-C_0/N + C_0 + (4*np.log(2)/N) - 4*np.log(2))
r_qq_22 = -3*C_0/(2*N) + (2/N) + 2*np.log(2)/N +\
        z*(-C_0/(2*N) + 1/(2*N) + 2*np.log(2)/N)
r_qq_23 = -3*C_0 + 4*(1 + np.log(2)) + z*(-C_0 + 1 + 4*np.log(2))
r_qq_32 = np.log(2) - 1 - z*(np.log(2)-1/4)
r_qq_33 = 3*C_0*(N/2 - 1/(2*N)) - 4*N + 2*(1 + np.log(2))/N +\
        z*(C_0*(N - 1/N)/2 - N + 1/(2*N) + 2*np.log(2)/N)
r_qq_44 = C_0*(3*N/2 - 1/N - 1/4) - 4*N + 3/N + 2*np.log(2)/N + 6 - 4*np.log(2) +\
        z*(C_0*(N - 3/N - 3/2)/2 - N + 2*(1 + np.log(2))/N + 3/2)
r_qq_45 = C_0*(1/N - 5/4)/12 + (-13/2 + 5*np.log(2))/(6*N) + 2*(1 - np.log(2))/3 +\
        z*(C_0*(1/N + 5/2)/24 + (-1 + np.log(2))/(6*N) - 1/24 - np.log(2)/3)
r_qq_45 = 4*r_qq_45
r_qq_54 = C_0*(4*N + 4/N + 11) - 8*N - 12/N + 40*np.log(2)/N - 32*np.log(2) +\
        z*(C_0*(-4*N + 2/N + 1) + 8*N + 8*(-1 + np.log(2))/N + 2 - 16*np.log(2))
r_qq_54 = r_qq_54/4
r_qq_55 = -C_0*(5*N/2 + 1/N + 5/4)/3 + 2*N + (-11 + 26*np.log(2))/(3*N) + 10/3 - 28*np.log(2)/3 +\
        z*(C_0*(5*N - 7/N + 5/2)/6 - N + (2 + 10*np.log(2))/(3*N) + 1/6 - 8*np.log(2)/3)


r_qq_mtx = np.zeros(shape=(5, 5))
r_qq_mtx[0, 0] = r_qq_11
r_qq_mtx[1, 1], r_qq_mtx[1, 2] = r_qq_22, r_qq_23
r_qq_mtx[2, 2], r_qq_mtx[2, 1] = r_qq_33, r_qq_32
r_qq_mtx[3, 3], r_qq_mtx[3, 4] = r_qq_44, r_qq_45
r_qq_mtx[4, 4], r_qq_mtx[4, 3] = r_qq_55, r_qq_54

def R_RISMOM_MSbar(mu, scheme='gamma', obj='bag', **kwargs):
    
    r_mtx = r_qq_mtx if scheme=='qslash' else r_gg_mtx

    R_ij = np.identity(5) - (g(mu)**2)*r_mtx/(16*np.pi**2)

    if obj=='bag':
        if mu==3.0:
            R_S = 1.05259 if scheme=='gamma' else 1.01664
        else:
            CF = 4/3
            if scheme=='qslash':
                R_S = 1 + alpha_s(mu)*CF*(4 - 3*C0(0)/2)/(4*np.pi)
            elif scheme=='gamma':
                R_S = 1 + alpha_s(mu)*CF*4/(4*np.pi)
        R_ij[1:, 1:] = R_ij[1:, 1:]/(R_S**2)
    elif obj=='ratio':
        R_ij = R_ij/R_ij[0,0]

    if 'rotate' in kwargs:
        rot_mtx = kwargs['rotate']
        R_ij = rot_mtx@R_ij@np.linalg.inv(rot_mtx)

    return R_ij


# ====gamma_0 matrix elements============================
N_c = 3
g_11 = 6*(1-(1/N_c))
g_22 = 6/N_c
g_23 = 12
g_32 = 0
g_33 = 6*(-N_c+(1/N_c))
g_44 = 6*(1-N_c+(1/N_c))
g_45 = (1/2)-(1/N_c)
g_54 = -24-(48/N_c)
g_55 = 6+(2*N_c)-(2/N_c)
gamma_0 = np.zeros(shape=(5, 5))
gamma_0[0, 0] = g_11
gamma_0[1, 1], gamma_0[1, 2] = g_22, g_23
gamma_0[2, 1], gamma_0[2, 2] = g_32, g_33
gamma_0[3, 3], gamma_0[3, 4] = g_44, 4*g_45
gamma_0[4, 3], gamma_0[4, 4] = g_54/4, g_55
# gamma_0 = gamma_0*((4*np.pi)**(-2))

# ====gamma_1 matrix elements==========================


def gamma_1_MS(f):
    g_11 = -(22/3)-(57/(2*N_c**2))+(39/N_c)-(19*N_c/6)+(f*(2/3)*(1-(1/N_c)))
    g_22 = 15/(2*N_c**2)+(137/6)-f*22/(3*N_c)
    g_23 = (200*N_c/3)-(6/N_c)-(f*44/3)
    g_32 = (71*N_c/4)+(9/N_c)-f*2
    g_33 = -(203*N_c**2)/6+(479/6)+15/(2*N_c**2)+(f/3)*(10*N_c-(22/N_c))
    g_44 = -(203*N_c**2/6)+(107*N_c/3)+(136/3)-(12/N_c) - \
        107/(2*N_c**2)+(f/3)*(10*N_c-2-(10/N_c))
    g_45 = -(N_c/36)-(31/9)+(9/N_c)-4/(N_c**2)+(f/18)*((2/N_c)-1)
    g_54 = -(364*N_c/3)-(704/3)-(208/N_c)-320/(N_c**2)+(f/3)*(136+(176/N_c))
    g_55 = (343*N_c**2/18)+21*N_c-(188/9)+(44/N_c) + \
        21/(2*N_c**2)+(f/9)*(-26*N_c-54+(2/N_c))
    gamma_1 = np.zeros(shape=(5, 5))
    gamma_1[0, 0] = g_11
    gamma_1[1, 1], gamma_1[1, 2] = g_22, g_23
    gamma_1[2, 1], gamma_1[2, 2] = g_32, g_33
    gamma_1[3, 3], gamma_1[3, 4] = g_44, 4*g_45
    gamma_1[4, 3], gamma_1[4, 4] = g_54/4, g_55
    # gamma_1 = gamma_1*((4*np.pi)**(-4))
    return gamma_1


# ====calculating gamma_1_RISMOM=========
beta_0 = Bcoeffs(3)[0]
gamma_1_RISMOM = r_gg_mtx@gamma_0 - gamma_0@r_gg_mtx + \
    gamma_1_MS(f=3) - 2*beta_0*r_gg_mtx

J = np.loadtxt('J.txt', delimiter=',')
L = np.loadtxt('L.txt', delimiter=',')
def K(mu, **kwargs):
    g_term = g(mu)/(4*np.pi)
    return np.eye(5) + J*g_term**2 + L*(g_term**2)*np.log(g(mu))

def K_inv(mu, **kwargs):
    g_term = g(mu)/(4*np.pi)
    return np.eye(5) - J*g_term**2 - L*(g_term**2)*np.log(g(mu))
