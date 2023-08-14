from basics import *
#from eta_c import *

from scipy.special import spence
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
    return -real(li*2*1j/np.sqrt(3))

from scipy.integrate import quad
from scipy import real, imag
def mylog(x):
    if x<0:
        return np.log(-x)+1j*np.pi
    else:
        return np.log(x)

def C0_int(u):
    if u==0:
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
            return real(func(y))
        def imag_func(y):
            return imag(func(y))

        return -quad(real_func,0,1)[0]
    


from coeffs import *
def alpha_s(mu):
    return (g(mu)**2)/(4*np.pi)

def R_mSMOM_to_MSbar(mu, mbar):
    CF = 4/3
    sq = (mbar/mu)**2
    mrat = (mbar**2)/(mbar**2 + mu**2)

    cons = -4-(C0(0)/2)+2*C0(sq)
    mass = 1+4*np.log(mrat)-sq*np.log(mrat)
    mass = -sq*mass-3*np.log(sq/mrat)
    return 1 + (alpha_s(mu)*CF/(4*np.pi))*(cons+mass)


