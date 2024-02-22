import glob
import itertools
import os
import pdb
import pickle
import sys

import h5py
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.special import gammaincc
from tqdm import tqdm

from ensemble_parameters import *
from plot_settings import plotparams
import warnings
warnings.filterwarnings('ignore')

# ====misc functions ==================================================


def err_disp(num, err, n=2, sys_err=None, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    if err==0.0:
        return str(np.around(num,2))
    else:
        if sys_err != None:
            err_size = max(int(np.floor(np.log10(np.abs(err)))),
                           int(np.floor(np.log10(np.abs(sys_err)))))
        else:
            err_size = int(np.floor(np.log10(np.abs(err))))

        num_size = int(np.floor(np.log10(np.abs(num))))
        min_size = min(err_size, num_size+(n-1))
        err_n_digits = int(err*10**(-(min_size-(n-1))))

        if min_size > (n-1):
            disp_str = f'{num}({err})'
        else:
            disp_str = "{:.{m}f}".format(num, m=-(min_size-(n-1)))
            disp_str += f'({err_n_digits})'

        if sys_err != None:
            sys_err_n_digits = int(sys_err*10**(-(min_size-(n-1))))
            disp_str += f'({sys_err_n_digits})'

        return disp_str


def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)


def call_PDF(filename, open=True, **kwargs):
    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format='pdf')
    pdf.close()
    plt.close('all')
    if open:
        os.system('open '+filename)


def fit_func(x, y, ansatz, guess,
             start=0, end=None,
             verbose=False,
             correlated=False,
             pause=False, **kwargs):

    if end == None:
        end = len(x.val)

    if correlated:
        cov = COV(y.btsp[:, start:end], center=y.val[start:end])
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)
    else:
        L = np.diag(1/y.err[start:end])

    def diff(inp, out, param, fit='central', k=0):
        return out - ansatz(inp, param, fit=fit, k=0, **kwargs)

    def LD(param):
        return L.dot(diff(x.val[start:end],
                          y.val[start:end],
                          param, fit='central'))

    res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
    if verbose:
        print(res)

    chi_sq = LD(res.x).dot(LD(res.x))
    DOF = len(x.val[start:end])-np.count_nonzero(guess)

    res_btsp = np.zeros(shape=(N_boot, len(guess)))
    for k in range(N_boot):
        def LD_k(param):
            return L.dot(diff(x.btsp[k, start:end],
                              y.btsp[k, start:end],
                              param, fit='btsp', k=k))
        res_k = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
        res_btsp[k,] = res_k.x

    res = stat(val=res.x, err='fill', btsp=res_btsp)

    def mapping(am):
        if not isinstance(am, stat):
            am = stat(
                val=np.array(am),
                btsp='fill')

        return stat(
            val=ansatz(am.val, res.val),
            err='fill',
            btsp=np.array([ansatz(am.btsp[k,], res.btsp[k])
                           for k in range(N_boot)])
        )
    res.mapping = mapping
    res.chi_sq = chi_sq
    res.DOF = DOF
    res.pvalue = gammaincc(DOF/2, chi_sq/2)
    res.range = (start, end)
    if pause:
        pdb.set_trace()

    return res
# =====statistical obj class======================================


class stat:
    N_boot = 1000

    def __init__(self, val, err=None, btsp=None,
                 dtype=None, **kwargs):
        self.val = np.array(val)
        self.shape = self.val.shape
        self.dtype = self.val.dtype if dtype is None else dtype

        accept_types = [np.ndarray, list, int, float]
        self.err = np.array(err) if type(err) in accept_types else err
        self.btsp = np.array(btsp) if type(btsp) in accept_types else btsp

        if type(err) == str:
            if err == 'fill':
                self.calc_err()
            elif err[-1] == '%':
                percent = float(err[:-1])
                dist = np.random.normal(percent, percent/4,
                                        size=self.val.shape)
                self.err = np.multiply(dist, self.val)/100

        if type(btsp) == str:
            if btsp == 'fill':
                self.calc_btsp()
            if btsp == 'seed':
                seed = kwargs['seed']
                self.calc_btsp(seed=seed)

    def calc_err(self):
        if type(self.btsp) == np.ndarray:
            self.err = np.zeros(shape=self.shape)
            btsp = np.moveaxis(self.btsp, 0, -1)
            for idx, central in np.ndenumerate(self.val):
                self.err[idx] = st_dev(btsp[idx], central)
        else:
            self.err = np.zeros(shape=self.shape)

    def calc_btsp(self, seed=None):
        if type(self.err) != np.ndarray:
            self.err = np.zeros(shape=self.shape)

        self.btsp = np.zeros(shape=self.shape+(self.N_boot,))
        for idx, central in np.ndenumerate(self.val):
            if seed != None:
                np.random.seed(seed)
            self.btsp[idx] = np.random.normal(
                central, self.err[idx], self.N_boot)
        self.btsp = np.moveaxis(self.btsp, -1, 0)

    def use_func(self, func, **kwargs):
        central = func(self.val, **kwargs)

        btsp = np.array([func(self.btsp[k,], **kwargs)
                         for k in range(N_boot)])

        return stat(val=central, err='fill', btsp=btsp)

    def __add__(self, other):
        if not isinstance(other, stat):
            other = np.array(other)
            other = stat(
                val=other,
                btsp='fill'
            )
        new_stat = stat(
            val=self.val+other.val,
            err='fill',
            btsp=np.array([self.btsp[k,]+other.btsp[k,]
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __sub__(self, other):
        if not isinstance(other, stat):
            other = np.array(other)
            other = stat(
                val=other,
                btsp='fill'
            )
        new_stat = stat(
            val=self.val-other.val,
            err='fill',
            btsp=np.array([self.btsp[k,]-other.btsp[k,]
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __mul__(self, other):
        if not isinstance(other, stat):
            other = np.array(other)
            other = stat(
                val=other,
                btsp='fill'
            )
        new_stat = stat(
            val=self.val*other.val,
            err='fill',
            btsp=np.array([self.btsp[k,]*other.btsp[k,]
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __matmul__(self, other):
        if not isinstance(other, stat):
            other = np.array(other)
            other = stat(
                val=other,
                btsp='fill'
            )
        new_stat = stat(
            val=self.val@other.val,
            err='fill',
            btsp=np.array([self.btsp[k]@other.btsp[k]
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __truediv__(self, other):
        if not isinstance(other, stat):
            other = np.array(other)
            other = stat(
                val=other,
                btsp='fill'
            )
        new_stat = stat(
            val=self.val/other.val,
            err='fill',
            btsp=np.array([self.btsp[k,]/other.btsp[k,]
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __pow__(self, num):
        new_stat = stat(
            val=self.val**num,
            err='fill',
            btsp=np.array([self.btsp[k,]**num
                           for k in range(self.N_boot)])
        )
        return new_stat

    def __neg__(self):
        new_stat = stat(
            val=-self.val,
            err=self.err,
            btsp=-self.btsp
        )
        return new_stat

    def __getitem__(self, indices):
        key = indices
        new_stat = stat(
            val=self.val[key],
            err=self.err[key],
            btsp=self.btsp[:, key]
        )
        return new_stat


def join_stats(stats):
    N = len(stats)
    return stat(
        val=np.array([s.val for s in stats]),
        err=np.array([s.err for s in stats]),
        btsp=np.array([s.btsp for s in stats]).swapaxes(0, 1)
    )


plt.rcParams.update(plotparams)
path = '/home/rm/external/NPR/'
N_d = 4  # Dirac indices
N_c = 3  # Color indices
# N_bl = 16 # number of bilinears
N_fq = 16  # number of fourquarks
N_boot = stat.N_boot  # number of bootstrap samples


dirs = ['X', 'Y', 'Z', 'T']
currents = ['S', 'P', 'V', 'A', 'T']
operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP', 'TT']
UKQCD_ens = ['C0', 'C1', 'C2',
             'M0', 'M1', 'M2', 'M3',
             'F1M', 'F1S']
KEK_ens = ['KEKC1L', 'KEKC1S',
           'KEKC2a', 'KEKC2b',
           'KEKM1a', 'KEKM1b',
           'KEKF1']
all_ens = UKQCD_ens+KEK_ens
phys_ens = ['C0', 'M0']

NPR_to_SUSY = np.zeros(shape=(len(operators), len(operators)))
NPR_to_SUSY[0, 0] = 1
NPR_to_SUSY[1, 3] = 1
NPR_to_SUSY[2, 3:] = -0.5, 0.5
NPR_to_SUSY[3, 2] = 1
NPR_to_SUSY[4, 1] = -0.5


def norm_factors(rotate=np.eye(len(operators)), **kwargs):
    N_NPR = np.array([8/3, -4/3, 2, -5/3, -1])
    return rotate@N_NPR

def chiral_logs(rotate=np.eye(len(operators)), obj='bag', **kwargs):
    bag_logs_SUSY = np.array([-0.5, -0.5, -0.5, 0.5, 0.5])
    ratio_logs_SUSY = np.array([1, 1.5, 1.5, 2.5, 2.5])
    if (rotate==NPR_to_SUSY).all():
        if obj=='bag':
            return bag_logs_SUSY
        elif obj=='ratio':
            return ratio_logs_SUSY
    elif (rotate==np.eye(len(operators))).all():
        if obj=='bag':
            return np.linalg.inv(NPR_to_SUSY)@bag_logs_SUSY
        elif obj=='ratio':
            return np.linalg.inv(NPR_to_SUSY)@ratio_logs_SUSY

flag_mus = [2.0, 3.0, 3.0, 3.0, 3.0]
flag = stat(
    val=[0.5570, 0.502, 0.766, 0.926, 0.720],
    err=[0.0071, 0.014, 0.032, 0.019, 0.038],
    btsp='fill')

flag_N = stat(
    val=norm_factors(rotate=NPR_to_SUSY)*flag.val,
    err='fill',
    btsp=np.array([norm_factors(rotate=NPR_to_SUSY)*flag.btsp[k,]
                   for k in range(N_boot)])
)

flag_vals, flag_errs = flag_N.val, flag_N.err
# =====gamma matrices=============================================
gamma = {'I': np.identity(N_d, dtype='complex128'),
         'X': np.zeros(shape=(N_d, N_d), dtype='complex128'),
         'Y': np.zeros(shape=(N_d, N_d), dtype='complex128'),
         'Z': np.zeros(shape=(N_d, N_d), dtype='complex128'),
         'T': np.zeros(shape=(N_d, N_d), dtype='complex128')}

for i in range(N_d):
    gamma['X'][i, N_d-i-1] = 1j if i <= 1 else -1j
    gamma['Y'][i, N_d-i-1] = 1 if (i == 1 or i == 2) else -1
    gamma['Z'][i, (i+2) % N_d] = (-1j) if (i == 1 or i == 2) else 1j
    gamma['T'][i, (i+2) % N_d] = 1

gamma['5'] = gamma['X']@gamma['Y']@gamma['Z']@gamma['T']
# =====put color structure into gamma matrices====================
Gamma = {name: np.einsum('ab,cd->acbd', mtx, np.identity(N_c)).reshape((12, 12))
         for name, mtx in gamma.items()}

# =====commutators===============================================


def commutator(str1, str2, g=Gamma):
    return (g[str1]@g[str2]-g[str2]@g[str1])


def anticommutator(str1, str2, g=Gamma):
    return (g[str1]@g[str2]+g[str2]@g[str1])
# ====bootstrap sampling==============================================


def bootstrap(data, seed=1, K=N_boot, **kwargs):
    ''' bootstrap samples generator - if input data has same size as K,
    assumes it's already a bootstrap sample and does no further sampling '''

    C = data.shape[0]
    if C == K:  # goes off when data itself is bootstrap data
        samples = data
    else:
        np.random.seed(seed)
        slicing = np.random.randint(0, C, size=(C, K))
        samples = np.mean(data[tuple(slicing.T if ax == 0 else slice(None)
                                     for ax in range(data.ndim))], axis=1)
    return np.array(samples, dtype=data.dtype)


def COV(data, **kwargs):
    ''' covariance matrix calculator - accounts for cov matrices
        centered around sample avg vs data avg and accordingly normalises'''

    C, T = data.shape

    if 'center' in kwargs.keys():
        center = kwargs['center']
        norm = C
    else:
        center = np.mean(data, axis=0)
        norm = C-1

    COV = np.array([[((data[:, t1]-center[t1]).dot(data[:, t2]-center[t2]))/norm
                    for t2 in range(T)] for t1 in range(T)])

    return COV


def m_eff(data, ansatz='cosh', **kwargs):
    if ansatz == 'cosh':
        m_eff = np.arccosh(0.5*(data[2:]+data[:-2])/data[1:-1])
    elif ansatz == 'exp':
        m_eff = np.abs(np.log(data[1:]/data[:-1]))
    return m_eff.real
# =====parsing filenames===================================================


def common_cf_files(data, corr, prefix=None):
    cfgs = os.listdir(data)[1:]
    cfgs.sort()
    file_names = {cf: os.listdir(f'{data}/{cf}/NPR/{corr}/')
                  for cf in cfgs}

    list_of_cf_files = []
    for cf in file_names.keys():
        for i in range(len(file_names[cf])):
            file_names[cf][i] = file_names[cf][i].rsplit(f'.{cf}.h5')[0]
            if prefix != None:
                file_names[cf][i] = file_names[cf][i].rsplit(prefix)[1]
        list_of_cf_files.append(file_names[cf])

    common_files = list(set.intersection(*map(set, list_of_cf_files)))
    common_files.sort()
    return common_files


def decode_prop(prop_name):
    info_list = prop_name.rsplit('prop_')[1].rsplit('_')
    Ls_idx = info_list.index('Ls')
    prop_info = {'prop': '_'.join(info_list[:Ls_idx]),
                 'Ls': info_list[Ls_idx+1],
                 'M5': info_list[info_list.index('M5')+1],
                 'am': info_list[info_list.index('am')+1],
                 'tw': '_'.join(info_list[info_list.index('tw')+1:info_list.index('tw')+5]),
                 'src_mom_p': '_'.join(info_list[info_list.index('p')+1:info_list.index('p')+5])}
    return prop_info


def encode_prop(prop_info):
    prop_name = ''
    for k, v in prop_info.items():
        prop_name += f'{k}_{v}_'
    return prop_name[:-1]


m_pi_plus_minus = stat(val=139.5709/1000, err=0.00018/1000, btsp='fill')
m_pi_0 = stat(val=134.9700/1000, err=0.0005/1000, btsp='fill')


m_pi_PDG = (m_pi_plus_minus*2 + m_pi_0)/3
f_pi_PDG = stat(val=130.41/1000, err=0.23/1000, btsp='fill')
m_K_PDG = stat(val=497.614/1000, err=0.024/1000, btsp='fill')
f_K_PDG = stat(val=158.1/1000, err=3.9/1000, btsp='fill')

m_f_sq_PDG = (m_pi_PDG/f_pi_PDG)**2
Lambda_QCD = 1.0

# ====coloring and markers=================================================
color_list = list(mc.TABLEAU_COLORS.keys())
marker_list = ['o', 's', 'D', 'x', '*', 'v', '^', 'h', '8']
