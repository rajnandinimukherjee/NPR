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

plt.rcParams.update(plotparams)
path = '/home/rm/external/NPR/'
N_d = 4  # Dirac indices
N_c = 3  # Color indices
# N_bl = 16 # number of bilinears
N_boot = 200  # number of bootstrap samples
N_fq = 16  # number of fourquarks
seed = 1  # random seed


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


flag_mus = [2.0, 3.0, 3.0, 3.0, 3.0]
flag_vals = [0.5570, 0.502, 0.766, 0.926, 0.720]
flag_errs = [0.0071, 0.014, 0.032, 0.019, 0.038]
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

# ====misc functions ==================================================


def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)


def err_disp(num, err, n=2, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    err_dec_place = int(np.floor(np.log10(np.abs(err))))
    err_n_digits = int(err*10**(-(err_dec_place+1-n)))
    num_dec_place = int(np.floor(np.log10(np.abs(num))))
    if num_dec_place < err_dec_place:
        # print('Error is larger than measurement')
        return str(np.around(num, 3))
    else:
        num_sf = num*10**(-(num_dec_place))
        num_trunc = round(num_sf, num_dec_place-(err_dec_place+1-n))
        digs = -err_dec_place+n-1
        str_num_trunc = str(num)[:digs+2]
        # num_nontrunc = round(num,-err_dec_place+n-1)
        # pdb.set_trace()
        # return str(num_trunc)+'('+str(err_n_digits)+')E%+d'%num_dec_place
        return str_num_trunc+'('+str(err_n_digits)+')'


# =====statistical obj class======================================


class stat:
    def __init__(self, val, err=None, btsp=None,
                 dtype=None, **kwargs):
        self.val = np.array(val)
        self.shape = self.val.shape
        self.dtype = self.val.dtype if dtype == None else dtype

        self.err = np.array(err)
        self.btsp = np.array(btsp)

        if type(btsp) == str:
            if btsp == 'fill':
                self.btsp = np.zeros(shape=self.shape+(N_boot,))
                for idx, central in np.ndenumerate(self.val):
                    np.random.seed(1)
                    self.btsp[idx] = np.random.normal(
                        central, self.err[idx], N_boot)
                self.btsp = np.moveaxis(self.btsp, -1, 0)

        if type(err) == str:
            if err == 'fill':
                self.err = np.zeros(shape=self.shape)
                btsp = np.moveaxis(self.btsp, 0, -1)
                for idx, central in np.ndenumerate(self.val):
                    self.err[idx] = st_dev(btsp[idx], central)


m_pi_PDG = stat(val=139.5709/1000, err=0.00018/1000, btsp='fill')
f_pi_PDG = stat(val=130.41/1000, err=0.23/1000, btsp='fill')

m_f_sq_PDG = stat(
    val=(m_pi_PDG.val/f_pi_PDG.val)**2,
    err='fill',
    btsp=(m_pi_PDG.btsp/f_pi_PDG.btsp)**2)

# ====coloring and markers=================================================
color_list = list(mc.TABLEAU_COLORS.keys())
marker_list = ['o', 's', 'D', 'x', '*', 'v', '^', 'h', '8']
