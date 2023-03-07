import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from plot_settings import plotparams
from ensemble_parameters import *
from tqdm import tqdm
plt.rcParams.update(plotparams)
import os
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import itertools
from matplotlib.ticker import FormatStrFormatter 

path = '/home/rm/external/NPR/'
N_d = 4 # Dirac indices
N_c = 3 # Color indices
#N_bl = 16 # number of bilinears
N_boot = 200 # number of bootstrap samples
N_fq = 16 # number of fourquarks

dirs = ['X','Y','Z','T']
currents = ['S','P','V','A','T']
operators = ['VV+AA', 'VV-AA', 'SS-PP', 'SS+PP', 'TT']
UKQCD_ens = ['C0', 'C1', 'C2',
             'M0', 'M1', 'M2', 'M3',
             'F1M', 'F1S']
KEK_ens = ['KEKC1L', 'KEKC1S',
           'KEKC2a', 'KEKC2b',
           'KEKM1a', 'KEKM1b',
           'KEKF1']
all_ens = UKQCD_ens+KEK_ens
phys_ens = ['C0', 'M0']

#=====gamma matrices=============================================
gamma = {'I':np.identity(N_d,dtype='complex128'),
         'X':np.zeros(shape=(N_d,N_d),dtype='complex128'),
         'Y':np.zeros(shape=(N_d,N_d),dtype='complex128'),
         'Z':np.zeros(shape=(N_d,N_d),dtype='complex128'),
         'T':np.zeros(shape=(N_d,N_d),dtype='complex128')}

for i in range(N_d):
    gamma['X'][i,N_d-i-1] = 1j if i<=1 else -1j
    gamma['Y'][i,N_d-i-1] = 1 if (i==1 or i==2) else -1
    gamma['Z'][i,(i+2)%N_d] = (-1j) if (i==1 or i==2) else 1j
    gamma['T'][i,(i+2)%N_d] = 1

gamma['5'] = gamma['X']@gamma['Y']@gamma['Z']@gamma['T']
#=====put color structure into gamma matrices====================
Gamma = {name:np.einsum('ab,cd->acbd',mtx,np.identity(N_c)).reshape((12,12))
         for name, mtx in gamma.items()}

#=====commutators===============================================
def commutator(str1, str2, g=Gamma):
    return (g[str1]@g[str2]-g[str2]@g[str1])

def anticommutator(str1, str2, g=Gamma):
    return (g[str1]@g[str2]+g[str2]@g[str1])

#=====bootstrap sampling========================================
def bootstrap(data, seed=1, K=N_boot, sigma=None, **kwargs):
    ''' bootstrap samples generator - if input data has same size as K,
    assumes it's already a bootstrap sample and does no further sampling '''
    
    C = data.shape[0]
    if C==K: # goes off when data itself is bootstrap data
        samples = data
    else:
        np.random.seed(seed)
        slicing = np.random.randint(0, C, size=(C, K))
        samples = np.mean(data[tuple(slicing.T if ax==0 else slice(None)
                                for ax in range(data.ndim))],axis=1)
    return np.array(samples,dtype='complex128')

#=====parsing filenames===================================================
def common_cf_files(data, corr, prefix=None):
    cfgs = os.listdir(data)[1:]
    cfgs.sort()
    file_names = {cf:os.listdir(f'{data}/{cf}/NPR/{corr}/')
                  for cf in cfgs}

    list_of_cf_files = []
    for cf in file_names.keys():
        for i in range(len(file_names[cf])):
            file_names[cf][i] = file_names[cf][i].rsplit(f'.{cf}.h5')[0]
            if prefix != None:
                file_names[cf][i] = file_names[cf][i].rsplit(prefix)[1]
        list_of_cf_files.append(file_names[cf])

    common_files = list(set.intersection(*map(set,list_of_cf_files)))
    common_files.sort()
    return common_files


def decode_prop(prop_name):
    info_list = prop_name.rsplit('prop_')[1].rsplit('_')
    Ls_idx = info_list.index('Ls')
    prop_info = {'prop':'_'.join(info_list[:Ls_idx]),
                 'Ls':info_list[Ls_idx+1],
                 'M5':info_list[info_list.index('M5')+1], 
                 'am':info_list[info_list.index('am')+1],
                 'tw':'_'.join(info_list[info_list.index('tw')+1:info_list.index('tw')+5]),
                 'src_mom_p':'_'.join(info_list[info_list.index('p')+1:info_list.index('p')+5])}
    return prop_info

def encode_prop(prop_info):
    prop_name=''
    for k, v in prop_info.items():
        prop_name += f'{k}_{v}_'
    return prop_name[:-1]

