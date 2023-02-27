import numpy as np
import h5py
import os
import pdb
import matplotlib.pyplot as plt
from plot_settings import plotparams
from ensemble_parameters import *
from tqdm import tqdm
import pickle
plt.rcParams.update(plotparams)

path = '/home/rm/external/NPR/'
N_d = 4 # Dirac indices
N_c = 3 # Color indices
N_bl = 16 # number of bilinears
N_boot = 20 # number of bootstrap samples
N_fq = 16 # number of fourquarks
L = 24 # lattice spatial extent

dirs = ['X','Y','Z','T']
currents = ['S','P','V','A','T']
operators = ['VV+AA', 'VV-AA', 'SS-PP', 'SS+PP', 'TT']
#operators = ['VV+AA','VV-AA']

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

#=====bilinear projectors==============================
bl_proj = {'S':[Gamma['I']],
           'P':[Gamma['5']],
           'V':[Gamma[i] for i in dirs],
           'A':[Gamma[i]@Gamma['5'] for i in dirs],
           'T':sum([[Gamma[dirs[i]]@Gamma[dirs[j]]
               for j in range(i+1,4)] for i in range(0,4-1)],[])}
bl_gamma_F = {k:np.trace(np.sum([mtx@mtx for mtx in bl_proj[k]],axis=0))
              for k in currents}
#=====fourquark projectors=======================================
fq_proj = {'SS':np.einsum('ab,cd->abcd',Gamma['I'],Gamma['I']),
        'PP':np.einsum('ab,cd->abcd',Gamma['5'],Gamma['5']),
        'VV':np.sum(np.array([np.einsum('ab,cd->abcd',Gamma[i],Gamma[i])
             for i in dirs]), axis=0),
        'AA':np.sum(np.array([np.einsum('ab,cd->abcd',
             Gamma[i]@Gamma['5'],Gamma[i]@Gamma['5'])
             for i in dirs]), axis=0),
        'TT':np.sum(np.array(sum([[np.einsum('ab,cd->abcd',
             Gamma[dirs[i]]@Gamma[dirs[j]],Gamma[dirs[i]]@Gamma[dirs[j]])
             for j in range(i+1,4)] for i in range(0,4-1)],[])), axis=0)}

fq_gamma_projector = {'VV+AA':fq_proj['VV']+fq_proj['AA'],
                  'VV-AA':fq_proj['VV']-fq_proj['AA'],
                  'SS-PP':fq_proj['SS']-fq_proj['PP'],
                  'SS+PP':fq_proj['SS']+fq_proj['PP'],
                  'TT':fq_proj['TT']}

fq_gamma_F = np.array([[np.einsum('abcd,badc',2*(fq_gamma_projector[k1]-fq_gamma_projector[k1].swapaxes(1,3)),
               fq_gamma_projector[k2]) for k2 in operators] for k1 in operators])

def fq_qslash_projector(q, p1, p2, **kwargs):
    qslash = np.sum([q[i]*gamma[dirs[i]] for i in range(len(dirs))], axis=0)
    q_sq = np.linalg.norm(qslash)**2
    p1_sq = np.linalg.norm(p1)**2
    p2_sq = np.linalg.norm(p2)**2
    p1dotp2 = p1.dot(p2)

    proj = {'VV':np.einsum('ab,cd->abcd',qslash,qslash)/q_sq,
            'AA':np.einsum('ab,cd->abcd',qslash@gamma['5'],qslash@gamma['5'])/q_sq}
    P_L = (np.identity(4)-gamma['5'])/2
    p_sig_P_p = np.sum([np.sum([p1[dirs.index(i)]*((commutator(i,j,
                          g=gamma)/2)@P_L)*p2[dirs.index(j)] for j in dirs],axis=0) for i in dirs],axis=0)
    proj['TT'] = np.einsum('ab,cd->abcd',p_sig_P_p, p_sig_P_p)/(p1_sq*p2_sq - p1dotp2**2)

    q_proj = {'VV+AA':np.einsum('abcd,ef,gh->abcdefgh',
                      proj['VV']+proj['AA'],np.identity(N_c), 
                      np.identity(N_c)),
              'VV-AA':np.einsum('abcd,ef,gh->abcdefgh',
                      proj['VV']-proj['AA'],np.identity(N_c),
                      np.identity(N_c))}
    q_proj['SS-PP'] = q_proj['VV-AA'].swapaxes(4,6)
    q_proj['TT'] = np.einsum('abcd,ef,gh->abcdefgh',proj['TT'],
                             np.identity(N_c),np.identity(N_c))
    q_proj['TT'] = np.moveaxis(np.moveaxis(np.moveaxis(q_proj['TT'],4,1),5,3),6,5)
    q_proj['SS+PP'] = q_proj['TT'].swapaxes(4,6)
    for k in q_proj.keys():
        print(q_proj[k].shape)
    return {k:np.q_proj[k].reshape((12,12,12,12)) for k in operators}


mask = np.full((len(operators),len(operators)),False)
for i in range(len(operators)):
    mask[i,i]=True
mask[1,2], mask[2,1] = True, True
mask[3,4], mask[4,3] = True, True 
#=====filenames===================================================
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


class external:
    def __init__(self, ensemble, mom=[-3,-3,0,0], tw=[0.00, 0.00, 0.00, 0.00],
                 prop='fgauge_SDWF', Ls='16', M5='1.80', am='0.0100',
                 filename='', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        cfgs = os.listdir(data)[1:]
        cfgs.sort()
        N_cf = len(cfgs) # number of configs)
        L = params[ensemble]['XX']
        self.type ='externalleg'
        self.prefix = 'external_'
        if filename != '':
            self.filename = filename
            self.info = decode_prop(self.filename)
            self.mom = [int(i) for i in self.info['src_mom_p'].rsplit('_')]
            self.tw = [float(i) for i in self.info['tw'].rsplit('_')]
        else:
            self.mom = mom
            self.tw = tw
            self.info = {'prop':prop,
                         'Ls':Ls,
                         'M5':M5,
                         'am':am,
                         'tw':'_'.join(['%.2f'%i for i in self.tw]),
                         'src_mom_p':'_'.join([str(i) for i in self.mom])}
            self.filename = encode_prop(self.info)

        self.tot_mom = (2*np.pi*a_inv/L)*(np.array(self.mom)+np.array(self.tw))
        self.norm = np.linalg.norm(self.tot_mom)
        self.mom_sq = self.norm**2
        
        self.N_cf = len(cfgs)
        self.data = np.empty(shape=(self.N_cf,12,12),dtype='complex128')
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.type}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['ExternalLeg']['corr'][0,0,:]
            self.data[cf,:] = np.array(h5_data['re']+h5_data['im']*1j).swapaxes(1,
                                2).reshape((12,12))

        self.prop = np.mean(self.data,axis=0)
        self.dagger = self.prop.conj().T
        self.prop_out = Gamma['5']@self.dagger@Gamma['5']
        self.inv = np.linalg.inv(self.prop)
        self.inv_out = np.linalg.inv(self.prop_out)
            

    def errs(self):
        self.samples = bootstrap(self.data)
        self.samples_out = np.array([Gamma['5']@(self.samples[k].conj().T)@Gamma['5']
                                    for k in range(N_boot)])
        self.samples_inv = np.array([np.linalg.inv(self.samples[k])
                                    for k in range(N_boot)])
        self.samples_out = np.array([np.linalg.inv(self.samples_out[k])
                                    for k in range(N_boot)])



class bilinear:
    def __init__(self, ensemble, prop1, prop2, scheme='gamma', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        cfgs = os.listdir(data)[1:]
        cfgs.sort()
        N_cf = len(cfgs) # number of configs)
        L = params[ensemble]['XX']
        self.type = 'bilinears'
        self.prefix = 'bi_'
        self.filename = prop1.filename+'__'+prop2.filename

        self.N_cf = len(cfgs)
        self.bilinear = np.array([np.empty(shape=(self.N_cf,12,12),dtype='complex128')
                                  for i in range(N_bl)], dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.type}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['Bilinear']
            for i in range(N_bl):
                bilinear_i = h5_data[f'Bilinear_{i}']['corr'][0,0,:]
                self.bilinear[i][cf,:] = np.array(bilinear_i['re']+bilinear_i['im']*1j).swapaxes(1,
                                                   2).reshape((12,12))
        self.pOut = [int(x) for x in h5_data['Bilinear_0']['info'].attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data['Bilinear_0']['info'].attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.prop_in = prop1 if prop1.mom==self.pIn else prop2
        self.prop_out = prop1 if prop1.mom==self.pOut else prop2
        self.tot_mom = self.prop_out.tot_mom-self.prop_in.tot_mom
        self.mom_sq = self.prop_in.mom_sq
        self.q = self.mom_sq**0.5

        self.org_gammas = np.array([h5_data[f'Bilinear_{i}']['info'].attrs['gamma'][0].decode()
                                          for i in range(N_bl)])
        self.gammas = [[x for x in list(self.org_gammas[i]) if x in list(gamma.keys())]
                                for i in range(N_bl)]
        self.avg_bilinear = np.array([np.mean(self.bilinear[i],axis=0)
                                      for i in range(N_bl)],dtype=object)
        self.amputated = np.array([self.prop_out.inv_out@self.avg_bilinear[i]@self.prop_in.inv
                                   for i in range(N_bl)])
        self.operator = {'S':[self.amputated[self.gammas.index(['I'])]],
                         'P':[self.amputated[self.gammas.index(['5'])]],
                         'V':[self.amputated[self.gammas.index([i])]
                             for i in dirs],
                         'A':[self.amputated[self.gammas.index([i,'5'])]
                             for i in dirs],
                         'T':sum([[self.amputated[self.gammas.index([dirs[i],
                             dirs[j]])] for j in range(i+1,4)] for i in range(0,4-1)], [])}
        
        if scheme=='gamma':
            self.projector = bl_proj
            self.F = bl_gamma_F

        self.projected = {k:np.trace(np.sum([self.projector[k][i]@self.operator[k][i]
                          for i in range(len(self.operator[k]))],axis=0)) for k in currents}
        self.proj_arr = np.array([(self.projected[k]/self.F[k]).real
                                 for k in currents])
        self.qslash = np.sum([self.tot_mom[i]*Gamma[dirs[i]] for i in range(len(dirs))], axis=0)
        self.qslash_V = np.sum([self.tot_mom[i]*np.trace(self.qslash@self.amputated[self.gammas.index([dirs[i]])])
                                for i in range(len(dirs))], axis=0)/(12*self.mom_sq)

    def errs(self):
        self.prop_in.errs()
        self.prop_out.errs()

        self.samp_bl = np.array([bootstrap(self.bilinear[i]) for i in range(N_bl)])
        self.samp_proj = {k:np.zeros(N_boot,dtype='complex128') for k in currents}
        self.proj_err = {k:0 for k in currents}
        for b in range(N_boot):
            inv_out, inv = self.prop_out.samples_out[b,], self.prop_in.samples_inv[b,]
            amputated = np.array([inv_out@self.samp_bl[i][b,]@inv for i in range(N_bl)])
            operator = {'S':[amputated[self.gammas.index(['I'])]],
                        'P':[amputated[self.gammas.index(['5'])]],
                        'V':[amputated[self.gammas.index([i])]
                                 for i in dirs],
                        'A':[amputated[self.gammas.index([i,'5'])]
                                 for i in dirs],
                        'T':sum([[amputated[self.gammas.index([dirs[i],
                                 dirs[j]])] for j in range(i+1,4)] for i in range(0,4-1)], [])}
            for k in currents:
                self.samp_proj[k][b] = np.trace(np.sum([self.projector[k][i]@operator[k][i]
                                                       for i in range(len(operator[k]))],axis=0))
                self.samp_proj[k][b] = (self.samp_proj[k][b]/self.F[k]).real

        for k in currents:
            diff = self.samp_proj[k]-self.proj_arr[currents.index(k)]
            self.proj_err[k] = (diff.dot(diff)/N_boot)**0.5

class fourquark:
    def __init__(self, ensemble, prop1, prop2, scheme='gamma', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        cfgs = os.listdir(data)[1:]
        cfgs.sort()
        N_cf = len(cfgs) # number of configs)
        L = params[ensemble]['XX']
        self.type = 'fourquarks'
        self.prefix = 'fourquarks_'
        self.filename = prop1.filename+'__'+prop2.filename
        fq_str = 'FourQuarkFullyConnected'
        self.scheme = 'gamma'

        self.N_cf = len(cfgs)
        self.fourquark = np.array([np.empty(shape=(self.N_cf,12,12,12,12),dtype='complex128')
                                  for i in range(N_fq)], dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.type}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')[fq_str]
            for i in range(N_fq):
                fourquark_i = h5_data[f'{fq_str}_{i}']['corr'][0,0,:]
                self.fourquark[i][cf,:] = np.array(fourquark_i['re']+fourquark_i['im']*1j).swapaxes(1,
                                          2).swapaxes(5,6).reshape((12,12,12,12))
        self.pOut = [int(x) for x in h5_data[f'{fq_str}_0']['info'].attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data[f'{fq_str}_0']['info'].attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.prop_in = prop1 if prop1.mom==self.pIn else prop2
        self.prop_out = prop1 if prop1.mom==self.pOut else prop2
        self.tot_mom = self.prop_out.tot_mom-self.prop_in.tot_mom
        self.mom_sq = self.prop_in.mom_sq
        self.q = self.mom_sq**0.5

        self.org_gammas = [[h5_data[f'{fq_str}_{i}']['info'].attrs['gammaA'][0].decode(),
                        h5_data[f'{fq_str}_{i}']['info'].attrs['gammaB'][0].decode()]
                        for i in range(N_fq)]
        self.gammas = [[[x for x in list(self.org_gammas[i][0]) if x in list(gamma.keys())],
                        [y for y in list(self.org_gammas[i][1]) if y in list(gamma.keys())]]
                        for i in range(N_fq)]

        self.doublets = {'SS':self.fourquark[self.gammas.index([['I'],['I']])],
                         'PP':self.fourquark[self.gammas.index([['5'],['5']])],
                         'VV':np.sum(np.array([self.fourquark[self.gammas.index([[i],[i]])]
                              for i in dirs]), axis=0),
                         'AA':np.sum(np.array([self.fourquark[self.gammas.index([[i,'5'],[i,'5']])]
                              for i in dirs]), axis=0),
                         'TT':np.sum(np.array(sum([[self.fourquark[self.gammas.index([[dirs[i],
                              dirs[j]],[dirs[i],dirs[j]]])] for j in range(i+1,4)] 
                              for i in range(0,4-1)], [])), axis=0)}
        self.operator = {'VV+AA':self.doublets['VV']+self.doublets['AA'],
                         'VV-AA':self.doublets['VV']-self.doublets['AA'],
                         'SS-PP':self.doublets['SS']-self.doublets['PP'],
                         'SS+PP':self.doublets['SS']+self.doublets['PP'],
                         'TT':self.doublets['TT']}
        
        self.amputated = {}
        in_, out_ = self.prop_in.inv, self.prop_out.inv_out
        for k in operators:
            op = 2*(self.operator[k]-self.operator[k].swapaxes(2,4))
            op_avg = np.array(np.mean(op, axis=0),dtype='complex128')
            self.amputated[k] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                          out_,in_,out_,in_,op_avg) 

        if self.scheme=='gamma':
            self.projector = fq_gamma_projector
            self.F = fq_gamma_F

        self.projected = np.array([[np.einsum('abcd,badc',self.projector[k1],
                          self.amputated[k2]) for k2 in operators]
                          for k1 in operators]) 
        self.bl = bilinear(ensemble, self.prop_in, self.prop_out, cfgs=cfgs)
        self.Z_V = ((self.F/(self.bl.F['V']**2))@np.linalg.inv(self.projected.T/self.bl.projected['V']**2)).real 
        self.Z_V = np.multiply(self.Z_V,mask)
        self.Z_A = ((self.F/self.bl.F['A']**2)@np.linalg.inv(self.projected/self.bl.projected['A']**2)).real
        self.Z_S = np.multiply(self.Z_A,mask)

    def errs(self):
        self.prop_in.errs()
        self.prop_out.errs()
        self.bl.errs()

        self.samples = {k:bootstrap(self.operator[k]) for k in operators}

        self.samples_Z_V = np.zeros(shape=(N_boot,len(operators),len(operators)))
        for k in tqdm(range(N_boot), leave=False):
        #for k in range(N_boot):
            in_, out_ = self.prop_in.samples_inv[k], self.prop_out.samples_out[k]
            samples_amp = {}
            for key in self.samples.keys():
                samples_op = 2*(self.samples[key][k,]-self.samples[key][k,].swapaxes(1,3))
                samples_amp[key] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                        out_,in_,out_,in_,samples_op)
            samples_proj = np.array([[np.einsum('abcd,badc',self.projector[k1],
                                              samples_amp[k2]) for k2 in operators]
                                              for k1 in operators])
            self.samples_Z_V[k,:,:] = (self.F@np.linalg.inv(samples_proj.T/(self.bl.samp_proj['V'][k]**2))).real
        self.Z_V_errs = np.zeros(shape=(len(operators),len(operators)))
        for i in range(len(operators)):
            for j in range(len(operators)):
                diff = self.samples_Z_V[:,i,j]-self.Z_V[i,j]
                self.Z_V_errs[i,j] = (diff.dot(diff)/N_boot)**0.5
