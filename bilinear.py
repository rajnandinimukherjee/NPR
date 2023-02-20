import numpy as np
from gamma_scheme import *
from NPR_structures import *
import h5py
from ensemble_parameters import *

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
        h5_path = f'{data}/{cfgs[0]}/NPR/{self.type}/{self.prefix}{self.filename}.{cfgs[0]}.h5'
        self.N_bl = len(h5py.File(h5_path, 'r')['Bilinear'].keys())

        self.N_cf = len(cfgs)
        self.bilinear = np.array([np.empty(shape=(self.N_cf,12,12),dtype='complex128')
                                  for i in range(self.N_bl)], dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.type}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['Bilinear']
            for i in range(self.N_bl):
                bilinear_i = h5_data[f'Bilinear_{i}']['corr'][0,0,:]
                self.bilinear[i][cf,:] = np.array(bilinear_i['re']+bilinear_i['im'
                                         ]*1j).swapaxes(1,2).reshape((12,12))
        self.pOut = [int(x) for x in h5_data['Bilinear_0']['info'
                    ].attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data['Bilinear_0']['info'
                   ].attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.prop_in = prop1 if prop1.mom==self.pIn else prop2
        self.prop_out = prop1 if prop1.mom==self.pOut else prop2
        self.tot_mom = self.prop_out.tot_mom-self.prop_in.tot_mom
        self.mom_sq = self.prop_in.mom_sq
        self.q = self.mom_sq**0.5

        self.org_gammas = np.array([h5_data[f'Bilinear_{i}']['info'
                          ].attrs['gamma'][0].decode() for i in range(self.N_bl)])
        self.gammas = [[x for x in list(self.org_gammas[i]) if x in list(gamma.keys())]
                                for i in range(self.N_bl)]
        self.avg_bilinear = np.array([np.mean(self.bilinear[i],axis=0)
                                      for i in range(self.N_bl)],dtype=object)
        self.amputated = np.array([self.prop_out.inv_out@self.avg_bilinear[i]@self.prop_in.inv
                                   for i in range(self.N_bl)])
        self.operator = {'S':[self.amputated[self.gammas.index(['I'])]],
                         'P':[self.amputated[self.gammas.index(['5'])]],
                         'V':[self.amputated[self.gammas.index([i])]
                             for i in dirs],
                         'A':[self.amputated[self.gammas.index([i,'5'])]
                             for i in dirs],
                         'T':sum([[self.amputated[self.gammas.index([dirs[i],
                             dirs[j]])] for j in range(i+1,4)] for i in range(0,4-1)], [])}
        
        #if scheme=='gamma':
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
