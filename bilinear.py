from NPR_structures import *
from externalleg import *

#=====bilinear projectors==============================
gamma_proj = {'S':[Gamma['I']],
           'P':[Gamma['5']],
           'V':[Gamma[i] for i in dirs],
           'A':[Gamma[i]@Gamma['5'] for i in dirs],
           'T':sum([[Gamma[dirs[i]]@Gamma[dirs[j]]
               for j in range(i+1,4)] for i in range(0,4-1)],[])}

gamma_F = {k:np.trace(np.sum([mtx@mtx
           for mtx in bl_proj[k]],axis=0))
           for k in currents}

def gamma_Z(operator, current, **kwargs):
    projected = np.trace(np.sum([bl_proj[current][i]@operator[i]
                       for i in range(len(operator))],axis=0))
    gamma_Z = (bl_gamma_F[current]/projected).real
    return gamma_Z 

def qslash_Z(operator, p_vec, current, massive=False, **kwargs):
    qslash = np.sum([p_vec[i]*Gamma[dirs[i]]
                    for i in range(len(dirs))], axis=0)
    q_sq = np.linalg.norm(p_vec)**2
    if current in ['S', 'P', 'T']:
        qslash_Z = gamma_Z(operator, current)
    elif current=='V':
        qslash_Z = np.trace(np.sum([p_vec[i]*operator[i]
                   for i in range(len(dirs))])@qslash)/(12*q_sq)
    elif current=='A':
        mass_term = 




def bilinear_Zs(S_in_inv, S_out_inv, G, p_vec, m_q, 
                gammas, massive=False, **kwargs):

    amputated = np.array([S_out_inv@G[i]@S_in_inv 
                         for i in range(self.N_bl)])
    operator = {'S':[amputated[gammas.index(['I'])]],
                'P':[amputated[gammas.index(['5'])]],
                'V':[amputated[gammas.index([i])]
                    for i in dirs],
                'A':[amputated[gammas.index([i,'5'])]
                    for i in dirs],
                'T':sum([[amputated[gammas.index([dirs[i],
                    dirs[j]])] for j in range(i+1,4)]
                    for i in range(0,4-1)], [])}
    
    Z['q']['qslash'] = (np.trace(1j*S_in_inv)/(12*p_vec.dot(p_vec))).real
        Z[c]['gamma'] = normalised
        if c in ['S','P','T']:
            Z[c]['qslash'] = normalised*Z['q']['qslash']

    Z['V']['qslash'] = 

    


class bilinear:
    currents = ['S','P','V','A','T']
    schemes = ['gamma','qslash']
    obj = 'bilinears'
    prefix = 'bi_'
    def __init__(self, ensemble, prop1, prop2, scheme='gamma', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        L = params[ensemble]['XX']
        cfgs = sorted(os.listdir(data)[1:])
        self.N_cf = len(cfgs) 
        self.filename = prop1.filename+'__'+prop2.filename
        h5_path = f'{data}/{cfgs[0]}/NPR/{self.type}/{self.obj}{self.filename}.{cfgs[0]}.h5'
        self.N_bl = len(h5py.File(h5_path, 'r')['Bilinear'].keys())

        self.bilinears = np.array([np.empty(shape=(self.N_cf,12,12),
                                  dtype='complex128')
                                  for i in range(self.N_bl)], 
                                  dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.obj}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['Bilinear']
            for i in range(self.N_bl):
                bilinear_i = h5_data[f'Bilinear_{i}']['corr'][0,0,:]
                self.bilinears[i][cf,:] = np.array(bilinear_i['re']+bilinear_i['im'
                                         ]*1j).swapaxes(1,2).reshape((12,12))
        self.pOut = [int(x) for x in h5_data['Bilinear_0']['info'
                    ].attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data['Bilinear_0']['info'
                   ].attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.org_gammas = np.array([h5_data[f'Bilinear_{i}']['info'
                          ].attrs['gamma'][0].decode() 
                          for i in range(self.N_bl)])
        self.gammas = [[x for x in list(self.org_gammas[i])
                        if x in list(gamma.keys())]
                        for i in range(self.N_bl)]

        self.prop_in = prop1 if prop1.mom==self.pIn else prop2
        self.prop_out = prop1 if prop1.mom==self.pOut else prop2
        self.tot_mom = self.prop_out.total_momentum-self.prop_in.total_momentum
        self.mom_sq = self.prop_in.momentum_squared
        self.q = self.mom_sq**0.5

        self.avg_bilinear = np.array([np.mean(self.bilinear[i],axis=0)
                                      for i in range(self.N_bl)],dtype=object)
    def compute_Zs
        
        if scheme=='gamma':
            self.projector = bl_proj
            self.F = bl_gamma_F

            self.projected = {k:np.trace(np.sum([self.projector[k][i]@self.operator[k][i]
                              for i in range(len(self.operator[k]))],axis=0)) for k in currents}
            self.proj_arr = np.array([(self.projected[k]/self.F[k]).real
                                     for k in currents])
        if scheme=='qslash':
            self.qslash = np.sum([self.tot_mom[i]*Gamma[dirs[i]]
                            for i in range(len(dirs))], axis=0)
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
