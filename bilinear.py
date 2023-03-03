from externalleg import *

#=====bilinear projectors==============================
gamma_proj = {'S':[Gamma['I']],
           'P':[Gamma['5']],
           'V':[Gamma[i] for i in dirs],
           'A':[Gamma[i]@Gamma['5'] for i in dirs],
           'T':sum([[Gamma[dirs[i]]@Gamma[dirs[j]]
               for j in range(i+1,4)] for i in range(0,4-1)],[])}

gamma_F = {k:np.trace(np.sum([mtx@mtx
           for mtx in gamma_proj[k]],axis=0))
           for k in currents}

class bilinear:
    currents = ['S','P','V','A','T']
    schemes = ['gamma','qslash']
    obj = 'bilinears'
    prefix = 'bi_'
    N_boot = N_boot
    def __init__(self, ensemble, prop1, prop2, scheme='gamma', **kwargs):

        data = path+ensemble
        self.a_inv = params[ensemble]['ainv'] 
        self.L = params[ensemble]['XX']
        cfgs = sorted(os.listdir(data)[1:])
        self.N_cf = len(cfgs) 
        self.filename = prop1.filename+'__'+prop2.filename
        h5_path = f'{data}/{cfgs[0]}/NPR/{self.obj}/{self.prefix}{self.filename}.{cfgs[0]}.h5'
        self.N_bl = len(h5py.File(h5_path, 'r')['Bilinear'].keys())

        self.bilinears = np.array([np.empty(shape=(self.N_cf,12,12),
                                  dtype='complex128') for i in range(self.N_bl)], 
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

        self.prop_in = prop1 if prop1.momentum==self.pIn else prop2
        self.prop_out = prop1 if prop1.momentum==self.pOut else prop2
        self.tot_mom = self.prop_out.total_momentum-self.prop_in.total_momentum
        self.mom_sq = self.prop_in.momentum_squared
        self.q = self.mom_sq**0.5

        self.avg_bilinear = np.array([np.mean(self.bilinears[i],axis=0)
                                      for i in range(self.N_bl)],dtype=object)
        self.btsp_bilinear = np.array([bootstrap(self.bilinears[i], K=self.N_boot)
                                       for i in range(self.N_bl)])

    def gamma_Z(self, operators, **kwargs):
        gamma_Z = {}
        for c in bilinear.currents: 
            projected = np.trace(np.sum([gamma_proj[c][i]@operators[c][i]
                               for i in range(len(operators[c]))],axis=0))
            gamma_Z[c] = (gamma_F[c]/projected).real
        return gamma_Z 

    def qslash_Z(self, operators, Z_q, q_vec, **kwargs):
        qslash = np.sum([q_vec[i]*Gamma[dirs[i]]
                        for i in range(len(dirs))], axis=0)
        q_sq = np.linalg.norm(q_vec)**2
        qslash_Z = {}
        qslash_Z['P'] = Z_q*self.gamma_Z(operators)['P']
        qslash_Z['T'] = Z_q*self.gamma_Z(operators)['T']
        qslash_Z['V'] = Z_q/(np.trace(np.sum([q_vec[i]*operators['V'][i]
                       for i in range(len(dirs))],axis=0)@qslash).real/(12*q_sq))

        m_q = float(self.prop_in.info['am'])
        _1 = Z_q
        _2 = Z_q/qslash_Z['P'] 
        _3 = np.trace(2*1j*m_q*operators['P'][0]@Gamma['5']@qslash).real/(12*q_sq)
        _4 = np.trace(np.sum([q_vec[i]*operators['A'][i]
             for i in range(len(dirs))],axis=0)@Gamma['5']).real/(12*m_q)
        numerator = _4*_2*((_3/_1)+2*(_1/_4)+2*(_3/_4))
        denominator = 2*(_1**2) + 2*_1*_3 + _3*_4
        qslash_Z['m'] = numerator/denominator
        a_term  = np.trace(np.sum([q_vec[i]*operators['A'][i]
                           for i in range(len(dirs))],
                           axis=0)@Gamma['5']@qslash).real/(12*q_sq)
        mass_term = qslash_Z['m']*_3/_2
        qslash_Z['A'] = Z_q*(1+mass_term)/a_term

        s_term = np.trace(operators['S'][0]).real/12
        mass_term = 2*_3*qslash_Z['m']/_2
        qslash_Z['S'] = Z_q*(1+mass_term)/s_term
        return qslash_Z

    def construct_operators(self, S_in, S_out, Gs, **kwargs):
        amputated = np.array([S_out@Gs[b]@S_in for b in range(self.N_bl)])
        operators = {'S':[amputated[self.gammas.index(['I'])]],
                    'P':[amputated[self.gammas.index(['5'])]],
                    'V':[amputated[self.gammas.index([i])]
                             for i in dirs],
                    'A':[amputated[self.gammas.index([i,'5'])]
                             for i in dirs],
                    'T':sum([[amputated[self.gammas.index([dirs[i],
                             dirs[j]])] for j in range(i+1,4)]
                             for i in range(0,4-1)], [])}
        return operators

    def NPR(self, massive=False, **kwargs):
        #==central===
        S_in = self.prop_in.inv_avg_propagator
        S_out = self.prop_out.inv_outgoing_avg_propagator
        operators = self.construct_operators(S_in, S_out, self.avg_bilinear)
        if not massive:
            self.Z = self.gamma_Z(operators)
        else:
            Z_q = self.prop_in.Z_q_avg_qslash
            self.Z = self.qslash_Z(operators, Z_q, self.tot_mom) 
        #==bootstrap===
        self.Z_btsp = {c:np.zeros(self.N_boot) for c in self.Z.keys()}
        for k in range(self.N_boot):
            S_in = self.prop_in.btsp_inv_propagator[k,:,:]
            S_out = self.prop_out.btsp_inv_outgoing_propagator[k,:,:]
            operators = self.construct_operators(S_in, S_out,
                        self.btsp_bilinear[:,k,:])
            if not massive:
                Z_k = self.gamma_Z(operators)
                for c in Z_k.keys():
                    self.Z_btsp[c][k] = Z_k[c]
            else:
                Z_q = self.prop_in.Z_q_btsp_qslash[k]
                Z_k = self.qslash_Z(operators, Z_q, self.tot_mom)
                for c in Z_k.keys():
                    self.Z_btsp[c][k] = Z_k[c]

        #==errors===
        self.Z_err = {}
        for c in self.Z.keys():
            self.Z_err[c] = ((self.Z_btsp[c][:]-self.Z[c]).dot(
                            self.Z_btsp[c][:]-self.Z[c])/self.N_boot)**0.5

