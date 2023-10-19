from externalleg import *
from eta_c import *

# =====bilinear projectors==============================
bl_gamma_proj = {'S': [Gamma['I']],
                 'P': [Gamma['5']],
                 'V': [Gamma[i] for i in dirs],
                 'A': [Gamma[i]@Gamma['5'] for i in dirs],
                 'T': sum([[Gamma[dirs[i]]@Gamma[dirs[j]]
                            for j in range(i+1, 4)]
                           for i in range(0, 4-1)], [])}

bl_gamma_F = {k: np.trace(np.sum([mtx@mtx
                                  for mtx in bl_gamma_proj[k]], axis=0))
              for k in currents}


class bilinear:
    currents = ['S', 'P', 'V', 'A', 'T']
    schemes = ['gamma', 'qslash']
    obj = 'bilinears'
    prefix = 'bi_'

    def __init__(self, ensemble, prop1, prop2, scheme='gamma',
                 mres=True, **kwargs):

        data = path+ensemble
        self.L = params[ensemble]['XX']
        cfgs = sorted(os.listdir(data)[1:])
        self.N_cf = len(cfgs)
        self.filename = prop1.filename+'__'+prop2.filename
        h5_path = f'{data}/{cfgs[0]}/NPR/{self.obj}/{self.prefix}{self.filename}.{cfgs[0]}.h5'
        self.N_bl = len(h5py.File(h5_path, 'r')['Bilinear'].keys())

        self.bilinears = np.array([np.empty(shape=(self.N_cf, 12, 12),
                                  dtype='complex128') for i in range(self.N_bl)],
                                  dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.obj}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['Bilinear']
            for i in range(self.N_bl):
                bilinear_i = h5_data[f'Bilinear_{i}']['corr'][0, 0, :]
                self.bilinears[i][cf, :] = np.array(
                    bilinear_i['re']+bilinear_i['im']*1j).\
                    swapaxes(1, 2).reshape((12, 12))
        self.pOut = [int(x) for x in h5_data['Bilinear_0']['info'].
                     attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data['Bilinear_0']['info'].
                    attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.org_gammas = np.array([h5_data[f'Bilinear_{i}']['info'].
                                    attrs['gamma'][0].decode()
                                    for i in range(self.N_bl)])
        self.gammas = [[x for x in list(self.org_gammas[i])
                        if x in list(gamma.keys())]
                       for i in range(self.N_bl)]

        self.prop_in = prop1 if prop1.momentum == self.pIn else prop2
        self.prop_out = prop1 if prop1.momentum == self.pOut else prop2
        self.tot_mom = self.prop_out.total_momentum-self.prop_in.total_momentum
        self.mom_sq = self.prop_in.momentum_squared
        self.q = self.mom_sq**0.5

        self.avg_bilinear = np.array([np.mean(self.bilinears[i], axis=0)
                                      for i in range(self.N_bl)], dtype=object)
        self.btsp_bilinear = np.array([bootstrap(self.bilinears[i], K=N_boot)
                                       for i in range(self.N_bl)])

        self.m_q = float(self.prop_in.info['am'])
        self.m_pole = self.pole_mass()
        self.mres = 0
        if ensemble in valence_ens and mres:
           ens = etaCvalence(ensemble)
           try:
               sig_figs = len(str(self.m_q))-2
               mres_key = next((key for key, val in ens.mass_comb.items()
                                if round(val, sig_figs) == self.m_q), None)
               self.mres = ens.data[mres_key]['mres']
           except KeyError:
               print(f'no mres info for am_q={self.m_q}')
               self.mres = 0

        self.m_pole += self.mres

    def pole_mass(self):
        p_hat = 2*np.sin(self.tot_mom/2)
        p_hat_sq = np.linalg.norm(p_hat)**2
        p_dash = np.sin(self.tot_mom)
        p_dash_sq = np.linalg.norm(p_dash)**2

        M5 = float(self.prop_in.info['M5'])
        Ls = float(self.prop_in.info['Ls'])

        W = 1 - M5 + p_hat_sq/2
        alpha = np.arccosh((1+W**2+p_dash_sq)/(2*np.abs(W)))
        Z = np.abs(W)*np.exp(alpha)
        delta = (W/Z)**Ls
        R =  1-(W**2)/Z + (delta**2)*(Z-(W**2)/Z)/(1-delta**2)

        E = np.arcsinh(self.m_q*R + delta*(Z-(W**2)/Z)/(1-delta**2))
        return E

    def gamma_Z(self, operators, **kwargs):
        projected = {c: np.trace(np.sum([bl_gamma_proj[c][i]@operators[c][i]
                                         for i in range(len(operators[c]))],
                                        axis=0))
                     for c in bilinear.currents}

        gamma_Z = {c: (bl_gamma_F[c]/projected[c]
                       ).real for c in bilinear.currents}
        return projected, gamma_Z

    def qslash_Z(self, operators, q_vec, Z_q, S_inv,
                 renorm='mSMOM', printval=False, **kwargs):
        q_vec = np.sin(q_vec)
        qslash = np.sum([q_vec[i]*Gamma[dirs[i]]
                        for i in range(len(dirs))], axis=0)
        q_sq = np.linalg.norm(q_vec)**2
        Z_P = Z_q*self.gamma_Z(operators)[1]['P']
        Z_T = Z_q*self.gamma_Z(operators)[1]['T']
        Z_V = Z_q/(np.trace(np.sum([q_vec[i]*operators['V'][i]
                                    for i in range(len(dirs))],
                                   axis=0)@qslash).real/(12*q_sq))
        m_q = self.m_pole
        A1 = np.trace(np.sum([q_vec[i]*operators['A'][i]
                              for i in range(len(dirs))],
                             axis=0)@Gamma['5'])
        A2 = np.trace(np.sum([q_vec[i]*operators['A'][i]
                              for i in range(len(dirs))],
                             axis=0)@Gamma['5']@qslash)
        P = np.trace(operators['P'][0]@Gamma['5']@qslash)
        S = np.trace(S_inv)

        if renorm == 'SMOM':
            Z_A = 12*q_sq*Z_q/A2
        else:
            Z_A = (144*q_sq*(Z_q**2)-2*Z_P*S*P)/(12*Z_q*A2 + 1j*Z_P*A1*P)
        Z_m = (S+(Z_A*A1*1j)/2)/(12*m_q*Z_q)
        Z_mm_q = (S+(Z_A*A1*1j)/2)/(12*Z_q)

        s_term = np.trace(operators['S'][0])
        mass_term = 4*m_q*Z_m*Z_P*P
        if renorm == 'SMOM':
            Z_S = 12*Z_q/s_term
        else:
            Z_S = (12*q_sq*Z_q-mass_term)/(q_sq*s_term)
        qslash_Z = {'S': Z_S.real, 'P': Z_P.real, 'V': Z_V.real,
                    'A': Z_A.real, 'T': Z_T.real, 'm': Z_m.real,
                    'mam_q': Z_mm_q.real}
        return qslash_Z

    def construct_operators(self, S_in, S_out, Gs, **kwargs):
        amputated = np.array([S_out@Gs[b]@S_in for b in range(self.N_bl)])
        operators = {'S': [amputated[self.gammas.index(['I'])]],
                     'P': [amputated[self.gammas.index(['5'])]],
                     'V': [amputated[self.gammas.index([i])]
                           for i in dirs],
                     'A': [amputated[self.gammas.index([i, '5'])]
                           for i in dirs],
                     'T': sum([[amputated[self.gammas.index([dirs[i],
                                                             dirs[j]])]
                                for j in range(i+1, 4)]
                               for i in range(0, 4-1)], [])}
        return operators

    def NPR(self, massive=False, **kwargs):
        # ==central===
        S_in = self.prop_in.inv_propagator.val
        S_out = self.prop_out.inv_outgoing_propagator.val
        operators = self.construct_operators(S_in, S_out, self.avg_bilinear)
        if not massive:
            self.avg_projected, self.Z = self.gamma_Z(operators)
        else:
            Z_q = self.prop_in.Z_q_qslash.val
            S_inv = self.prop_in.inv_propagator.val
            self.Z = self.qslash_Z(operators, self.tot_mom, Z_q, S_inv,
                                   printval=False, **kwargs)
        # ==bootstrap===
        self.Z_btsp = {c: np.zeros(N_boot) for c in self.Z.keys()}
        self.btsp_projected = {c: np.zeros(N_boot, dtype=object)
                               for c in self.Z.keys()}
        for k in range(N_boot):
            S_in = self.prop_in.inv_propagator.btsp[k,]
            S_out = self.prop_out.inv_outgoing_propagator.btsp[k,]
            operators = self.construct_operators(S_in, S_out,
                                                 self.btsp_bilinear[:, k, :])
            if not massive:
                proj_k, Z_k = self.gamma_Z(operators)
                for c in Z_k.keys():
                    self.btsp_projected[c][k] = proj_k[c]
                    self.Z_btsp[c][k] = Z_k[c]
            else:
                Z_q = self.prop_in.Z_q_qslash.btsp[k]
                S_inv = self.prop_in.inv_propagator.btsp[k,]
                Z_k = self.qslash_Z(operators, self.tot_mom,
                                    Z_q, S_inv, **kwargs)
                for c in Z_k.keys():
                    self.Z_btsp[c][k] = Z_k[c].real

        # ==errors===
        self.Z_err = {}
        for c in self.Z.keys():
            self.Z_err[c] = st_dev(self.Z_btsp[c], self.Z[c])
