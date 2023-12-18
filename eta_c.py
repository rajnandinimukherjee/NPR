from basics import *

# === measured eta_c masses in lattice units======
eta_c_data = {'C1': {'central': {0.30: 1.24641,
                                 0.35: 1.37227,
                                 0.40: 1.49059},
                     'errors': {0.30: 0.00020,
                                0.35: 0.00019,
                                0.40: 0.00017}},
              'M1': {'central': {0.22: 0.96975,
                                 0.28: 1.13226,
                                 0.34: 1.28347,
                                 0.40: 1.42374},
                     'errors': {0.22: 0.00018,
                                0.28: 0.00015,
                                0.34: 0.00016,
                                0.40: 0.00015}},
              'F1S': {'central': {0.18: 0.82322,
                                  0.23: 0.965045,
                                  0.28: 1.098129,
                                  0.33: 1.223360,
                                  0.40: 1.385711},
                      'errors': {0.18: 0.00010,
                                 0.23: 0.000093,
                                 0.28: 0.000086,
                                 0.33: 0.000080,
                                 0.40: 0.000074}}
              }

eta_PDG = stat(
    val=2983.9/1000,
    err=0.5/1000,
    btsp='fill')

eta_stars = [1.8, 2.2, float(eta_PDG.val)]

m_C_PDG = 1.27
m_C_PDG_err = 0.02


def interpolate_eta_c(ens, find_y, **kwargs):
    x = np.array(list(eta_c_data[ens]['central'].keys()))[:-1]
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']
    f_central = interp1d(y*ainv, x, fill_value='extrapolate')
    pred_x = f_central(find_y)

    btsp = np.array([np.random.normal(y[i], yerr[i], 100)
                    for i in range(len(y))])
    pred_x_k = np.zeros(100)
    for k in range(100):
        y_k = btsp[:, k]
        f_k = interp1d(y_k*ainv, x, fill_value='extrapolate')
        pred_x_k[k] = f_k(find_y)
    pred_x_err = ((pred_x_k[:]-pred_x).dot(pred_x_k[:]-pred_x)/100)**0.5
    return [pred_x.item(), pred_x_err]


valence_ens = ['C1', 'M1', 'F1S']


class etaCvalence:
    vpath = 'valence/'
    eta_C_filename = 'eta_C.h5'
    eta_C_gamma = ('Gamma5', 'Gamma5')

    def __init__(self, ens, create=False, **kwargs):
        self.ens = ens
        self.NPR_masses = params[self.ens]['masses']
        self.N_mass = len(self.NPR_masses)
        self.mass_comb = {('l', 'l'): self.NPR_masses[0]}
        self.mass_comb.update({(f'c{m-1}', f'c{m-1}'): self.NPR_masses[m]
                               for m in range(1, self.N_mass)})
        self.data = {key: {} for key in self.mass_comb.keys()}
        self.eta_C_file = h5py.File(self.eta_C_filename, 'a')

        if ens not in self.eta_C_file or create:
            self.createH5()
        else:
            self.N_cf, self.T = self.eta_C_file[self.ens][str(
                list(self.mass_comb.keys())[0])]['corr'].shape
            self.data = {k: {obj: {
                'data': np.zeros(shape=(self.N_cf, self.T),
                                 dtype=float)}
                             for obj in ['corr', 'PJ5q']}
                         for k in self.mass_comb.keys()}
            for key in self.data.keys():
                for obj in ['corr', 'PJ5q']:
                    self.data[key][obj]['data'][:, :] = np.array(
                        self.eta_C_file[self.ens][str(key)][obj])

        self.T_half = int(self.T/2)
        for key in self.data.keys():
            for obj in ['corr', 'PJ5q']:
                data = self.data[key][obj]['data']
                avg_data = np.mean(data, axis=0)
                err = np.array([st_dev(data[:, t], mean=avg_data[t])
                                for t in range(self.T)])
                btsp_expanded = bootstrap(data, K=N_boot).real

                folded_data = 0.5 * \
                    (data[:, 1:] + data[:, ::-1][:, :-1])[:, :self.T_half]
                folded_avg_data = np.mean(folded_data, axis=0)
                if data.shape[0] == 1:
                    folded_err = np.array([folded_avg_data[t]*0.0001
                                          for t in range(self.T_half)])
                    btsp_data = np.array([np.random.normal(
                        folded_avg_data[t], folded_err[t], N_boot)
                        for t in range(self.T_half)]).T
                else:
                    folded_err = np.array([st_dev(folded_data[:, t])
                                           for t in range(self.T_half)])
                    btsp_data = bootstrap(folded_data, K=N_boot).real

                cov = COV(btsp_data, center=folded_avg_data)

                self.data[key][obj].update({'avg': avg_data,
                                            'err': err,
                                            'folded': folded_data,
                                            'folded_avg': folded_avg_data,
                                            'folded_err': folded_err,
                                            'btsp': btsp_data,
                                            'btsp_exp': btsp_expanded,
                                            'COV': cov})
                if obj == 'corr':
                    meff = m_eff(folded_avg_data, ansatz='cosh')
                    self.data[key]['mccbar'] = meff[-2]
                    m_btsp = np.array([m_eff(btsp_data[b, :])[-2]
                                       for b in range(N_boot)])
                    self.data[key]['mccbar_err'] = st_dev(m_btsp, meff[-2])

            ratio = self.data[key]['PJ5q']['avg'] /\
                self.data[key]['corr']['avg']
            btsp_ratio = np.array([self.data[key]['PJ5q']['btsp_exp'][k, :] /
                                   self.data[key]['corr']['btsp_exp'][k, :]
                                   for k in range(N_boot)])
            cov_ratio = COV(btsp_ratio, center=ratio)
            err_ratio = np.diag(cov_ratio)**0.5
            self.data[key]['ratio'] = {'avg': ratio,
                                       'err': err_ratio,
                                       'btsp': btsp_ratio,
                                       'COV': cov_ratio}
            self.data[key]['mres'] = ratio[self.T_half]

    def toDict(self, mres=True, keys='all'):
        ens_dict = {'central': {}, 'errors': {}}
        keylist = keys if keys != 'all' else self.mass_comb.keys()

        for key in keylist:
            q_mass = self.mass_comb[key]+mres*self.data[key]['mres']
            ens_dict['central'][q_mass] = self.data[key]['mccbar']
            ens_dict['errors'][q_mass] = self.data[key]['mccbar_err']

        eta_c_data[self.ens] = ens_dict
        print(f'modified eta_c_data dictionary for {self.ens} ensemble')

    def plot(self, keys='all', plotfits=False, **kwargs):
        keylist = keys if keys != 'all' else self.mass_comb.keys()

        fig = plt.figure()
        x = np.arange(self.T_half)
        for k in keylist:
            mass = self.mass_comb[k]
            mass_idx = self.NPR_masses.index(mass)
            y = m_eff(self.data[k]['corr']['folded_avg'], ansatz='cosh')
            btsp = np.array([m_eff(self.data[k]['corr']['btsp'][b, :],
                             ansatz='cosh') for b in range(N_boot)])
            e = np.array([st_dev(btsp[:, t], mean=y[t])
                         for t in range(len(y))])
            plt.errorbar(x[:-2], y, yerr=e, fmt='o',
                         capsize=4, label=str(mass),
                         c=color_list[mass_idx])

            if plotfits:
                (s, e, t) = (15, 27, 2)
                p, perr, pbtsp = self.objfit(k, (s, e, t), obj='corr')
                x_fit = np.arange(s, e+1, t)
                y_fit = p[1]*np.ones(len(x_fit))
                y_plus = y_fit+perr[1]
                y_minus = y_fit-perr[1]
                plt.plot(x_fit, y_fit, c=color_list[mass_idx])
                plt.fill_between(x_fit, y_plus, y_minus,
                                 color=color_list[mass_idx],
                                 alpha=0.5)

        plt.title(f'{self.ens} meff')
        plt.xlabel(r'$t$')
        plt.legend()

        fig = plt.figure()
        x = np.arange(self.T)
        for k in keylist:
            mass = self.mass_comb[k]
            mass_idx = self.NPR_masses.index(mass)
            y = self.data[k]['ratio']['avg']
            e = self.data[k]['ratio']['err']
            plt.errorbar(x, y, yerr=e, fmt='o',
                         capsize=4, label=str(mass),
                         c=color_list[mass_idx])

            if plotfits:
                (s, e, t) = (25, 40, 2)
                p, perr, pbtsp = self.objfit(k, (s, e, t), obj='ratio')
                x_fit = np.arange(s, e+1, t)
                y_fit = self.ansatz(p, x_fit, obj='ratio')
                y_plus = self.ansatz(p+perr, x_fit, obj='ratio')
                y_minus = self.ansatz(p-perr, x_fit, obj='ratio')
                plt.plot(x_fit, y_fit, c=color_list[mass_idx])
                plt.fill_between(x_fit, y_plus, y_minus,
                                 color=color_list[mass_idx],
                                 alpha=0.5)

        plt.title(f'{self.ens} mres')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$am_{res} = \langle 0|J_{5q}|\pi\rangle/' +
                   r'\langle 0|J_{5}|\pi\rangle$')
        plt.legend()

        filename = f'plots/{self.ens}_masses.pdf'
        pdf = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pdf, format='pdf')
        pdf.close()
        plt.close('all')
        os.system('open '+filename)

    def ansatz(self, params, t, obj='corr', **kwargs):
        if obj == 'corr':
            return params[0]*np.exp(-params[1]*t)
        elif obj == 'ratio':
            return params[0]*0+params[1]*np.ones(len(t))

    def objfit(self, key, t_info, obj='corr', **kwargs):
        (start, end, thin) = t_info
        t = np.arange(start, end+1, thin)

        def diff(params, fit='central', k=0, **kwargs):
            if fit == 'central':
                y = self.data[key]['corr']['folded_avg'
                                           if obj == 'corr' else 'avg'][t]
            else:
                y = self.data[key][obj]['btsp'][k, t]
            return y - self.ansatz(params, t, obj=obj)

        cov = self.data[key][obj]['COV'][start:end+1:thin,
                                         start:end+1:thin]
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)

        def LD(params, fit='central', k=0):
            return L.dot(diff(params, fit=fit, k=k))

        guess = [1e+2, 0.1] if obj == 'corr' else [0, 0.003]
        res = least_squares(LD, guess, args=('central', 0),
                            ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x)).real
        dof = len(t)-np.count_nonzero(np.array(guess))
        pvalue = gammaincc(dof/2, chi_sq/2)

        params_central = np.array(res.x).real
        print(key, params_central)
        pdb.set_trace()

        params_btsp = np.zeros(shape=(N_boot, len(guess)))
        for k in range(N_boot):
            res_k = least_squares(LD, guess, args=('btsp', k))
            params_btsp[k, :] = np.array(res_k.x).real

        params_err = np.array([st_dev(params_btsp[:, i],
                               mean=params_central[i])
                               for i in range(len(params_central))])
        return params_central, params_err, params_btsp

    def createH5(self, **kwargs):
        self.datapath = path+self.vpath+self.ens
        self.cf_list = sorted(next(os.walk(self.datapath))[1])
        for cf in self.cf_list.copy():
            try:
                for keys in list(self.mass_comb.keys()):
                    key0, key1 = keys
                    filepath = self.datapath+'/'+str(cf)+'/mesons/'
                    filepath = glob.glob(filepath+f'{key0}_R*{key1}_R*')[0]+'/'

                    filepath = self.datapath+'/'+str(cf)+'/conserved/'
                    filepath = glob.glob(filepath+f'*{key0}_R*t*')[0]+'/'
            except IndexError:
                self.cf_list.remove(cf)
                print(f'removed {cf}')

        self.N_cf = len(self.cf_list)

        test_path = self.datapath+f'/{self.cf_list[0]}/mesons/'
        testfile_path = glob.glob(test_path+'c0*c0*')[0]+'/'
        testfile_path = glob.glob(testfile_path+'*c0*t00*')[0]
        self.testfile = h5py.File(testfile_path, 'r')['meson']
        self.T = len(np.array(self.testfile['meson_0']['corr'])['re'])

        self.gammas = []
        for mes_idx in range(16**2):
            gamma_src = self.testfile[f'meson_{mes_idx}'].attrs['gamma_src'][0].decode(
            )
            gamma_snk = self.testfile[f'meson_{mes_idx}'].attrs['gamma_snk'][0].decode(
            )
            self.gammas.append((gamma_src, gamma_snk))
        self.eta_C_idx = self.gammas.index(self.eta_C_gamma)

        self.data = {k: {obj: {'data': np.zeros(shape=(self.N_cf, self.T), dtype=float)}
                         for obj in ['corr', 'PJ5q']}
                     for k in self.mass_comb.keys()}

        for key in self.data.keys():
            for c in range(self.N_cf):
                for obj in ['corr', 'PJ5q']:
                    suffix = 'mesons/' if obj == 'corr' else 'conserved/'
                    filepath = self.datapath+f'/{self.cf_list[c]}/'+suffix
                    if obj == 'corr':
                        print(key, obj, filepath)
                        filepath = glob.glob(
                            filepath+f'{key[0]}_R*{key[1]}_R*')[0]+'/'
                    filenames = glob.glob(filepath+f'*{key[0]}_R*t*')
                    T_src = len(filenames)
                    config_data = np.zeros(shape=(T_src, self.T))
                    for t in range(T_src):
                        filename = filenames[t]
                        datafile = h5py.File(filename, 'r')['meson'][f'meson_{self.eta_C_idx}']\
                            if obj == 'corr' else h5py.File(filename, 'r')['wardIdentity']
                        data = np.array(datafile[obj])['re']
                        t_val = int(filename.rsplit(
                            '/')[-1].rsplit('_t')[1].rsplit('_')[0])
                        config_data[t, :] = np.roll(data, -t_val)
                    self.data[key][obj]['data'][c, :] = np.mean(
                        config_data, axis=0)

        if self.ens in self.eta_C_file:
            del self.eta_C_file[self.ens]
        ens_group = self.eta_C_file.create_group(self.ens)
        for key in self.data.keys():
            key_group = ens_group.create_group(str(key))
            for obj in ['corr', 'PJ5q']:
                key_group.create_dataset(obj, data=self.data[key][obj]['data'])
            key_group.attrs['mass'] = self.mass_comb[key]

        print(f'Added correlator data to {self.ens} group in eta_C.h5 file')

    def merge_mixed(self, **kwargs):
        mass_combinations = self.avg_results[(0, 1)].keys()

        for masses in list(mass_combinations):
            momenta = self.momenta[(0, 1)][masses]
            res1 = self.avg_results[(0, 1)][masses]
            res2 = self.avg_results[(1, 0)][masses]
            self.avg_results[(0, 1)][masses] = (res1+res2)/2.0

            err1 = self.avg_errs[(0, 1)][masses]
            err2 = self.avg_errs[(1, 0)][masses]
            self.avg_errs[(0, 1)][masses] = err1+err2

        self.avg_results.pop((1, 0))
        self.avg_errs.pop((1, 0))
