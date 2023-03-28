from fourquark import *

class bilinear_analysis:
    keys = ['S','P', 'V', 'A', 'T', 'm']
    N_boot = 200
    def __init__(self, ensemble, loadpath=None, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1,len(info['masses']))]
        
        gauge_count = len(info['baseactions'])
        self.actions = [info['gauges'][i]+'_'+info['baseactions'][i] for
                        i in range(gauge_count)]

        if loadpath==None:
            self.momenta, self.avg_results, self.avg_errs = {}, {}, {}
            for data in [self.momenta, self.avg_results, self.avg_errs]:
                for a1, a2 in itertools.product([0,1],[0,1]):
                    data[(a1,a2)] = {}

        else:
            print('Loading NPR bilinear data from '+loadpath)
            self.momenta, self.avg_results, self.avg_errs = pickle.load(
                                                            open(loadpath, 'rb')) 
            self.all_masses = list(self.momenta[(0,0)].keys()) 

    
    def NPR(self, masses, action=(0,0), scheme=1, massive=False, **kwargs): 
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]
        
        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.bl_list = common_cf_files(self.data, 'bilinears', prefix='bi_')

            results = {}
            errs = {}

            for b in tqdm(range(len(self.bl_list)), leave=False):
                prop1_name, prop2_name = self.bl_list[b].split('__')
                prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am']==m1 and prop2_info['am']==m2)
                condition2 = (prop1_info['prop']==a1 and prop2_info['prop']==a2) 

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name)
                    prop2 = external(self.ens, filename=prop2_name)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared==prop2.momentum_squared
                    condition4 = prop1.momentum_squared==scheme*np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        bl = bilinear(self.ens, prop1, prop2)
                        bl.NPR(massive=massive)
                        if bl.q not in results.keys():
                            results[bl.q] = bl.Z
                            errs[bl.q] = bl.Z_err

            self.momenta[action][(m1,m2)] = sorted(results.keys())
            self.avg_results[action][(m1,m2)] = np.array([results[mom]
                            for mom in self.momenta[action][(m1,m2)]])
            self.avg_errs[action][(m1,m2)] = np.array([errs[mom]
                            for mom in self.momenta[action][(m1,m2)]])
    
    def save_NPR(self, addl_txt='', **kwargs):
        filename = 'pickles/'+self.ens+'_bl.p'
        pickle.dump([self.momenta, self.avg_results, self.avg_errs],
                    open(filename,'wb'))
        print('Saved bilinear NPR results to '+filename)

    def NPR_all(self, massive=False, save=True, **kwargs):
        for a1,a2 in itertools.product([0,1],[0,1]):
            self.NPR((self.sea_mass, self.sea_mass), action=(a1,a2))
        if massive:
            for mass in self.non_sea_masses:
                self.NPR((self.sea_mass, mass),massive=True)
                self.NPR((mass, self.sea_mass),massive=True)
                self.NPR((mass, mass),massive=True)

        if save:
            self.save_NPR()

    def extrap_Z(self, mu, masses, action=(0,0), **kwargs):
        Z = {}
        Z_err = {}
        momentas = self.momenta[action][masses]
        for c in self.avg_results[action][masses][0].keys():
            Zs = [self.avg_results[action][masses][i][c]
                 for i in range(len(momentas))]
            f = interp1d(momentas, Zs, fill_value='extrapolate')
            Z[c] = f(mu)

            Z_es = [self.avg_errs[action][masses][i][c]
                     for i in range(len(momentas))]
            np.random.seed(1)
            Z_btsp = np.random.multivariate_normal(Zs,
                     np.diag(Z_es)**2, self.N_boot)
            store = []
            for k in range(self.N_boot):
                f = interp1d(momenta, Z_btsp[k,:],
                         fill_value='extrapolate')
                store.append(f(point))
            Z_err[c] = st_dev(np.array(store),mean=Z[c])

        return Z, Z_err

    def plot_masswise(self, action=(0,0), save=False, **kwargs):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        if 'mass_combination' in kwargs.keys():
            if kwargs['mass_combination']=='nondeg':
                plot_masses = []
                for m in [self.sea_mass]+self.non_sea_masses:
                    plot_masses.append((m, self.sea_mass))
            elif kwargs['mass_combination']=='deg':
                plot_masses = []
                for m in [self.sea_mass]+self.non_sea_masses:
                    plot_masses.append((m, m))
            else:
                plot_masses = kwargs['mass_combination']
        else:
            plot_masses = self.momenta[action].keys()
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for m in plot_masses:
                label = '('+m[0]+','+m[1]+')'
                mom = self.momenta[action][m]
                res = self.avg_results[action][m][c]
                err = self.avg_errs[action][m][c]
                val_col.scatter(mom, res, label=label)
                err_col.scatter(mom, err, label=label)
            val_col.title.set_text(c)
            handles, labels = err_col.get_legend_handles_labels()
        ax[1,2].set_xlabel('$|q|/GeV$')
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(self.ens+' comparison of masses')
        fig.tight_layout()


        filename = 'plots/'+self.ens+'_mass_comp_bl.pdf'
        print('Plot saved to plots/'+self.ens+'_mass_comp_bl.pdf')
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        os.system(filename)

    def plot_actionwise(self, mass, save=False, **kwargs):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        action_combinations = self.momenta.keys()
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for action in action_combinations:
                mom = self.momenta[action][(m1,m2)]
                res = self.avg_results[action][(m1,m2)][c]
                err = self.avg_errs[action][(m1,m2)][c]
                val_col.scatter(mom, res, label=action)
                err_col.scatter(mom, err, label=action)
            val_col.title.set_text(c)
            handles, labels = err_col.get_legend_handles_labels()
        ax[1,2].set_xlabel('$|q|/GeV$')
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(self.ens+' comparison of actions')
        fig.tight_layout()

        if save:
            plt.savefig('plots/'+self.ens+'_action_comp_bl.pdf')
            print('Plot saved to plots/'+self.ens+'_action_comp_bl.pdf')

    def massive_Z_plots(self, mu=2, action=(0,0), **kwargs):
        plt.figure()
        plt.title(self.ens+' $Z_m$({:.3f})'.format(mu))
        x = np.array([float(m) for m in self.non_sea_masses])

        m_nsm_0 = self.non_sea_masses[0]
        if mu in self.momenta[action][(m_nsm_0,m_nsm_0)]:
            mu_idx = self.momenta[action][(m_nsm_0,m_nsm_0)].index(mu)
            y = np.array([self.avg_results[(0,0)][(m,m)][mu_idx]['m']
                          for m in self.non_sea_masses])
            e = np.array([self.avg_errs[(0,0)][(m,m)][mu_idx]['m']
                          for m in self.non_sea_masses])
            
        else:
            y = np.array([self.extrap_Z(mu=mu, masses=(m,m),
                          action=action)[0]['m']
                          for m in self.non_sea_masses])
            e = np.array([self.extrap_Z(mu=mu, masses=(m,m),
                          action=action)[1]['m']
                          for m in self.non_sea_masses])

        plt.errorbar(x,y,yerr=e,fmt='o',capsize=4)
        plt.xlabel(r'$am_q$')

        def Z_m_ansatz(params, am, **kwargs):
            return params[0] + params[1]*am + params[2]*(am**2)

        def diff(params):
            return y - Z_m_ansatz(params, x)


        filename = 'plots/'+self.ens+'_massive_Z.pdf'
        print('Plot saved to '+filename)
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        os.system('open '+filename)
        

class fourquark_analysis:
    N_boot = 200
    def __init__(self, ensemble, loadpath=None, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1,len(info['masses']))]

        self.actions = [info['gauges'][0]+'_'+info['baseactions'][0],
                   info['gauges'][-1]+'_'+info['baseactions'][-1]]

        if loadpath==None:
            self.momenta, self.avg_results, self.avg_errs = {}, {}, {}
            for data in [self.momenta, self.avg_results, self.avg_errs]:
                for a1, a2 in itertools.product([0,1],[0,1]):
                    data[(a1,a2)] = {}

        else:
            print('Loading NPR fourquark data from '+loadpath)
            self.momenta, self.avg_results, self.avg_errs = pickle.load(
                                                            open(loadpath, 'rb')) 
            self.all_masses = list(self.momenta[(0,0)].keys()) 

    
    def NPR(self, masses, action=(0,0), scheme=1, **kwargs): 
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]
        
        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.fq_list = common_cf_files(self.data,
                                           'fourquarks', prefix='fourquarks_')

            results = {}
            errs = {}

            desc = '_'.join([self.ens, m1, m2, str(a1), str(a2)])
            for f in tqdm(range(len(self.fq_list)), leave=False, desc=desc):
                prop1_name, prop2_name = self.fq_list[f].split('__')
                prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am']==m1 and prop2_info['am']==m2)
                condition2 = (prop1_info['prop']==a1 and prop2_info['prop']==a2) 

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name)
                    prop2 = external(self.ens, filename=prop2_name)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared==prop2.momentum_squared
                    condition4 = prop1.momentum_squared==scheme*np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        fq = fourquark(self.ens, prop1, prop2)
                        fq.errs()
                        if fq.q not in results.keys():
                            results[fq.q] = fq.Z_over_Z_A
                            errs[fq.q] = fq.Z_A_errs

                self.momenta[action][(m1,m2)] = sorted(results.keys())
                self.avg_results[action][(m1,m2)] = np.array([results[mom]
                                for mom in self.momenta[action][(m1,m2)]])
                self.avg_errs[action][(m1,m2)] = np.array([errs[mom]
                                for mom in self.momenta[action][(m1,m2)]])
    
    def save_NPR(self, addl_txt='', **kwargs):
        filename = 'pickles/'+self.ens+'_fq'+addl_txt+'.p'
        pickle.dump([self.momenta, self.avg_results, self.avg_errs],
                    open(filename,'wb'))
        print('Saved fourquark NPR results to '+filename)

    def NPR_all(self, massive=False, save=True, **kwargs):
        num_actions = len(self.actions)
        for a1,a2 in itertools.product(range(num_actions),range(num_actions)):
            self.NPR((self.sea_mass, self.sea_mass), action=(a1,a2))
        if massive:
            for mass in self.non_sea_masses:
                self.NPR((self.sea_mass, mass))
                self.NPR((mass, self.sea_mass))
                self.NPR((mass, mass))

        if save:
            self.save_NPR()

    def extrap_Z(self, action, point, masses=None, **kwargs): 
        if masses==None:
            masses = (self.sea_mass, self.sea_mass)

        momentas = self.momenta[action][masses]
        Z_facs = self.avg_results[action][masses]
        Z_errs = self.avg_errs[action][masses]

        mtx, errs = np.zeros(shape=(5,5)), np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5), range(5)):
            Zs = [Z[i,j] for Z in Z_facs]
            f = interp1d(momenta, Zs, fill_value='extrapolate')
            mtx[i,j] = f(point)

            Z_es = [e[i,j] for e in Z_errs]
            np.random.seed(1)
            Z_btsp = np.random.multivariate_normal(Zs,
                     np.diag(Z_es)**2, self.N_boot)
            store = []
            for k in range(self.N_boot):
                f = interp1d(momenta, Z_btsp[k,:],
                         fill_value='extrapolate')
                store.append(f(point))
            errs[i,j] = st_dev(np.array(store),mean=mtx[i,j])

        return mtx, errs
    
    def merge_mixed(self, **kwargs):
        mass_combinations = self.avg_results[(0,1)].keys()

        for masses in list(mass_combinations):
            momenta = self.momenta[(0,1)][masses]
            res1 = self.avg_results[(0,1)][masses]
            res2 = self.avg_resutls[(1,0)][masses]
            self.avg_results[(0,1)][masses] = (res1+res2)/2.0

            err1 = self.avg_errs[(0,1)][masses]
            err2 = self.avg_errs[(1,0)][masses]
            self.avg_errs[(0,1)] = err1+err2

        self.avg_results.pop((1,0))
        self.avg_errs.pop((1,0))

    def sigma(self, action, mu1, mu2, masses=None, **kwargs):
        if masses==None:
            masses = (self.sea_mass, self.sea_mass)

        cond1 = mu1 in self.momenta[action][masses]
        cond2 = mu2 in self.momenta[action][masses]
        if not cond1:
            Z1, Z1_err = self.extrap_Z(action, mu1, masses=masses)
        else:
            mu1_idx = self.momenta[action][masses].index(mu1)
            Z1 = self.avg_results[action][masses][mu1_idx,:,:]
            Z1_err = self.avg_errs[action][masses][mu1_idx,:,:]

        if not cond2:
            Z2, Z2_err = self.extrap_Z(action, mu2, masses=masses)
        else:
            mu2_idx = self.momenta[action][masses].index(mu2)
            Z2 = self.avg_results[action][masses][mu2_idx,:,:]
            Z2_err = self.avg_errs[action][masses][mu2_idx,:,:]

        sig = Z2@np.linalg.inv(Z1)
        
        var_Z1 = np.zeros(shape=(5,5,self.N_boot))
        var_Z2 = np.zeros(shape=(5,5,self.N_boot))
        for i,j in itertools.product(range(5),range(5)):
            var_Z1[i,j,:] = np.random.normal(Z1[i,j],Z1_err[i,j],self.N_boot)  
            var_Z2[i,j,:] = np.random.normal(Z2[i,j],Z2_err[i,j],self.N_boot) 
        sig_btsp = np.array([var_Z2[:,:,b]@np.linalg.inv(var_Z1[:,:,b])
                           for b in range(self.N_boot)])
        sig_err = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5),range(5)):
            sig_err[i,j] = st_dev(sig_btsp[:,i,j], mean=sig[i,j])

        return sigma, sig_err

    def plot_actions(self, masses=None, **kwargs):
        if masses==None:
            masses = (self.sea_mass, self.sea_mass)
        
        action_combinations = list(self.avg_results.keys())

        plt.figure()
        for action in action_combinations:
            x = self.momenta[action][masses]
            y = [mtx[0,0] for mtx in self.avg_results[action][masses]]
            e = [mtx[0,0] for mtx in self.avg_errs[action][masses]]
            plt.errorbar(x,y,yerr=e,fmt='o',capsize=4,label=str(action))
        plt.legend()
        plt.xlabel('$q/GeV$')
        plt.title(f'Block 1 Ensemble {self.ens}')

        fig, ax = plt.subplots(2,2,sharex=True)
        for i,j in itertools.product(range(2), range(2)):
            k, l = i+1, j+1
            for action in action_combinations:
                x = self.momenta[action][masses]
                y = [mtx[k,l] for mtx in self.avg_results[action][masses]]
                e = [mtx[k,l] for mtx in self.avg_errs[action][masses]]
                ax[i,j].errorbar(x,y,yerr=e,fmt='o',capsize=4,label=str(action))
                if i==1:
                    ax[i,j].set_xlabel('$q/GeV$')
        handles, labels = ax[1,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.suptitle(f'Block 2 Ensemble {self.ens}')

        fig, ax = plt.subplots(2,2,sharex=True)
        for i,j in itertools.product(range(2), range(2)):
            k, l = i+3, j+3
            for action in action_combinations:
                x = self.momenta[action][masses]
                y = [mtx[k,l] for mtx in self.avg_results[action][masses]]
                e = [mtx[k,l] for mtx in self.avg_errs[action][masses]]
                ax[i,j].errorbar(x,y,yerr=e,fmt='o',capsize=4,label=str(action))
                if i==1:
                    ax[i,j].set_xlabel('$q/GeV$')
        handles, labels = ax[1,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.suptitle(f'Block 3 Ensemble {self.ens}')

        filename = f'plots/{self.ens}_summary.pdf'
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        os.system(f'open {filename}')


