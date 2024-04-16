from basics import *

class valence:
    eta_h_gamma = ('Gamma5', 'Gamma5')

    def __init__(self, ens, action=(0,0), **kwargs):
        self.ens = ens
        self.action = action
        self.info = params[self.ens]
        self.ainv = stat(
            val=self.info['ainv'],
            err=self.info['ainv_err'],
            btsp='fill')

        self.T = int(self.info['TT'])
        self.all_masses = ['{:.4f}'.format(m) for m in self.info['masses']]
        self.N_mass = len(self.all_masses)
        self.mass_names = {self.all_masses[0]: 'l',
                           self.all_masses[1]: 'l_double',
                           self.all_masses[2]: 's_half',
                           self.all_masses[3]: 's'}
        self.mass_names.update(
            {self.all_masses[m]: f'c{m-4}'
             for m in range(4, self.N_mass)})

        #self.compute_amres(load=False)
        #self.compute_eta_h(load=False)
        #self.masses = {}
        #for m_idx, mass in enumerate(self.all_masses):
        #    original_mass = self.info['masses'][m_idx]
        #    self.masses[mass] = eval(original_mass)+self.amres[mass]

    def amres_correlator(self, mass, load=True, 
                         cfgs=None, meson_num=1,
                         N_src=16, save=True,
                         num_end_points=5,
                         **kwargs):

        a1, a2 = self.action
        fname = f'NPR/action{a1}_action{a2}/'
        fname += '__'.join(['NPR', self.ens,
                    self.info['baseactions'][a1],
                    self.info['baseactions'][a2]])
        fname += '_mSMOM.h5'
        grp_name = f'{str((mass, mass))}/PJ5q'

        if load:
            datapath = path+self.ens
            datapath += 'S/results' if self.ens[-1]!='M' else '/results'
            R = 'R08' if self.all_masses.index(mass)<4 else 'R16'
            massname = self.mass_names[mass]

            self.cfgs = self.info['valence_cfgs'] if cfgs==None else cfgs
            t_src_range = range(0,self.T,int(self.T/N_src))
            corr = np.zeros(shape=(len(self.cfgs), N_src, self.T))

            for cf_idx, cf in enumerate(self.cfgs):
                filename = f'{datapath}/{cf}/conserved/'
                for t_idx, t_src in enumerate(t_src_range):
                    f_string = f'prop_{massname}_{R}_Z2_t'+'{:02d}'.format(t_src)
                    f_string += f'_p+0_+0_+0.{cf}.h5'
                    data = h5py.File(filename+f_string, 'r')['wardIdentity']
                    corr[cf_idx,t_idx,:] = np.roll(
                            np.array(data['PJ5q'][:]['re']),-t_src)

            corr = np.mean(corr, axis=1)
            corr = stat(
                val=np.mean(corr,axis=0),
                err='fill',
                btsp=bootstrap(corr)
                )

            meson, meson_fit = self.meson_correlator(mass, meson_num=meson_num,
                                          load=False)
            corr = corr/meson
            folded_corr = (corr[1:]+corr[::-1][:-1])[:int(self.T/2)]*0.5

            fit_points = np.arange(int(self.T/2))[-num_end_points:]
            def constant_mass(t, param, **kwargs):
                return param[0]*np.ones(len(t))
            x = stat(val=fit_points, btsp='fill')
            y = folded_corr[fit_points]
            res = fit_func(x, y, constant_mass, [0.1, 0])
            fit = res[0] if res[0].val!=0 else folded_corr[-1]
            print(f'For am_q={mass}, am_res={err_disp(fit.val,fit.err)}')

            if save:
                f = h5py.File(fname, 'a')
                if grp_name in f:
                    del f[grp_name]
                mes_grp = f.create_group(grp_name)
                mes_grp.create_dataset('central', data=corr.val) 
                mes_grp.create_dataset('errors', data=corr.err) 
                mes_grp.create_dataset('bootstrap', data=corr.btsp) 

                mes_grp.create_dataset('fit/central', data=fit.val) 
                mes_grp.create_dataset('fit/errors', data=fit.err) 
                mes_grp.create_dataset('fit/bootstrap', data=fit.btsp) 
                f.close()
                print(f'Saved amres data and fit to {grp_name} in {fname}')
        else:
            f = h5py.File(fname, 'r')[grp_name]
            corr = stat(
                    val=np.array(f['central'][:]),
                    err=np.array(f['errors'][:]),
                    btsp=np.array(f['bootstrap'][:]))
            fit = stat(
                    val=np.array(f['fit/central']),
                    err=np.array(f['fit/errors']),
                    btsp=np.array(f['fit/bootstrap'][:]))
        return corr, fit

    def compute_amres(self, masses=None, plot=False, **kwargs):
        self.amres = {}
        if masses==None:
            masses = self.all_masses

        for mass in masses:
            corr, fit = self.amres_correlator(mass, **kwargs)
            folded_corr = (corr[1:]+corr[::-1][:-1])[:int(self.T/2)]*0.5

            self.amres[mass] = fit
            if plot:
                fig, ax = plt.subplots()
                ax.errorbar(np.arange(2,int(self.T/2)), folded_corr.val[2:],
                            yerr=folded_corr.err[2:], fmt='o', capsize=4,
                            label=mass)
                ax.axhspan(self.amres[mass].val+self.amres[mass].err,
                           self.amres[mass].val-self.amres[mass].err,
                           color='k', alpha=0.1)
                ax.set_xlabel(r'$at$')
                ax.set_ylabel(r'PJ5q/PP')
                ax.legend()

        if plot:
            call_PDF(f'{self.ens}_amres.pdf', open=True)

    def meson_correlator(self, mass, load=True, 
                         cfgs=None, meson_num=1,
                         N_src=16, save=True, 
                         fit_start=15, fit_end=30, 
                         **kwargs):

        a1, a2 = self.action
        fname = f'NPR/action{a1}_action{a2}/'
        fname += '__'.join(['NPR', self.ens,
                    self.info['baseactions'][a1],
                    self.info['baseactions'][a2]])
        fname += '_mSMOM.h5'
        grp_name = f'{str((mass, mass))}/meson_{meson_num}/corr'

        if load:
            datapath = path+self.ens
            datapath += 'S/results' if self.ens[-1]!='M' else '/results'
            R = 'R08' if self.all_masses.index(mass)<4 else 'R16'
            massname = self.mass_names[mass]
            foldername = f'{massname}_{R}__{massname}_{R}'
            mes = f'meson_{meson_num}'

            self.cfgs = self.info['valence_cfgs'] if cfgs==None else cfgs
            t_src_range = range(0,self.T,int(self.T/N_src))
            corr = np.zeros(shape=(len(self.cfgs), N_src, self.T))

            for cf_idx, cf in enumerate(self.cfgs):
                filename = f'{datapath}/{cf}/mesons/{foldername}/meson_'
                for t_idx, t_src in enumerate(t_src_range):
                    f_string = f'prop_{massname}_{R}_Z2_t'+'{:02d}'.format(t_src)
                    f_string += '_p+0_+0_+0'
                    fstring = '__'.join([f_string, f_string, f'snk_0_0_0.{cf}.h5'])
                    data = h5py.File(filename+fstring, 'r')['meson'][mes]
                    corr[cf_idx,t_idx,:] = np.roll(
                            np.array(data['corr'][:]['re']),-t_src)

            corr = np.mean(corr, axis=1)
            corr = stat(
                val=np.mean(corr,axis=0),
                err='fill',
                btsp=bootstrap(corr)
                )

            folded_corr = (corr[1:]+corr[::-1][:-1])[:int(self.T/2)]*0.5
            div = ((folded_corr[2:]+folded_corr[:-2])/folded_corr[1:-1])*0.5
            m_eff = div.use_func(np.arccosh)

            def constant_mass(t, param, **kwargs):
                return param[0]*np.ones(len(t))
            x = stat(val=np.arange(fit_start, fit_end), btsp='fill')
            y = m_eff[fit_start:fit_end]
            res = fit_func(x, y, constant_mass, [0.1, 0])
            fit = res[0]
            print(f'For am_q={mass}, am_eta_h={err_disp(fit.val, fit.err)}')

            if save:
                f = h5py.File(fname, 'a')
                if grp_name in f:
                    del f[grp_name]
                mes_grp = f.create_group(grp_name)
                mes_grp.create_dataset('central', data=corr.val) 
                mes_grp.create_dataset('errors', data=corr.err) 
                mes_grp.create_dataset('bootstrap', data=corr.btsp) 

                mes_grp.create_dataset('fit/central', data=fit.val) 
                mes_grp.create_dataset('fit/errors', data=fit.err) 
                mes_grp.create_dataset('fit/bootstrap', data=fit.btsp) 
                
                f.close()
                print(f'Saved meson_{meson_num} corr data '+\
                        f'and fit to {grp_name} in {fname}')
        else:
            f = h5py.File(fname, 'r')[grp_name]
            corr = stat(
                    val=np.array(f['central'][:]),
                    err=np.array(f['errors'][:]),
                    btsp=np.array(f['bootstrap'][:]))
            fit = stat(
                    val=np.array(f['fit/central']),
                    err=np.array(f['fit/errors']),
                    btsp=np.array(f['fit/bootstrap'][:]))
        return corr, fit

    def compute_eta_h(self, plot=False, **kwargs):
        self.eta_h_masses = {}
        for mass in self.all_masses:
            corr, fit = self.meson_correlator(mass, meson_num=1, **kwargs)
            folded_corr = (corr[1:]+corr[::-1][:-1])[:int(self.T/2)]*0.5
            div = ((folded_corr[2:]+folded_corr[:-2])/folded_corr[1:-1])*0.5
            m_eff = div.use_func(np.arccosh)

            self.eta_h_masses[mass] = fit

            if plot:
                fig, ax = plt.subplots(figsize=(8,2))
                ax.errorbar(np.arange(1,int(self.T/2)-1), m_eff.val,
                            yerr=m_eff.err, fmt='o', capsize=4,
                            label=mass)
                ax.fill_between(x.val,
                                (fit.val+fit.err)*np.ones(len(x.val)),
                                (fit.val-fit.err)*np.ones(len(x.val)),
                                color='k', alpha=0.1)
                ax.legend()
                ax.set_xlabel(r'$at$')
                ax.set_ylabel(r'$m_\mathrm{eff}$')

        if 'amres' in self.__dict__.keys():
            fig, ax = plt.subplots(figsize=(5,6))
            x = join_stats([self.amres[mass]+self.info['masses'][
                self.all_masses.index(mass)]
                    for mass in self.all_masses])
            y = join_stats([self.ainv*self.eta_h_masses[mass]
                for mass in self.all_masses])
            ax.errorbar(x.val, y.val, yerr=y.err, xerr=x.err,
                    fmt='o', capsize=4)
            ax.set_xlabel(r'$am_q+am_\mathrm{res}$')
            ax.set_ylabel(r'$M_{\eta_h}$')

        if plot:
            call_PDF(f'{self.ens}_eta_h.pdf', open=True)

    def calc_all(self, **kwargs):
        self.compute_eta_h(**kwargs)
        self.compute_amres(**kwargs)

