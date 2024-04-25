from NPR_classes import *
from valence import *
from coeffs import *

eta_PDG = stat(
    val=2983.9/1000,
    err=0.5/1000,
    btsp='fill')

eta_star = eta_PDG/2


class Z_bl_analysis:

    grp = 'bilinear'

    def __init__(self, ens, action=(0, 0), scheme='qslash',
                 renorm='mSMOM', **kwargs):

        self.ens = ens
        self.renorm = renorm

        a1, a2 = action
        datafile = f'NPR/action{a1}_action{a2}/'
        datafile += '__'.join(['NPR', self.ens, params[self.ens]['baseactions'][a1],
                              params[self.ens]['baseactions'][a2]])
        datafile += f'_{renorm}.h5'
        self.data = h5py.File(datafile, 'r')

        info = params[self.ens]
        self.ainv = stat(
            val=info['ainv'],
            err=info['ainv_err'],
            btsp='fill')

        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1, len(info['masses']))]
        self.all_masses = [self.sea_mass]+self.non_sea_masses
        self.valence = valence(self.ens)
        self.valence.calc_all(load=False)

        self.momenta, self.Z = {}, {}

        for m in self.data.keys():
            masses = (m[2:8], m[12:18])
            self.momenta[masses] = stat(
                val=self.data[m][self.grp]['ap'][:],
                btsp='fill')

            self.Z[masses] = {}
            for key in self.data[m][self.grp].keys():
                if key != 'ap':
                    self.Z[masses][key] = stat(
                        val=self.data[m][self.grp][key]['central'][:],
                        err=self.data[m][self.grp][key]['errors'][:],
                        btsp=self.data[m][self.grp][key]['bootstrap'][:]
                    )

    def plot_mass_dependence(self, mu, eta_pdg, eta_star,
                             key='m', start=0,
                             stop=None, open_file=True,
                             filename='', add_pdg=False,
                             pass_vals=False, key_only=False,
                             normalise=False, Z_only=False,
                             **kwargs):

        if filename=='':
            filename = f'{self.renorm}_{self.ens}_vs_amq.pdf'

        x = join_stats([self.valence.amres[m]+eval(m)
                        for m in self.all_masses[start:stop]])
        if key_only:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.set_title(self.ens)
            if key=='m' and self.renorm=='SMOM':
                n_pts = 4
                y = join_stats([self.interpolate(mu, (m,m), 'P')
                                for m in self.all_masses[start:stop]])**(-1)
                ax.set_ylabel(r'$Z_m(\mu='+str(mu)+r'\,\mathrm{GeV}) = 1/Z_P$')
                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err,
                            capsize=4, fmt='o')
                ax.errorbar(x.val[:n_pts], y.val[:n_pts],
                            xerr=x.err[:n_pts], yerr=y.err[:n_pts],
                            capsize=4, fmt='o', c='r')
                
                def chiral_extrap_ansatz(am, param, **kwargs):
                    return param[0] + param[1]*am
                res = fit_func(x,y,chiral_extrap_ansatz,[1,0.1],end=n_pts)
                y0 = res[0]
                ax.errorbar([0], [y0.val], yerr=[y0.err],
                            capsize=4, fmt='o', c='k',
                            label='$Z_m^\mathrm{SMOM}:$'+\
                                    err_disp(y0.val, y0.err))
                ymin, ymax = ax.get_ylim()
                ax.vlines(x=0, ymin=ymin, ymax=ymax,
                          color='k', linestyle='dashed')

                xmin, xmax = ax.get_xlim()
                xrange = np.linspace(xmin, x.val[n_pts-1], 100)
                yrange = res.mapping(xrange)
                ax.fill_between(xrange, yrange.val+yrange.err,
                                yrange.val-yrange.err,
                                color='k', alpha=0.1)
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.legend(frameon=False)
                ax.set_xlabel(r'$am_q+am_\mathrm{res}$')
                
                if pass_vals:
                    plt.close(fig)
                    return y0
                else:
                    call_PDF(filename, open=open_file)
                
            else:
                y = join_stats([self.interpolate(mu, (m,m), key)
                                for m in self.all_masses[start:stop]])
                ax.set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+r'\,\mathrm{GeV})$')
                if normalise:
                    q = join_stats([self.interpolate(mu, (m,m), 'q')
                                    for m in self.all_masses[start:stop]])
                    y = y/q
                    ax.set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+r'\,\mathrm{GeV})/Z_q$')

                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err,
                            capsize=4, fmt='o')
                ax.set_xlabel(r'$am_q+am_\mathrm{res}$')
                call_PDF(filename, open=open_file)

        else:
            if stop==None:
                stop = len(self.all_masses)

            nrows = 2 if Z_only else 3
            figsize = (3,7) if Z_only else (3,10)
            fig, ax = plt.subplots(nrows=nrows, ncols=1, 
                                   sharex='col',
                                   figsize=figsize)
            plt.subplots_adjust(hspace=0, wspace=0)

            y = join_stats([self.valence.eta_h_masses[m]*self.ainv
                            for m in self.all_masses[start:stop]])
            def eta_mass_ansatz(am, param, **kwargs):
                return param[0] + param[1]*am**0.5 + param[2]*am
            ax[0].text(1.05, 0.5,
                       r'$\alpha + \beta\sqrt{am_q} + \gamma\,am_q$',
                       va='center', ha='center', rotation=90,
                       color='k', alpha=0.3,
                       transform=ax[0].transAxes)

            res = fit_func(x, y, eta_mass_ansatz, [0.1,1,1], verbose=False)
            def pred_amq(eta_mass, param, **kwargs):
                a, b, c = param
                root = (-b+(b**2 + 4*c*(eta_mass-a))**0.5)/(2*c)
                return root**2
            am_c = stat(
                    val=pred_amq(eta_pdg.val, res.val),
                    err='fill',
                    btsp=[pred_amq(eta_pdg.btsp[k], res.btsp[k])
                          for k in range(N_boot)]
                    )
            am_star = stat(
                       val=pred_amq(eta_star.val, res.val),
                       err='fill',
                       btsp=[pred_amq(eta_star.btsp[k], res.btsp[k])
                             for k in range(N_boot)]
                       )

            ax[0].errorbar(x.val, y.val, yerr=y.err,
                        capsize=4, fmt='o')
            if add_pdg:
                ymin, ymax = ax[0].get_ylim()
                ax[0].hlines(y=eta_pdg.val, xmin=0, xmax=am_c.val, color='k')
                ax[0].text(0.05, eta_pdg.val-0.05, r'$M_{\eta_c}^\mathrm{PDG}$',
                           va='top', ha='center', color='k')
                ax[0].fill_between(np.linspace(0,am_c.val,100),
                                   eta_pdg.val+eta_pdg.err,
                                   eta_pdg.val-eta_pdg.err,
                                   color='k', alpha=0.2)
                ax[0].vlines(x=am_c.val, ymin=ymin, ymax=eta_pdg.val,
                             color='k', linestyle='dashed')
                ax[0].text(am_c.val, 0.5, r'$am_c$', rotation=90,
                           va='center', ha='right', color='k')
                ax[0].fill_between(np.linspace(am_c.val-am_c.err,
                                               am_c.val+am_c.err,100),
                                   ymin, eta_pdg.val+eta_pdg.err,
                                   color='k', alpha=0.2)

                ax[0].hlines(y=eta_star.val, xmin=0, xmax=am_star.val, color='r')
                ax[0].text(0.05, eta_star.val-0.05, r'$M^\star$',
                           va='top', ha='center', color='r')
                ax[0].fill_between(np.linspace(0,am_star.val,100),
                                   eta_star.val+eta_star.err,
                                   eta_star.val-eta_star.err,
                                   color='r', alpha=0.2)
                ax[0].vlines(x=am_star.val, ymin=ymin, ymax=eta_star.val,
                             color='r', linestyle='dashed')
                ax[0].text(am_star.val, 0.5, r'$am^\star$', rotation=90,
                           va='center', ha='right', color='r')
                ax[0].fill_between(np.linspace(am_star.val-am_star.err,
                                               am_star.val+am_star.err,100),
                                   ymin, eta_star.val+eta_star.err,
                                   color='r', alpha=0.2)

                discard, new_ymax = ax[0].get_ylim()
                ax[0].set_ylim([ymin, new_ymax])

            xmin, xmax = ax[0].get_xlim()
            xrange = np.linspace(x.val[0], xmax, 100)
            yrange = res.mapping(xrange)
            ax[0].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                    str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                    ha='center', va='center',
                    transform=ax[0].transAxes)
            ax[0].fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color='k', alpha=0.1)
            ax[0].set_ylabel(r'$M_{\eta_h}\,[\mathrm{GeV}]$')
            ax[0].set_xlim(0, xmax)
            ax[0].set_title(f'{self.ens}')





            y = join_stats([self.interpolate(mu, (m,m), key)
                            for m in self.all_masses[start:stop]])
            if not Z_only:
                y = y*am_c*self.ainv

            def Z_m_ansatz(am, param, **kwargs):
                return param[0] + param[1]/am + param[2]*am + param[3]*am**2
            ax[1].text(1.05, 0.5,
                       r'$\alpha + \beta/am_q + \gamma\,am_q + \delta\,(am_q)^2$',
                       va='center', ha='center', rotation=90,
                       color='k', alpha=0.3,
                       transform=ax[1].transAxes)

            res = fit_func(x, y, Z_m_ansatz, [0.1,0.1,0.1,0.1],
                           pause=False, verbose=False)
            Z_m_amstar = res.mapping(am_star)

            ax[1].errorbar(x.val, y.val, xerr=x.err,
                           yerr=y.err, capsize=4, fmt='o')
            if add_pdg:
                ymin, ymax = ax[1].get_ylim()
                ax[1].errorbar([am_star.val], [Z_m_amstar.val],
                               xerr=[am_star.err], yerr=[Z_m_amstar.err],
                               fmt='o', capsize=4, color='r')
                ax[1].vlines(x=am_star.val, ymin=ymin, ymax=ymax,
                             color='r', linestyle='dashed')
                ax[1].fill_between(np.linspace(am_star.val-am_star.err,
                                               am_star.val+am_star.err,100),
                                   ymin, eta_star.val+eta_star.err,
                                   color='r', alpha=0.2)

                ax[1].set_ylim([ymin, ymax])

            xmin, xmax = ax[1].get_xlim()
            xrange = np.linspace(x.val[0], xmax, 100)
            yrange = res.mapping(xrange)
            ax[1].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                    str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                    ha='center', va='center',
                    transform=ax[1].transAxes)
            ax[1].fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color='k', alpha=0.1)

            SMOM = Z_bl_analysis(self.ens, renorm='SMOM')
            Z_m_SMOM = SMOM.plot_mass_dependence(mu, eta_pdg, eta_star,
                                                 key=key, stop=stop,
                                                 open_file=False,
                                                 key_only=True,
                                                 pass_vals=True)
            Z_m_amc_SMOM = Z_m_SMOM
            if not Z_only:
                Z_m_amc_SMOM = Z_m_amc_SMOM*am_c*self.ainv

            ax[1].axhspan(Z_m_amc_SMOM.val+Z_m_amc_SMOM.err,
                          Z_m_amc_SMOM.val-Z_m_amc_SMOM.err,
                          color='r', alpha=0.1)

            label = r'\,\mathrm{GeV})$' if Z_only else\
                    r'\,\mathrm{GeV})\cdot am_c\cdot a^{-1}\, [\mathrm{GeV}]$'
            ax[1].set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+label)
            ax[1].set_xlim(0, xmax)
            if Z_only:
                ax[1].set_xlabel(r'$am_q+am_\mathrm{res}$')






            if not Z_only:
                y = (x*y)*self.ainv
                def Z_m_m_q_ansatz(am, param, **kwargs):
                    return param[0]*am + param[1] + param[2]*am**2
                ax[2].text(1.05, 0.5,
                           r'$\alpha + \beta\,am_q + \gamma\,(am_q)^2$',
                           va='center', ha='center', rotation=90,
                           color='k', alpha=0.3,
                           transform=ax[2].transAxes)

                res = fit_func(x, y, Z_m_m_q_ansatz, [0.1,1,1], verbose=False)
                mbar = res.mapping(am_star) 

                ax[2].errorbar(x.val, y.val, xerr=x.err,
                               yerr=y.err,
                            capsize=4, fmt='o')
                if add_pdg:
                    ymin, ymax = ax[2].get_ylim()
                    ax[2].errorbar([am_star.val], [mbar.val],
                                   xerr=[am_c.err], yerr=[mbar.err],
                                   fmt='o', capsize=4, color='r')
                    ax[2].vlines(x=am_star.val, ymin=ymin, ymax=ymax,
                                 color='r', linestyle='dashed')
                    ax[2].fill_between(np.linspace(am_star.val-am_star.err,
                                                   am_star.val+am_star.err,100),
                                       ymin, eta_star.val+eta_star.err,
                                       color='r', alpha=0.2)

                    ax[2].set_ylim([ymin, ymax])
                xmin, xmax = ax[2].get_xlim()
                xrange = np.linspace(x.val[0], xmax, 100)
                yrange = res.mapping(xrange)
                ax[2].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                        str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                        ha='center', va='center',
                        transform=ax[2].transAxes)
                ax[2].fill_between(xrange, yrange.val+yrange.err,
                                yrange.val-yrange.err,
                                color='k', alpha=0.1)
                ax[2].set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+\
                        '\,\mathrm{GeV})\cdot am_q\cdot a^{-1}\, [\mathrm{GeV}]$')
                ax[2].set_xlabel(r'$am_q+am_\mathrm{res}$')
                #ax[2].set_xlim([0.001, 0.005])

            if pass_vals:
                plt.close(fig)
                return Z_m_amstar, Z_m_amc_SMOM, mbar
            else:
                call_PDF(filename, open=open_file)




    def interpolate(self, mu, masses, key,
                    method='scipy', plot=False,
                    pass_plot=False, **kwargs):
        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        if method == 'fit':
            mapping = self.fit_momentum_dependence(
                masses, key, **kwargs)
            Z_mu = mapping(mu)
        else:
            Z_mu = stat(
                val=interp1d(x.val, y.val,
                             fill_value='extrapolate')(mu),
                err='fill',
                btsp=np.array([interp1d(x.btsp[k], y.btsp[k],
                                        fill_value='extrapolate')(mu)
                               for k in range(N_boot)])
            )
            if pass_plot or plot:
                plt = self.plot_momentum_dependence(
                    masses, key, pass_fig=True, **kwargs)

        if pass_plot or plot:
            plt.errorbar([mu], Z_mu.val, yerr=Z_mu.err,
                         capsize=4, fmt='o', color='k')
            if plot:
                filename = f'plots/Z_{key}_v_ap.pdf'
                call_PDF(filename)

            if pass_plot:
                return Z_mu, plt
        return Z_mu

    def fit_momentum_dependence(self, masses, key,
                                plot=False, fittype='quadratic',
                                pass_plot=False, normalise=False,
                                **kwargs):

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]
        if normalise:
            q = self.Z[masses]['q']
            y = y/q

        if fittype == 'quadratic':
            guess = [1, 1e-1, 1e-2]

            def ansatz(mus, param, **kwargs):
                return param[0] + param[1]*mus + param[2]*(mus**2)
        elif fittype == 'linear':
            guess = [1, 1e-1]

            def ansatz(mus, param, **kwargs):
                return param[0] + param[1]*mus

        res = fit_func(x, y, ansatz, guess)

        if pass_plot or plot:
            plt = self.plot_momentum_dependence(
                masses, key, pass_fig=True, **kwargs)
            xmin, xmax = plt.xlim()
            mu_grain = np.linspace(xmin, xmax, 100)
            Z_grain = res.mapping(mu_grain)
            plt.fill_between(mu_grain,
                             Z_grain.val+Z_grain.err,
                             Z_grain.val-Z_grain.err,
                             color='0.8')
            plt.text(0.5, 0.9, r'$\chi^2$/DOF:'+str(np.around(
                res.chi_sq/res.DOF, 3)), va='center', ha='center',
                transform=plt.gca().transAxes)

            filename = f'plots/Z_{key}_v_ap.pdf'
            if plot:
                call_PDF(filename)
            if pass_plot:
                return res.mapping, plt
        return res.mapping

    def plot_momentum_dependence(self, masses, key,
                                 pass_fig=False, **kwargs):
        plt.figure()

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        plt.errorbar(x.val, y.val,
                     xerr=x.err, yerr=y.err,
                     capsize=4, fmt='o:')
        plt.xlabel(r'$\mu$ (GeV)')
        plt.ylabel(r'$Z_'+key+r'$')

        if pass_fig:
            return plt
        else:
            filename = f'plots/Z_{key}_v_ap.pdf'
            call_PDF(filename)


class cont_extrap:

    def __init__(self, ens_list):
        self.ens_list = ens_list
        self.mSMOM_dict = {ens: Z_bl_analysis(ens, renorm='mSMOM')
                          for ens in ens_list}
        self.SMOM_dict = {ens: Z_bl_analysis(ens, renorm='SMOM')
                          for ens in ens_list}

        self.a_sq = join_stats([val.ainv**(-2)
                                for key, val in self.mSMOM_dict.items()])

    def load_renorm_masses(self, mu, eta_pdg, eta_star, **kwargs):
        m_c_ren_mSMOM = []
        m_c_ren_SMOM = []
        m_bar = []

        for ens in self.ens_list:
            quantities = self.mSMOM_dict[ens].plot_mass_dependence(
                    mu, eta_pdg, eta_star, key='m', stop=-2, 
                    open_file=False, pass_vals=True, **kwargs)

            m_c_ren_mSMOM.append(quantities[0])
            m_c_ren_SMOM.append(quantities[1])
            m_bar.append(quantities[2])

        return join_stats(m_c_ren_mSMOM), join_stats(m_c_ren_SMOM),\
                join_stats(m_bar)

    def plot_cont_extrap(self, mu, eta_pdg, eta_star, **kwargs):

        ansatz, guess = self.ansatz()
        x = self.a_sq
        y_mc_mSMOM, y_mc_SMOM, y_mbar = self.load_renorm_masses(
                mu, eta_pdg, eta_star)

        fig, ax = plt.subplots(nrows=2, ncols=1,
                               sharex='col',
                               figsize=(3,5))
        plt.subplots_adjust(hspace=0, wspace=0)


        ax[0].errorbar(x.val, y_mc_mSMOM.val, 
                       xerr=x.err, yerr=y_mc_mSMOM.err,
                       fmt='o', capsize=4, color='b', 
                       mfc='None', label='mSMOM')

        res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess)
        y_mc_mSMOM_phys = res_mc_mSMOM[0]

        xmin, xmax = ax[0].get_xlim()
        xrange = np.linspace(0, xmax)
        yrange = res_mc_mSMOM.mapping(xrange)

        ax[0].errorbar([0.0], [y_mc_mSMOM_phys.val],
                       yerr=[y_mc_mSMOM_phys.err],
                       capsize=4, fmt='o', color='b')
        ax[0].fill_between(xrange, yrange.val+yrange.err,
                           yrange.val-yrange.err,
                           color='b', alpha=0.2)

        ax[0].errorbar(x.val, y_mc_SMOM.val, 
                       xerr=x.err, yerr=y_mc_SMOM.err,
                       fmt='o', capsize=4, color='k',
                       mfc='None', label='SMOM')

        res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
        y_mc_SMOM_phys = res_mc_SMOM[0]

        yrange = res_mc_SMOM.mapping(xrange)
        ax[0].errorbar([0.0], [y_mc_SMOM_phys.val],
                       yerr=[y_mc_SMOM_phys.err],
                       capsize=4, fmt='o', color='k')
        ax[0].fill_between(xrange, yrange.val+yrange.err,
                           yrange.val-yrange.err,
                           color='k', alpha=0.1)

        ax[1].errorbar(x.val, y_mbar.val, 
                       xerr=x.err, yerr=y_mbar.err,
                       fmt='o', capsize=4, color='b',
                       mfc='None', label='mSMOM')

        res_mbar = fit_func(x, y_mbar, ansatz, guess)
        y_mbar_phys = res_mbar[0]

        yrange = res_mbar.mapping(xrange)
        ax[1].errorbar([0.0], [y_mbar_phys.val],
                       yerr=[y_mbar_phys.err],
                       capsize=4, fmt='o', color='b')
        ax[1].fill_between(xrange, yrange.val+yrange.err,
                           yrange.val-yrange.err,
                           color='b', alpha=0.2)
        ax[1].set_xlim([-0.02, xmax])

        ax[0].set_title(r'$M_{\eta_c}:'+err_disp(eta_pdg.val, eta_pdg.err)+\
                r', M^\star:'+err_disp(eta_star.val, eta_star.err)+r'$')
        ax[0].set_ylabel(r'$m_{c,R}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
        ax[1].set_ylabel(r'$\overline{m}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
        ax[1].set_xlabel(r'$a^2\,[\mathrm{GeV}^2]$')

        for idx in range(2):
            ymin, ymax = ax[idx].get_ylim()
            ax[idx].set_ylim([ymin, ymax])
            ax[idx].vlines(x=0.0, ymin=ymin, ymax=ymax,
                         color='k', linestyle='dashed')
            ax[idx].set_ylim([ymin, ymax])

        ax[0].legend()

        filename = '_'.join(self.ens_list)+'_cont_extrap.pdf'
        call_PDF(filename, open=True)

    def plot_M_eta_pdg_variations(self, mu, M_pdg_list, eta_star,
                                  filename='', **kwargs):
        ansatz, guess = self.ansatz()
        if type(eta_star)==list:
            print('eta_star cannot be a list!')
        else:
            x = self.a_sq
            fig, ax = plt.subplots(nrows=2, ncols=1,
                                   sharex='col',
                                   figsize=(3,5.5),
                                   gridspec_kw={'height_ratios':[2,1]})
            plt.subplots_adjust(hspace=0, wspace=0)
            for eta_idx, eta_pdg in enumerate(M_pdg_list):
                label = r'$M_{\eta_c}$:'+err_disp(eta_pdg.val, eta_pdg.err)
                color = color_list[eta_idx]

                y_mc_mSMOM, y_mc_SMOM, y_mbar = self.load_renorm_masses(
                        mu, eta_pdg, eta_star)
                ax[0].errorbar(x.val, y_mc_mSMOM.val, 
                               xerr=x.err, yerr=y_mc_mSMOM.err,
                               fmt='o', capsize=4, color=color, 
                               mfc='None', label=label)

                res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess)
                y_mc_mSMOM_phys = res_mc_mSMOM[0]

                if eta_idx==0:
                    xmin, xmax = ax[0].get_xlim()
                xrange = np.linspace(0, xmax)
                yrange = res_mc_mSMOM.mapping(xrange)

                ax[0].errorbar([0.0], [y_mc_mSMOM_phys.val],
                               yerr=[y_mc_mSMOM_phys.err],
                               capsize=4, fmt='o', color=color)
                ax[0].fill_between(xrange, yrange.val+yrange.err,
                                   yrange.val-yrange.err,
                                   color=color, alpha=0.2)

                ax[0].errorbar(x.val, y_mc_SMOM.val, 
                               xerr=x.err, yerr=y_mc_SMOM.err,
                               fmt='x', capsize=4, color=color,
                               mfc='None')

                res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
                y_mc_SMOM_phys = res_mc_SMOM[0]

                yrange = res_mc_SMOM.mapping(xrange)
                ax[0].errorbar([0.0], [y_mc_SMOM_phys.val],
                               yerr=[y_mc_SMOM_phys.err],
                               capsize=4, fmt='x', color=color)
                ax[0].fill_between(xrange, yrange.val+yrange.err,
                                   yrange.val-yrange.err,
                                   color=color, alpha=0.1)

            ax[1].errorbar(x.val, y_mbar.val, 
                           xerr=x.err, yerr=y_mbar.err,
                           fmt='o', capsize=4, color='k',
                           mfc='None', label=label)

            res_mbar = fit_func(x, y_mbar, ansatz, guess)
            y_mbar_phys = res_mbar[0]

            yrange = res_mbar.mapping(xrange)
            ax[1].errorbar([0.0], [y_mbar_phys.val],
                           yerr=[y_mbar_phys.err],
                           capsize=4, fmt='o', color='k')
            ax[1].fill_between(xrange, yrange.val+yrange.err,
                               yrange.val-yrange.err,
                               color='k', alpha=0.2)
            ax[1].set_xlim([-0.02, xmax])

            ax[0].set_title(r'$M^\star:'+err_disp(eta_star.val, eta_star.err)+r'$')
            ax[0].set_ylabel(r'$m_{c,R}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
            ax[1].set_ylabel(r'$\overline{m}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
            ax[1].set_xlabel(r'$a^2\,[\mathrm{GeV}^2]$')

            ax[0].legend(bbox_to_anchor=(1, 0.4))

            for idx in range(2):
                ymin, ymax = ax[idx].get_ylim()
                ax[idx].set_ylim([ymin, ymax])
                ax[idx].vlines(x=0.0, ymin=ymin, ymax=ymax,
                             color='k', linestyle='dashed')
                ax[idx].set_ylim([ymin, ymax])

            if filename=='':
                filename = '_'.join(self.ens_list)+'_cont_extrap.pdf'
            call_PDF(filename, open=True)

    def plot_M_star_variations(self, mu, eta_pdg, M_star_list,
                               filename='', **kwargs):
        ansatz, guess = self.ansatz()
        if type(eta_pdg)==list:
            print('eta_PDG cannot be a list!')
        else:
            x = self.a_sq
            fig, ax = plt.subplots(nrows=2, ncols=1,
                                   sharex='col',
                                   figsize=(3,5))
            plt.subplots_adjust(hspace=0, wspace=0)
            for eta_idx, eta_star in enumerate(M_star_list):
                label = r'$M^\star$:'+err_disp(eta_star.val, eta_star.err)
                color = color_list[eta_idx]

                y_mc_mSMOM, y_mc_SMOM, y_mbar = self.load_renorm_masses(
                        mu, eta_pdg, eta_star)
                ax[0].errorbar(x.val, y_mc_mSMOM.val, 
                               xerr=x.err, yerr=y_mc_mSMOM.err,
                               fmt='o', capsize=4, color=color, 
                               mfc='None', label=label)

                res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess)
                y_mc_mSMOM_phys = res_mc_mSMOM[0]

                xmin, xmax = ax[0].get_xlim()
                xrange = np.linspace(0, xmax)
                yrange = res_mc_mSMOM.mapping(xrange)

                ax[0].errorbar([0.0], [y_mc_mSMOM_phys.val],
                               yerr=[y_mc_mSMOM_phys.err],
                               capsize=4, fmt='o', color=color)
                ax[0].fill_between(xrange, yrange.val+yrange.err,
                                   yrange.val-yrange.err,
                                   color=color, alpha=0.2)

                ax[1].errorbar(x.val, y_mbar.val, 
                               xerr=x.err, yerr=y_mbar.err,
                               fmt='o', capsize=4, color=color,
                               mfc='None', label=label)

                res_mbar = fit_func(x, y_mbar, ansatz, guess)
                y_mbar_phys = res_mbar[0]

                yrange = res_mbar.mapping(xrange)
                ax[1].errorbar([0.0], [y_mbar_phys.val],
                               yerr=[y_mbar_phys.err],
                               capsize=4, fmt='o', color=color)
                ax[1].fill_between(xrange, yrange.val+yrange.err,
                                   yrange.val-yrange.err,
                                   color=color, alpha=0.2)
                ax[1].set_xlim([-0.02, xmax])

            ax[0].errorbar(x.val, y_mc_SMOM.val, 
                           xerr=x.err, yerr=y_mc_SMOM.err,
                           fmt='x', capsize=4, color='k',
                           mfc='None')

            res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
            y_mc_SMOM_phys = res_mc_SMOM[0]

            yrange = res_mc_SMOM.mapping(xrange)
            ax[0].errorbar([0.0], [y_mc_SMOM_phys.val],
                           yerr=[y_mc_SMOM_phys.err],
                           capsize=4, fmt='x', color='k')
            ax[0].fill_between(xrange, yrange.val+yrange.err,
                               yrange.val-yrange.err,
                               color='k', alpha=0.1)

            ax[0].set_title(r'$M_{\eta_c}:'+\
                    err_disp(eta_pdg.val, eta_pdg.err)+r'$')
            ax[0].set_ylabel(r'$m_{c,R}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
            ax[1].set_ylabel(r'$\overline{m}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
            ax[1].set_xlabel(r'$a^2\,[\mathrm{GeV}^2]$')

            ax[1].legend(bbox_to_anchor=(1,0.4))

            for idx in range(2):
                ymin, ymax = ax[idx].get_ylim()
                ax[idx].set_ylim([ymin, ymax])
                ax[idx].vlines(x=0.0, ymin=ymin, ymax=ymax,
                             color='k', linestyle='dashed')
                ax[idx].set_ylim([ymin, ymax])

            if filename=='':
                filename = '_'.join(self.ens_list)+'_cont_extrap.pdf'
            call_PDF(filename, open=True)

    def loop_over_masses(self, mu, mass_list, name_list,
                         filepath, **kwargs):

        ens_label = '_'.join(self.ens_list)
        for idx, eta_star in enumerate(mass_list):
            filename = f'{filepath}/{ens_label}_M_eta_star_{name_list[idx]}'+\
                    f'_mu_{int(mu)}GeV.pdf'
            self.plot_M_eta_pdg_variations(mu, mass_list, eta_star, 
                                        filename=filename, **kwargs)

        for idx, eta_pdg in enumerate(mass_list):
            filename = f'{filepath}/{ens_label}_M_eta_c_{name_list[idx]}'+\
                    f'_mu_{int(mu)}GeV.pdf'
            self.plot_M_star_variations(mu, eta_pdg, mass_list, 
                                     filename=filename, **kwargs)

    def ansatz(self,**kwargs):

        def linear_ansatz(asq, param, **kwargs):
            return param[0] + param[1]*asq
        def quadratic_ansatz(asq, param, **kwargs):
            return param[0] + param[1]*asq + param[2]*(asq**2)

        ansatz = linear_ansatz if len(self.ens_list)==2 else quadratic_ansatz
        guess = [1, 1e-1] if len(self.ens_list)==2 else [1,1e-1,1e-2]
        return ansatz, guess 



