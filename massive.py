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

    def plot_mass_dependence(self, mu, key='m', start=0,
                             stop=None, open_file=True,
                             filename='', add_PDG=False,
                             pass_vals=False, **kwargs):
        if stop==None:
            stop = len(self.all_masses)

        fig, ax = plt.subplots(nrows=3, ncols=1, 
                               sharex='col',
                               figsize=(3,10))
        plt.subplots_adjust(hspace=0, wspace=0)

        x = join_stats([self.valence.amres[m]+eval(m)
                        for m in self.all_masses[start:stop]])

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
                val=pred_amq(eta_PDG.val, res.val),
                err='fill',
                btsp=[pred_amq(eta_PDG.btsp[k], res.btsp[k])
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
        if add_PDG:
            ymin, ymax = ax[0].get_ylim()
            ax[0].hlines(y=eta_PDG.val, xmin=0, xmax=am_c.val, color='k')
            ax[0].text(0.05, eta_PDG.val-0.05, r'$M_{\eta_c}^\mathrm{PDG}$',
                       va='top', ha='center', color='k')
            ax[0].fill_between(np.linspace(0,am_c.val,100),
                               eta_PDG.val+eta_PDG.err,
                               eta_PDG.val-eta_PDG.err,
                               color='k', alpha=0.2)
            ax[0].vlines(x=am_c.val, ymin=ymin, ymax=eta_PDG.val,
                         color='k', linestyle='dashed')
            ax[0].text(am_c.val, 0.5, r'$am_c$', rotation=90,
                       va='center', ha='right', color='k')
            ax[0].fill_between(np.linspace(am_c.val-am_c.err,
                                           am_c.val+am_c.err,100),
                               ymin, eta_PDG.val+eta_PDG.err,
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
        xrange = np.linspace(0.002, xmax, 100)
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
                        for m in self.all_masses[start:stop]])*am_c*self.ainv
        def Z_m_ansatz(am, param, **kwargs):
            return param[0] + param[1]/am + param[2]*am + param[3]*am**2
        ax[1].text(1.05, 0.5,
                   r'$\alpha + \beta/am_q + \gamma\,am_q + \delta\,(am_q)^2$',
                   va='center', ha='center', rotation=90,
                   color='k', alpha=0.3,
                   transform=ax[1].transAxes)

        res = fit_func(x, y, Z_m_ansatz, [0.1,0.1,0.1,0.1], verbose=False)
        Z_m_amstar = res.mapping(am_star)

        ax[1].errorbar(x.val, y.val, xerr=x.err,
                       yerr=y.err, capsize=4, fmt='o')
        if add_PDG:
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
        xrange = np.linspace(0.002, xmax, 100)
        yrange = res.mapping(xrange)
        ax[1].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                ha='center', va='center',
                transform=ax[1].transAxes)
        ax[1].fill_between(xrange, yrange.val+yrange.err,
                        yrange.val-yrange.err,
                        color='k', alpha=0.1)
        ax[1].set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+\
                #r'\,\mathrm{GeV})$')
                r'\,\mathrm{GeV})\cdot am_c\cdot a^{-1}\, [\mathrm{GeV}]$')
        ax[1].set_xlim(0, xmax)






        y = (x*y)/am_c
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
        if add_PDG:
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
        xrange = np.linspace(0.002, xmax, 100)
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


        if filename=='':
            filename = f'mSMOM_{self.ens}_vs_amq.pdf'

        call_PDF(filename, open=open_file)

        if pass_vals:
            return Z_m_amstar, mbar

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
                                pass_plot=False, **kwargs):

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

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



class mNPR:

    def __init__(self, ens, mu=2.0, **kwargs):
        self.ens = ens
        self.mu = mu

        self.ainv = stat(
            val=float(params[ens]['ainv']),
            err=float(params[ens]['ainv_err']),
            btsp='fill'
        )

        self.SMOM_bl = Z_bl_analysis(self.ens, renorm='SMOM')
        self.mSMOM_bl = Z_bl_analysis(self.ens, renorm='mSMOM')

        self.all_masses = self.mSMOM_bl.all_masses
        self.N_masses = len(self.all_masses)
        self.load_eta()
        self.m_C = self.am_C*self.ainv
        self.load_SMOM()

    def load_SMOM(self, **kwargs):
        sea_mass = self.SMOM_bl.sea_mass
        self.Z_SMOM = self.SMOM_bl.interpolate(
            self.mu, masses=(sea_mass, sea_mass), key='m')
        Z_P_SMOM = self.SMOM_bl.interpolate(
            self.mu, masses=(sea_mass, sea_mass), key='P')
        self.Z_P_inv_SMOM = Z_P_SMOM**(-1)
        am_bar_SMOM = self.SMOM_bl.interpolate(
            self.mu, masses=(sea_mass, sea_mass), key='mam_q')

        self.m_C_SMOM = self.Z_SMOM*self.m_C
        self.m_bar_SMOM = stat(val=0, btsp='fill')

    def load_mSMOM(self, key='m', **kwargs):
        mSMOM = np.array(
            [self.mSMOM_bl.interpolate(
                mu=self.mu, masses=(m, m), key=key)
             for m in self.all_masses[:len(self.eta_ax.val)]])
        mSMOM = join_stats(mSMOM)

        return self.eta_ax, mSMOM

    def load_eta(self, **kwargs):
        self.v = valence(self.ens)
        self.v.load_from_H5()
        info = self.v.mass_dict

        ax, ay = [], []
        for mass in self.all_masses:
            m = str(float(mass))
            if m == '0.0214':
                m = '0.02144'
            if 'aM_eta_h' in info[m] and 'am_res' in info[m]:
                ax.append(info[m]['am_res'] + float(mass))
                ay.append(info[m]['aM_eta_h'])

        self.eta_ax = join_stats(ax)
        self.eta_ay = join_stats(ay)
        self.eta_y = self.eta_ay*self.ainv

        if self.ens == 'C1':
            start = 0
        else:
            start = 1
        self.am_C = self.interpolate_eta_c(find=eta_PDG, start=start)

    def interpolate_eta_c(self, find, method='fit',
                          fittype='quadratic',
                          start=0, end=None,
                          plot=False, **kwargs):
        x, y = self.eta_y, self.eta_ax
        if not isinstance(find, stat):
            find = stat(
                val=find,
                err=0,
                btsp='fill'
            )

        if method == 'fit':
            if fittype == 'linear':
                guess = [1, 1]

                def ansatz(eta_h, param, **kwargs):
                    return param[0] + param[1]*eta_h

            else:
                guess = [1, 1, 1e-2]

                def ansatz(eta_h, param, **kwargs):
                    return param[0] + param[1]*eta_h + param[2]*eta_h**2

            res = fit_func(x, y, ansatz, guess,
                           start=start, end=end)
            pred = res.mapping(find)

            if plot:
                plt.errorbar(x.val, y.val,
                             xerr=x.err,
                             yerr=y.err,
                             capsize=4, fmt='o')
                plt.errorbar([find.val], [pred.val],
                             xerr=[find.err], yerr=[pred.err],
                             color='k', capsize=4, fmt='o')
                xmin, xmax = plt.xlim()
                x_grain = np.linspace(xmin, xmax, 100)
                y_grain = res.mapping(x_grain)
                plt.fill_between(x_grain, y_grain.val+y_grain.err,
                                 y_grain.val-y_grain.err,
                                 color='0.5')
                plt.xlim([xmin, xmax])
                plt.text(0.5, 0.9, r'$\chi^2$/DOF:'+str(np.around(
                    res.chi_sq/res.DOF, 3)), va='center', ha='center',
                    transform=plt.gca().transAxes)
                plt.xlabel(r'$\eta_h$ (GeV)')
                plt.ylabel(r'$am$')
                filename = f'plots/eta_h_fit_{self.ens}.pdf'
                call_PDF(filename)
        else:
            pred = stat(
                val=interp1d(x.val[start:end], y.val[start:end],
                             fill_value='extrapolate')(find.val),
                err='fill',
                btsp=np.array([interp1d(x.btsp[k, start:end],
                                        y.btsp[k, start:end],
                                        fill_value='extrapolate'
                                        )(find.btsp[k])
                               for k in range(N_boot)])
            )
        return pred


class cont_extrap:
    eta_stars = [2.4, 2.6, eta_PDG.val]

    def __init__(self, ens_list, mu=2.0):
        self.ens_list = ens_list
        self.mNPR_dict = {ens: mNPR(ens, mu)
                          for ens in ens_list}
        self.a_sq = stat(
            val=np.array([e.ainv.val**(-2)
                          for ens, e in self.mNPR_dict.items()]),
            err='fill',
            btsp=np.array([[e.ainv.btsp[k,]**(-2)
                           for ens, e in self.mNPR_dict.items()]
                           for k in range(N_boot)])
        )

    def load_SMOM(self):
        m_C = [e.m_C_SMOM for ens, e in self.mNPR_dict.items()]
        self.m_C_SMOM = stat(
            val=[m.val for m in m_C],
            err=[m.err for m in m_C],
            btsp=np.array([m.btsp for m in m_C]).T
        )
        m_bar = [e.m_bar_SMOM for ens, e in self.mNPR_dict.items()]
        self.m_bar_SMOM = stat(
            val=[m.val for m in m_bar],
            err=[m.err for m in m_bar],
            btsp=np.array([m.btsp for m in m_bar]).T
        )

    def extrap_mapping(self, data, fit='quad', **kwargs):

        if fit == 'quad':
            guess = [1, 0.1, 0.1]

            def ansatz(a_sq, params):
                return params[0] + params[1]*a_sq + params[2]*(a_sq**2)
        else:
            guess = [1, 0.1]

            def ansatz(a_sq, params):
                return params[0] + params[1]*a_sq

        x = self.a_sq
        y = data
        res = fit_func(x, y, ansatz, guess)
        return res.mapping
