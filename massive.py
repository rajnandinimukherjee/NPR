from NPR_classes import *
from valence import *
from coeffs import *

eta_PDG = stat(
    val=2983.9/1000,
    err=0.5/1000,
    btsp='fill')

eta_stars = [1.6, 2.2, float(eta_PDG.val)]


class Z_bl_analysis:

    def __init__(self, ens, action=(0, 0), scheme='qslash',
                 renorm='mSMOM', **kwargs):

        self.ens = ens
        self.datafolder = h5py.File(
            f'bilinear_Z_{scheme}_{renorm}.h5', 'r')[
            str(action)][ens]

        info = params[self.ens]
        self.ainv = stat(
            val=info['ainv'],
            err=info['ainv_err'],
            btsp='fill')

        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1, len(info['masses']))]
        self.all_masses = [self.sea_mass]+self.non_sea_masses

        self.momenta, self.Z = {}, {}

        for m in self.datafolder.keys():
            masses = (m[2:8], m[12:18])
            self.momenta[masses] = stat(
                val=self.datafolder[m]['momenta'][:],
                btsp='fill')

            self.Z[masses] = {}
            for key in self.datafolder[m].keys():
                if key != 'momenta':
                    self.Z[masses][key] = stat(
                        val=self.datafolder[m][key]['central'][:],
                        err=self.datafolder[m][key]['errors'][:],
                        btsp=self.datafolder[m][key]['bootstrap'][:]
                    )

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
