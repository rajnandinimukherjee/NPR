from NPR_classes import *
from eta_c import *
from coeffs import *


class Z_bl_analysis:

    def __init__(self, ens, action=(0, 0), scheme='qslash',
                 renorm='mSMOM', **kwargs):

        self.ens = ens
        self.datafolder = h5py.File(f'bilinear_Z_{scheme}.h5', 'r')[
            str(action)][ens]

        info = params[self.ens]
        self.ainv = stat(val=info['ainv'], err=info['ainv_err'], btsp='fill')

        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1, len(info['masses']))]
        self.all_masses = [self.sea_mass]+self.non_sea_masses

        self.momenta, self.Z = {}, {}

        for m in self.datafolder.keys():
            masses = (m[2:8], m[12:18])
            self.momenta[masses] = stat(
                val=np.array(self.datafolder[m]['momenta'][:]),
                btsp='fill')

            self.Z[masses] = {}
            for key in self.datafolder[m].keys():
                if key != 'momenta':
                    self.Z[masses][key] = stat(
                        val=np.array(self.datafolder[m][key]['central'][:]),
                        err=np.array(self.datafolder[m][key]['errors'][:]),
                        btsp=np.array(self.datafolder[m][key]['bootstrap'][:])
                    )

    def interpolate(self, mu, masses, key,
                    method='fit', plot=False,
                    pass_plot=False, **kwargs):
        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        if method == 'fit':
            mapping, plt = self.fit_momentum_dependence(
                masses, key, pass_plot=True)
            Z_mu = mapping(mu)
        else:
            Z_mu = stat(
                val=interp1d(x.val, y.val, fill_value='extrapolate')(mu),
                err='fill',
                btsp=np.array([interp1d(x.btsp[k], y.btsp[k], fill_value='extrapolate')(mu)
                               for k in range(N_boot)])
            )
            plt = self.plot_momentum_dependence(masses, key, pass_fig=True)

        plt.errorbar([mu], Z_mu.val, yerr=Z_mu.err,
                     capsize=4, fmt='o', color='k')

        if plot:
            filename = f'plots/Z_{key}_v_ap.pdf'
            call_PDF(filename)

        returns = [Z_mu]
        if pass_plot:
            returns.append(plt)

        return returns

    def fit_momentum_dependence(self, masses, key,
                                plot=False, guess=[1, 1e-1, 1e-2],
                                pass_plot=False, **kwargs):

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        COV = np.diag(y.err**2)
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def ansatz(mus, param, **kwargs):
            return param[0] + param[1]*mus + param[2]*(mus**2)

        def diff(in_, out_, param, **kwargs):
            return out_ - ansatz(in_, param, **kwargs)

        def LD(param):
            return L.dot(diff(x.val, y.val, param, fit='central'))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        DOF = len(x.val)-len(guess)

        res_btsp = np.zeros(shape=(N_boot, len(res.x)))
        for k in range(N_boot):
            def LD_btsp(param):
                return L.dot(diff(x.btsp[k], y.btsp[k], param))
            res_k = least_squares(LD_btsp, guess,
                                  ftol=1e-10, gtol=1e-10)
            res_btsp[k, :] = res_k.x

        def fit_mapping(m):
            return stat(val=ansatz(m, res.x),
                        err='fill',
                        btsp=np.array([ansatz(m, res_btsp[k])
                                       for k in range(N_boot)])
                        )

        returns = [fit_mapping]

        plt = self.plot_momentum_dependence(masses, key, pass_fig=True)
        xmin, xmax = plt.xlim()
        mu_grain = np.linspace(xmin, xmax, 100)
        Z_grain = fit_mapping(mu_grain)
        plt.fill_between(mu_grain,
                         Z_grain.val+Z_grain.err,
                         Z_grain.val-Z_grain.err,
                         color='0.8')
        plt.text(0.5, 0.9, r'$\chi^2$/DOF:'+str(np.around(chi_sq/DOF, 3)),
                 va='center', ha='center', transform=plt.gca().transAxes)

        filename = f'plots/Z_{key}_v_ap.pdf'
        if plot:
            call_PDF(filename)
        if pass_plot:
            returns.append(plt)

        return returns

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

    def __init__(self, ens, mu=2.0, mres=True, **kwargs):
        self.ens = ens
        self.mu = mu
        self.mres = mres

        self.ainv = stat(
            val=float(params[ens]['ainv']),
            err=float(params[ens]['ainv_err']),
            btsp='fill'
        )
        self.load_eta()

        self.SMOM_bl = bilinear_analysis(
            self.ens, mres=self.mres,
            loadpath=f'{prefix}/{self.ens}_bl_massive_SMOM.p')

        self.mSMOM_bl = bilinear_analysis(
            self.ens, mres=self.mres,
            loadpath=f'{prefix}/{self.ens}_bl_massive_mSMOM.p')
        self.all_masses = self.mSMOM_bl.all_masses
        self.N_masses = len(self.all_masses)

        self.m_C = self.am_C*self.ainv
        self.load_SMOM()

        self.Z_m_mSMOM_map = self.mSMOM_func(key='m')
        self.am_bar_mSMOM_map = self.mSMOM_func(key='mam_q')

    def load_SMOM(self, **kwargs):
        sea_mass = self.SMOM_bl.sea_mass
        self.Z_SMOM = self.SMOM_bl.extrap_Z(
            self.mu, masses=(sea_mass, sea_mass))['m']
        Z_P_SMOM = self.SMOM_bl.extrap_Z(
            self.mu, masses=(sea_mass, sea_mass))['P']
        self.Z_P_inv_SMOM = Z_P_SMOM**(-1)
        am_bar_SMOM = self.SMOM_bl.extrap_Z(
            self.mu, masses=(sea_mass, sea_mass))['mam_q']

        self.m_C_SMOM = self.Z_SMOM*self.m_C
        self.m_bar_SMOM = am_bar_SMOM*self.ainv

    def load_mSMOM(self, key='m', **kwargs):
        am_in = stat(
            val=list(eta_c_data[self.ens]['central'].keys()),
            btsp='fill'
        )

        mSMOM = np.array([self.mSMOM_bl.extrap_Z(
            mu=self.mu, masses=(m, m))[key]
            for m in self.all_masses])
        mSMOM = stat(
            val=np.array([mSMOM[m].val for m in range(self.N_masses)]),
            err=np.array([mSMOM[m].err for m in range(self.N_masses)]),
            btsp=np.array([mSMOM[m].btsp for m in range(self.N_masses)]).T
        )
        return am_in, mSMOM

    def calc_mSMOM(self, eta_star, **kwargs):
        am_star = self.interpolate_eta_c(eta_star)
        m_C_mSMOM = self.Z_m_mSMOM_map(am_star)*self.m_C
        m_bar_mSMOM = self.am_bar_mSMOM_map(am_star)*self.ainv
        # alternate fitting strategy
        # m_C_mSMOM = m_bar_mSMOM*self.m_C/(am_star*self.ainv)
        return m_C_mSMOM, m_bar_mSMOM

    def mSMOM_func(self, key='m', start=0, **kwargs):
        am_in, mSMOM = self.load_mSMOM(key=key)

        def ansatz(params, am):
            if key == 'm':
                return params[0] + params[1]*am + params[2]/am
            if key == 'mam_q':
                return params[0]*am + params[1]*(am**2) + params[2]

        # central fit
        e = 2 if self.ens == 'C1' else 1
        x = am_in.val[start:-e]
        y = mSMOM.val[start:-e]
        COV = np.diag(mSMOM.err[start:-e]**2)

        def diff(params):
            return y - ansatz(params, x)

        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params):
            return L.dot(diff(params))

        guess = [1, 1, 1]
        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        central = res.x

        btsp = np.zeros(shape=(N_boot, len(guess)))
        for k in range(N_boot):
            x_k = am_in.btsp[k, start:-e]
            y_k = mSMOM.btsp[k, start:-e]

            def diff_k(params):
                return y_k - ansatz(params, x_k)

            def LD_k(params):
                return L.dot(diff_k(params))
            res = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
            btsp[k,] = res.x

        def mapping(am):
            if not isinstance(am, stat):
                am = stat(
                    val=np.array(list(am)),
                    err=np.zeros(len(list(am))),
                    btsp='fill')

            return stat(
                val=ansatz(central, am.val),
                err='fill',
                btsp=np.array([ansatz(btsp[k,], am.btsp[k,])
                               for k in range(N_boot)])
            )
        return mapping

    def calc_m_C(self, **kwargs):
        am_in = stat(
            val=[float(m) for m in self.all_masses],
            btsp='fill'
        )

        am_poles = np.array([self.mSMOM_bl.extrap_Z(
            mu=self.mu, masses=(m, m))['m_q']
            for m in self.all_masses])
        am_poles = stat(
            val=np.array([am_poles[m].val for m in range(self.N_masses)]),
            err=np.array([am_poles[m].err for m in range(self.N_masses)]),
            btsp=np.array([am_poles[m].btsp for m in range(self.N_masses)]).T
        )

        f = interp1d(am_in.val, am_poles.val, fill_value='extrapolate')
        central = f(self.am_C.val)

        btsp = np.zeros(N_boot)
        for k in range(N_boot):
            f_k = interp1d(am_in.btsp[k,], am_poles.btsp[k,],
                           fill_value='extrapolate')
            btsp[k] = f_k(self.am_C.btsp[k])

        am_C_pole = stat(
            val=central,
            err='fill',
            btsp=btsp
        )
        m_C_pole = am_C_pole*self.ainv
        return m_C_pole

    def load_eta(self, **kwargs):
        self.eta_c_data = eta_c_data[self.ens]

        ax = np.array(list(self.eta_c_data['central'].keys()))
        self.eta_ax = stat(
            val=ax,
            err=np.zeros(ax.shape[0]),
            btsp='fill'
        )
        self.eta_ay = stat(
            val=np.array([self.eta_c_data['central'][x_q]
                          for x_q in ax]),
            err=np.array([self.eta_c_data['errors'][x_q]
                          for x_q in ax]),
            btsp='fill'
        )
        self.eta_y = self.eta_ay*self.ainv

        self.am_C = self.interpolate_eta_c(find_y=eta_PDG)

    def interpolate_eta_c(self, find_y, **kwargs):
        if not isinstance(find_y, stat):
            find_y = stat(
                val=find_y,
                err=0,
                btsp='fill'
            )

        pred = stat(
            val=interp1d(self.eta_y.val[1:-1], self.eta_ax.val[1:-1],
                         fill_value='extrapolate')(find_y.val),
            err='fill',
            btsp=np.array([interp1d(self.eta_y.btsp[k, 1:-1],
                                    self.eta_ax.btsp[k, 1:-1],
                                    fill_value='extrapolate'
                                    )(find_y.btsp[k])
                           for k in range(N_boot)])
        )
        return pred


class cont_extrap:
    eta_stars = [2.4, 2.6, eta_PDG.val]

    def __init__(self, ens_list, mres=True, mu=2.0):
        self.ens_list = ens_list
        self.mNPR_dict = {ens: mNPR(ens, mu, mres=mres)
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

    def load_mSMOM(self, eta_star, alternate=False, **kwargs):
        m_C = np.empty(len(self.ens_list), dtype=object)
        m_bar = np.empty(len(self.ens_list), dtype=object)

        for idx, ens in enumerate(self.ens_list):
            e = self.mNPR_dict[ens]
            m_C[idx], m_bar[idx] = e.calc_mSMOM(eta_star)
            if alternate:
                m_star = e.interpolate_eta_c(eta_star)*e.ainv
                m_C[idx] = m_bar[idx]*e.m_C/m_star

        m_C = stat(
            val=np.array([m_C[i].val for i in range(len(self.ens_list))]),
            err=np.array([m_C[i].err for i in range(len(self.ens_list))]),
            btsp=np.array([m_C[i].btsp for i in range(len(self.ens_list))]).T
        )
        m_bar = stat(
            val=np.array([m_bar[i].val for i in range(len(self.ens_list))]),
            err=np.array([m_bar[i].err for i in range(len(self.ens_list))]),
            btsp=np.array([m_bar[i].btsp for i in range(len(self.ens_list))]).T
        )
        return m_C, m_bar

    def extrap_mapping(self, data, fit='quad', **kwargs):

        if fit == 'quad':
            guess = [1, 0.1, 0.1]
        else:
            guess = [1, 0.1]

        def ansatz(params, a_sq):
            if fit == 'quad':
                return params[0] + params[1]*a_sq + params[2]*(a_sq**2)
            elif fit == 'linear':
                return params[0] + params[1]*a_sq

        # central fit
        x = self.a_sq.val
        y = data.val
        COV = np.diag(data.err**2)

        def diff(params):
            return y - ansatz(params, x)

        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params):
            return L.dot(diff(params))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        central = res.x

        btsp = np.zeros(shape=(N_boot, len(guess)))
        for k in range(N_boot):
            x_k = self.a_sq.btsp[k,]
            y_k = data.btsp[k,]

            def diff_k(params):
                return y_k - ansatz(params, x_k)

            def LD_k(params):
                return L.dot(diff_k(params))
            res = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
            btsp[k,] = res.x

        def mapping(a_sq):
            return stat(
                val=ansatz(central, a_sq),
                err='fill',
                btsp=np.array([ansatz(btsp[k,], a_sq)
                               for k in range(N_boot)])
            )
        return mapping
