from NPR_classes import *
from eta_c import *
from coeffs import *


class mNPR:
    mu = 2.0

    def __init__(self, ens, **kwargs):
        self.ens = ens
        self.ainv = stat(
            val=float(params[ens]['ainv']),
            err=float(params[ens]['ainv_err']),
            btsp='fill'
        )
        self.load_eta()

        self.SMOM_bl = bilinear_analysis(
            self.ens, mres=False,
            loadpath=f'no_mres/{self.ens}_bl_massive_SMOM.p')

        self.mSMOM_bl = bilinear_analysis(
            self.ens, mres=False,
            loadpath=f'no_mres/{self.ens}_bl_massive_mSMOM.p')
        self.all_masses = self.mSMOM_bl.all_masses
        self.N_masses = len(self.all_masses)

        self.m_C = self.calc_m_C()
        self.Z_m_mSMOM_map = self.mSMOM_func(key='m')
        self.m_bar_mSMOM_map = self.mSMOM_func(key='mam_q')

    def load_SMOM(self, **kwargs):
        sea_mass = self.SMOM_bl.sea_mass
        Z_SMOM = self.SMOM_bl.extrap_Z(
            self.mu, masses=(sea_mass, sea_mass))['m']
        am_bar_SMOM = self.SMOM_bl.extrap_Z(
            self.mu, masses=(sea_mass, sea_mass))['mam_q']

        m_C_SMOM = Z_SMOM*self.m_C
        m_bar_SMOM = am_bar_SMOM*self.ainv
        return m_C_SMOM, m_bar_SMOM

    def calc_mSMOM(self, eta_star, **kwargs):
        am_star = self.interpolate_eta_c(eta_star)
        m_C_mSMOM = self.Z_m_mSMOM_map(am_star)*self.m_C
        m_bar_mSMOM = self.m_bar_mSMOM_map(am_star)
        return m_C_mSMOM, m_bar_mSMOM

    def mSMOM_func(self, key='m', **kwargs):
        am_in = stat(
            val=[float(m) for m in self.all_masses],
            btsp='fill'
        )

        mSMOM = np.array([self.mSMOM_bl.extrap_Z(
            mu=self.mu, masses=(m, m))['m']
            for m in self.all_masses])
        mSMOM = stat(
            val=np.array([mSMOM[m].val for m in range(self.N_masses)]),
            err=np.array([mSMOM[m].err for m in range(self.N_masses)]),
            btsp=np.array([mSMOM[m].btsp for m in range(self.N_masses)]).T
        )

        def ansatz(params, am):
            if key == 'm':
                return params[0] + params[1]*am + params[2]/am
            elif key == 'mam_q':
                return params[0]*am + params[1]*(am**2) + params[2]

        # central fit
        x = am_in.val
        y = mSMOM.val
        COV = np.diag(mSMOM.err**2)

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
            x_k = am_in.btsp[k,]
            y_k = mSMOM.btsp[k,]

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
        return am_C_pole

    def load_eta(self, **kwargs):
        self.valence = etaCvalence(self.ens)
        self.valence.toDict(keys=list(
            self.valence.mass_comb.keys()), mres=False)
        self.eta_c_data = eta_c_data[self.ens]

        ax = np.array(list(self.eta_c_data['central'].keys()))[:-1]
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
            val=interp1d(self.eta_y.val, self.eta_ax.val,
                         fill_value='extrapolate')(find_y.val),
            err='fill',
            btsp=np.array([interp1d(self.eta_y.btsp[k, :],
                                    self.eta_ax.btsp[k, :],
                                    fill_value='extrapolate'
                                    )(find_y.btsp[k])
                           for k in range(N_boot)])
        )
        return pred
