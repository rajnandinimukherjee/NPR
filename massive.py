from NPR_classes import *
from eta_c import *
from coeffs import *


class mNPR:
    def __init__(self, ens, **kwargs):
        self.ens = ens
        self.ainv = stat(
            val=float(params[ens]['ainv']),
            err=float(params[ens]['ainv_err']),
            btsp='fill'
        )
        self.load_eta()

    def load_SMOM(self, mu, **kwargs):
        self.SMOM_bl = bilinear_analysis(
            self.ens, mres=False,
            loadpath=f'no_mres/{self.ens}_bl_massive_SMOM.p')

        sea_mass = self.SMOM_bl.sea_mass
        self.Z_SMOM = self.SMOM_bl.extrap_Z(
            mu, masses=(sea_mass, sea_mass))['m']
        self.m_C_SMOM = stat(
            val=self.Z_SMOM.val*self.m_C.val,
            err='fill',
            btsp=np.array([self.Z_SMOM.btsp[k]*self.m_C.btsp[k]
                           for k in range(N_boot)])
        )
        self.am_bar_SMOM = self.SMOM_bl.extrap_Z(
            mu, masses=(sea_mass, sea_mass))['mam_q']
        self.m_bar_SMOM = stat(
            val=self.am_bar_SMOM.val*self.ainv.val,
            err='fill',
            btsp=np.array(
                [self.am_bar_SMOM.btsp[k]*self.ainv.btsp[k]
                 for k in range(N_boot)])
        )

    def load_mSMOM(self, mu, **kwargs):
        self.mSMOM_bl = bilinear_analysis(
            self.ens, mres=False,
            loadpath=f'no_mres/{self.ens}_bl_massive_mSMOM.p')

        all_masses = self.mSMOM_bl.all_masses
        N_masses = len(all_masses)
        self.Z_ax = stat(
            val=[float(m) for m in all_masses],
            err=np.zeros(len(all_masses)),
            btsp='fill'
        )

        Zs = np.array([self.mSMOM_bl.extrap_Z(
            mu=mu, masses=(m, m))['m']
            for m in all_masses])

        self.Z_mSMOM = stat(
            val=np.array([Zs[m].val for m in range(N_masses)]),
            err=np.array([Zs[m].err for m in range(N_masses)]),
            btsp=np.array([Zs[m].btsp for m in range(N_masses)])
        )

        self.m_C_mSMOM = stat(
            val=np.array([self.Z_mSMOM.val[m]*self.m_C.val
                          for m in range(N_masses)]),
            err='fill',
                btsp=np.array([[self.Z_mSMOM.btsp[k, m]*self.m_C.btsp[k]
                                for m in range(N_masses)]
                               for k in range(N_boot)])
        )

        am_bars = np.array([self.mSMOM_bl.extrap_Z(
            mu=mu, masses=(m, m))['mam_q']
            for m in all_masses])

        self.am_bars_mSMOM = stat(
            val=np.array([am_bars[m].val for m in range(N_masses)]),
            err=np.array([am_bars[m].err for m in range(N_masses)]),
            btsp=np.array([am_bars[m].btsp for m in range(N_masses)])
        )

        self.m_bars_mSMOM = stat(
            val=self.am_bars_mSMOM.val*self.ainv.val,
            err='fill',
            btsp=np.array([self.am_bars_mSMOM.btsp[
                k, :]*self.ainv.btsp[k]
                for k in range(N_boot)])
        )

    def calc_mSMOM(self, eta_star, mu=m_C_PDG, **kwargs):
        self.load_mSMOM(mu)
        self.eta_ax = self.Z_ax

        eta_star_ax = self.interpolate_eta_c(find_y=eta_star)

    def load_eta(self, **kwargs):
        self.valence = etaCvalence(self.ens)
        self.valence.toDict(keys=list(
            self.valence.mass_comb.keys()), mres=True)
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
        self.eta_y = stat(
            val=self.eta_ay.val*self.ainv.val,
            err='fill',
            btsp=np.array([self.eta_ay.btsp[k, :]*self.ainv.btsp[k]
                           for k in range(N_boot)])
        )

        self.am_C = self.interpolate_eta_c(
            find_y=eta_PDG, find_y_err=eta_PDG_err)
        self.m_C = stat(
            val=self.am_C.val*self.ainv.val,
            err='fill',
            btsp=np.array([self.am_C.btsp[k]*self.ainv.btsp[k]
                           for k in range(N_boot)])
        )

    def interpolate_eta_c(self, find_y, find_y_err=0, **kwargs):
        find_y = stat(
            val=find_y,
            err=find_y_err,
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
