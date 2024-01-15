import itertools
from scipy.interpolate import interp1d
from NPR_classes import *

all_bag_data = h5py.File('kaon_bag_fits.h5', 'r')
bag_ensembles = [key for key in all_bag_data.keys() if key in UKQCD_ens]


def load_info(key, ens, ops=operators, meson='ls', **kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson == 'ls':
        central = np.array([np.array(h5_data[
            op][key]['central']).item()
            for op in ops])
        error = np.array([np.array(h5_data[
            op][key]['error']).item()
            for op in ops])
        bootstraps = np.zeros(shape=(N_boot, len(operators)))
        for i in range(len(operators)):
            op = ops[i]
            bootstraps[:, i] = np.array(h5_data[op][key][
                'Bootstraps'])[:, 0]

    elif meson == 'll':
        central = np.array(h5_data[key]['central']).item()
        error = np.array(h5_data[key]['error']).item()
        bootstraps = np.array(h5_data[key]['Bootstraps'])[:, 0]

    if key == 'gr-O-gr':
        central[1] *= -1
        bootstraps[:, 1] = -bootstraps[:, 1]
        central[3] *= -1
        bootstraps[:, 3] = -bootstraps[:, 3]
        central[4] *= -1
        bootstraps[:, 4] = -bootstraps[:, 4]

    return stat(val=central, err='fill', btsp=bootstraps)


class Z_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP', 'TT']
    N_boot = N_boot
    filename = 'fourquarks_Z.h5'
    mask = mask

    def __init__(self, ensemble, action=(0, 0), norm='V', **kwargs):

        self.ens = ensemble
        self.action = action
        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='fill')
        self.a_sq = stat(
            val=self.ainv.val**(-2),
            err='fill',
            btsp=self.ainv.btsp**(-2)
        )
        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][0])
        self.masses = (self.sea_m, self.sea_m)
        self.f_pi = stat(
            val=f_pi_PDG.val/self.ainv.val,
            btsp=f_pi_PDG.btsp/self.ainv.btsp)

        try:
            self.m_pi = load_info('m_0', self.ens, operators, meson='ll')
            self.m_f_sq = stat(
                val=(self.m_pi.val/self.f_pi.val)**2,
                err='fill',
                btsp=(self.m_pi.btsp**2)/(self.f_pi.btsp**2)
            )
        except KeyError:
            print('m_pi data not identified, check for consistent ensemble naming')

        self.norm = norm
        if norm == '11':
            self.mask[0, 0] = False
        self.load_fq_Z(norm=self.norm)

    def interpolate(self, m, xaxis='mu', ainv=None,
                    plot=False, **kwargs):
        if xaxis == 'mu':
            if ainv is None:
                x = self.am*self.ainv
            else:
                x = self.am*ainv
        elif xaxis == 'amu':
            x = self.am

        matrix = np.zeros(shape=(5, 5))
        btsp = np.zeros(shape=(N_boot, 5, 5))
        for i, j in itertools.product(range(5), range(5)):
            if self.mask[i, j]:
                f = interp1d(x.val, self.Z.val[:, i, j],
                             fill_value='extrapolate')
                matrix[i, j] = f(m)

                for k in range(self.N_boot):
                    f = interp1d(x.btsp[k,], self.Z.btsp[k, :, i, j],
                                 fill_value='extrapolate')
                    btsp[k, i, j] = f(m)
        Z = stat(
            val=matrix,
            err='fill',
            btsp=btsp
        )
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
            Z = stat(
                val=rot_mtx@Z.val@np.linalg.inv(rot_mtx),
                err='fill',
                btsp=np.array([rot_mtx@Z.btsp[k,]@np.linalg.inv(rot_mtx)
                               for k in range(N_boot)])
            )

        if plot:
            fig, ax, filename = self.plot_Z(xaxis=xaxis, pass_plot=True)
            N_ops = len(operators)
            for i, j in itertools.product(range(N_ops), range(N_ops)):
                if self.mask[i, j]:
                    ax[i, j].errorbar(m, Z.val[i, j],
                                      yerr=Z.err[i, j],
                                      c='k', fmt='o', capsize=4)
            call_PDF(filename)

        return Z

    def scale_evolve(self, mu2, mu1, **kwargs):
        Z1 = self.interpolate(mu1, type='mu', **kwargs)
        Z2 = self.interpolate(mu2, **kwargs)
        sigma = stat(
            val=self.mask*Z2.val@np.linalg.inv(Z1.val),
            err='fill',
            btsp=np.array([self.mask*Z2.btsp[k,]@np.linalg.inv(Z1.btsp[k,])
                           for k in range(N_boot)]))
        return sigma

    def plot_Z(self, xaxis='mu', filename='plots/Z_scaling.pdf',
               pass_plot=False, **kwargs):
        x = self.am.val if xaxis == 'am' else (self.am*self.ainv).val
        xerr = self.am.err if xaxis == 'am' else (self.am*self.ainv).err

        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y = self.Z.val[:, i, j]
                yerr = self.Z.err[:, i, j]
                ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr,
                                  fmt='o', capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis('off')
        plt.suptitle(
            r'$Z_{ij}^{'+self.ens+r'}/Z_{'+self.norm+r'}^2$ vs renormalisation scale $\mu$', y=0.9)

        if pass_plot:
            return fig, ax, filename
        else:
            call_PDF(filename)
            print(f'Saved plot to {filename}.')

    def plot_sigma(self, xaxis='mu', mu1=None, mu2=3.0,
                   filename='plots/Z_running.pdf', **kwargs):
        x = self.am.val if xaxis == 'am' else (self.am*self.ainv).val
        xerr = self.am.err if xaxis == 'am' else (self.am*self.ainv).err

        if mu1 == None:
            sigmas = [self.scale_evolve(mu2, mom)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{'+self.ens + \
                r'}('+str(mu2)+r'\leftarrow\mu)$'
        else:
            sigmas = [self.scale_evolve(mom, mu1)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{'+self.ens + \
                r'}(\mu\leftarrow '+str(mu1)+r')$'

        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y = [sig.val[i, j] for sig in sigmas]
                yerr = [sig.err[i, j] for sig in sigmas]

                ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr,
                                  fmt='o', capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis('off')
        if self.bag:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_{A/P}^2$', y=0.9)
        else:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_A^2$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)

    def load_fq_Z(self, norm='A', **kwargs):
        fq_data = h5py.File(self.filename, 'r')[
            str(self.action)][self.ens][str(self.masses)]
        self.am = stat(
            val=fq_data['ap'][:],
            err=np.zeros(len(fq_data['ap'][:])),
            btsp='fill'
        )
        self.N_mom = len(self.am.val)
        Z_ij_Z_q_2 = stat(
            val=(fq_data['central'][:]).real,
            err=fq_data['errors'][:],
            btsp=(fq_data['bootstrap'][:]).real
        )
        bl_data = h5py.File('bilinear_Z_gamma.h5', 'r')[
            str(self.action)][self.ens][str(self.masses)]

        if norm == 'bag':
            Z_ij_bag = Z_ij_Z_q_2.val
            Z_ij_bag[:, 0, 0] = np.array(
                    [Z_ij_bag[m, 0, 0] *
                     (bl_data['A']['central'][m]**(-2))
                     for m in range(self.N_mom)])
            Z_ij_bag[:, 1:, 1:] = np.array(
                    [Z_ij_bag[m, 1:, 1:] *
                     (bl_data['S']['central'][m]**(-2))
                     for m in range(self.N_mom)])

            Z_ij_bag_btsp = Z_ij_Z_q_2.btsp
            for k in range(N_boot):
                Z_ij_bag_btsp[k, :, 0, 0] = np.array(
                    [Z_ij_bag_btsp[k, m, 0, 0] *
                     (bl_data['A']['bootstrap']
                      [k, m]**(-2))
                     for m in range(self.N_mom)])
                Z_ij_bag_btsp[k, :, 1:, 1:] = np.array(
                    [Z_ij_bag_btsp[k, m, 1:, 1:] *
                     (bl_data['S']['bootstrap']
                      [k, m]**(-2))
                     for m in range(self.N_mom)])
            self.Z = stat(
                val=Z_ij_bag,
                err='fill',
                btsp=Z_ij_bag_btsp
            )

        elif norm == '11':
            self.Z = stat(
                val=[Z_ij_Z_q_2.val[m, :, :]/Z_ij_Z_q_2.val[m, 0, 0]
                     for m in range(self.N_mom)],
                err='fill',
                    btsp=[[Z_ij_Z_q_2.btsp[k, m, :, :]/Z_ij_Z_q_2.btsp[k, m, 0, 0]
                           for m in range(self.N_mom)]
                          for k in range(N_boot)]
            )
        elif norm in bilinear.currents:
            Z_bl_Z_q = stat(
                val=bl_data[norm]['central'][:],
                err=bl_data[norm]['errors'][:],
                btsp=bl_data[norm]['bootstrap'][:]
            )
            self.Z = stat(
                val=[Z_ij_Z_q_2.val[m, :, :]*(Z_bl_Z_q.val[m]**(-2))
                     for m in range(self.N_mom)],
                err='fill',
                btsp=np.array([[Z_ij_Z_q_2.btsp[k, m, :, :] *
                               (Z_bl_Z_q.btsp[k, m]**(-2))
                               for m in range(self.N_mom)]
                               for k in range(N_boot)])
            )


class bag_analysis:
    mask = mask

    def __init__(self, ensemble, obj='bag', action=(0, 0), **kwargs):

        self.ens = ensemble
        self.action = action
        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='fill')
        self.a_sq = stat(
            val=self.ainv.val**(-2),
            err='fill',
            btsp=self.ainv.btsp**(-2)
        )
        self.ms_phys = stat(
            val=params[self.ens]['ams_phys'],
            err=params[self.ens]['ams_phys_err'],
            btsp='fill'
        )*self.ainv
        self.ms_sea = self.ainv*params[self.ens]['ams_sea']
        self.ms_diff = (self.ms_sea-self.ms_phys)/self.ms_phys
        self.ra = ratio_analysis(self.ens)
        if obj == 'bag':
            self.bag = self.ra.B_N
        elif obj == 'ratio':
            self.bag = self.ra.ratio

        self.f_pi = stat(
            val=f_pi_PDG.val/self.ainv.val,
            err='fill',
            btsp=f_pi_PDG.btsp/self.ainv.btsp)

        self.m_pi = load_info('m_0', self.ens, meson='ll')
        self.m_f_sq = stat(
            val=(self.m_pi.val/self.f_pi.val)**2,
            err='fill',
            btsp=(self.m_pi.btsp**2)/(self.f_pi.btsp**2)
        )

        ens = self.ens
        if obj == 'bag':
            norm = 'bag'
        elif obj == 'ratio':
            norm = '11'
            self.mask[0, 0] = False
        self.Z_info = Z_analysis(ens, norm=norm)

    def interpolate(self, mu, rotate=np.eye(len(operators)), **kwargs):
        Z_mu = self.Z_info.interpolate(mu, rotate=rotate, **kwargs)
        rot_mtx = stat(
            val=rotate,
            btsp='fill'
        )

        bag = rot_mtx@self.bag
        bag_interp = Z_mu@bag
        return bag_interp

    def ansatz(self, param, operator, fit='central', **kwargs):
        op_idx = operators.index(operator)
        if fit == 'central':
            a_sq = self.a_sq.val
            m_f_sq = self.m_f_sq.val
            PDG = m_f_sq_PDG.val
            ms_diff = self.ms_diff.val
        else:
            k = kwargs['k']
            a_sq = self.a_sq.btsp[k]
            m_f_sq = self.m_f_sq.btsp[k]
            PDG = m_f_sq_PDG.btsp[k]
            ms_diff = self.ms_diff.btsp[k]

        def mpi_dep(m_f_sq):
            f = param[2]*m_f_sq
            if 'addnl_terms' in kwargs:
                if kwargs['addnl_terms'] == 'm4':
                    f += param[3]*(m_f_sq**2)
                elif kwargs['addnl_terms'] == 'log':
                    chir_log_coeffs = np.array([-0.5, -0.5, -0.5, 0.5, 0.5])
                    if 'rotate' in kwargs:
                        chir_log_coeffs = kwargs['rotate']@chir_log_coeffs
                    chir_log_coeffs = chir_log_coeffs/((4*np.pi)**2)
                    Lambda_QCD = 1.0
                    log_ratio = m_f_sq*(f_pi_PDG.val**2)/(Lambda_QCD**2)
                    log_term = chir_log_coeffs[op_idx]*np.log(log_ratio)
                    f += log_term*m_f_sq
            return f

        func = param[0] + param[1]*a_sq +\
            (mpi_dep(m_f_sq)-mpi_dep(PDG))
        if 'addnl_terms' in kwargs:
            if kwargs['addnl_terms'] == 'a4':
                func += param[3]*(a_sq**2)
            elif kwargs['addnl_terms'] == 'del_ms':
                func += param[3]*ms_diff

        return func


class ratio_analysis:

    def __init__(self, ens, action=(0, 0), **kwargs):
        self.ens = ens
        self.op_recon()

    def op_recon(self, **kwargs):
        self.ZP_L_0 = load_info('pLL_0', self.ens)
        self.ZA_L_0 = load_info('aLL_0', self.ens)

        self.gr_O_gr = load_info('gr-O-gr', self.ens)
        gr1 = stat(
            val=self.gr_O_gr.val[0],
            err=self.gr_O_gr.err[0],
            btsp=self.gr_O_gr.btsp[:, 0]
        )
        self.ratio = self.gr_O_gr/gr1

        self.Ni = norm_factors()
        B1 = stat(
            val=self.gr_O_gr.val[0]/(
                self.Ni[0]*self.ZA_L_0.val[0]**2),
            err='fill',
            btsp=np.array([self.gr_O_gr.btsp[k, 0]/(
                self.Ni[0]*self.ZA_L_0.btsp[k, 0]**2)
                for k in range(N_boot)])
        )

        B2_5 = [
            stat(
                val=self.gr_O_gr.val[i]/(
                    self.Ni[i]*self.ZP_L_0.val[i]**2),
                err='fill',
                btsp=np.array([self.gr_O_gr.btsp[k, i]/(
                    self.Ni[i]*self.ZP_L_0.btsp[k, i]**2)
                    for k in range(N_boot)]))
            for i in range(1, len(operators))]

        Bs = [B1]+B2_5
        self.bag = stat(
            val=[B.val for B in Bs],
            err=[B.err for B in Bs],
            btsp=np.array([[B.btsp[k] for B in Bs]
                           for k in range(N_boot)])
        )
        Ni_diag = stat(
            val=np.diag(self.Ni),
            btsp='fill'
        )
        self.B_N = Ni_diag@self.bag
