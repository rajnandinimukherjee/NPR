import itertools
from scipy.interpolate import interp1d
from NPR_classes import *

#fit_file = 'Felix'
fit_file = 'Tobi'

all_bag_data = h5py.File(f'kaon_bag_fits_{fit_file}.h5', 'r')
bag_ensembles = [key for key in all_bag_data.keys() if key in UKQCD_ens]
ensemble_seeds = {ens: int(hash(ens)) % (2**32)
                  for ens in bag_ensembles+['C1M', 'M1M']}


def load_info(key, ens, ops=operators, meson='ls', **kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson == 'ls':
        central = np.array([np.array(h5_data[
            op][key]['central']).item()
            for op in ops])
        error = np.array([np.array(h5_data[
            op][key]['error']).item()
            for op in ops])
        bootstraps = np.array([np.array(h5_data[op][key][
            'Bootstraps'])[:, 0] for op in ops]).T

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

    return stat(val=central, err=error, btsp=bootstraps)


class Z_analysis:
    N_boot = N_boot
    filename = 'fourquarks_Z.h5'
    N_ops = len(operators)

    def __init__(self, ensemble, action=(0, 0),
                 norm='V', mask=fq_mask.copy(), **kwargs):
        self.ens = ensemble
        self.action = action
        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='seed', seed=ensemble_seeds[self.ens])
        self.a_sq = self.ainv**(-2)
        self.mask = mask

        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][0])
        self.masses = (self.sea_m, self.sea_m)
        self.f_pi = f_pi_PDG/self.ainv

        try:
            self.m_pi = load_info('m_0', self.ens, operators, meson='ll')
            if self.m_pi.N_boot != N_boot:
                self.m_pi = stat(val=self.m_pi.val, err=self.m_pi.err, btsp='fill')
            self.m_f_sq = stat(
                val=(self.m_pi.val/self.f_pi.val)**2,
                err='fill',
                btsp=(self.m_pi.btsp**2)/(self.f_pi.btsp**2)
            )
        except KeyError:
            if self.ens == 'C1M':
                self.m_pi = stat(
                    val=0.16079,
                    err=0.00054,
                    btsp='fill')
            elif self.ens == 'M1M':
                self.m_pi = stat(
                    val=0.12005,
                    err=0.00065,
                    btsp='fill',
                )
            else:
                print(f'{self.ens} m_pi data not identified,' +
                      ' check for consistent ensemble naming')

        self.norm = norm
        if norm == '11':
            self.mask[0, 0] = False
        self.load_fq_Z(norm=self.norm, **kwargs)

    def interpolate(self, m, xaxis='mu', ainv=None,
                    plot=False, fittype='linear',
                    filename='plots/Z_scaling.pdf',
                    **kwargs):
        if xaxis == 'mu':
            if ainv is None:
                x = self.am*self.ainv
            else:
                x = self.am*ainv
        elif xaxis == 'amu':
            x = self.am

        if plot:
            fig, ax = self.plot_Z(xaxis=xaxis, pass_plot=plot,
                                  filename=filename)

        matrix = np.zeros(shape=(self.N_ops, self.N_ops))
        errors = np.zeros(shape=(self.N_ops, self.N_ops))
        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=self.Z.val[:, i, j],
                    err=self.Z.err[:, i, j],
                    btsp=self.Z.btsp[:, :, i, j]
                )

                if fittype == 'linear':
                    indices = np.sort(
                        self.closest_n_points(m, x.val, n=2))
                    x_1, y_1 = x[indices[0]], y[indices[0]]
                    x_2, y_2 = x[indices[1]], y[indices[1]]
                    slope = (y_2-y_1)/(x_2-x_1)
                    intercept = y_1 - slope*x_1

                    pred = intercept + slope*m
                    x_grain = np.linspace(
                        x.val[indices[0]], x.val[indices[-1]], 20)
                    pred_grain = join_stats(
                        [intercept+slope*g for g in x_grain])

                elif fittype == 'quadratic':
                    indices = np.sort(
                        self.closest_n_points(m, x.val, n=3))
                    x_1, y_1 = x[indices[0]], y[indices[0]]
                    x_2, y_2 = x[indices[1]], y[indices[1]]
                    x_3, y_3 = x[indices[2]], y[indices[2]]

                    a = y_1/((x_1-x_2)*(x_1-x_3)) + y_2 / \
                        ((x_2-x_1)*(x_2-x_3)) + y_3/((x_3-x_1)*(x_3-x_2))

                    b = (-y_1*(x_2+x_3)/((x_1-x_2)*(x_1-x_3))
                         - y_2*(x_1+x_3)/((x_2-x_1)*(x_2-x_3))
                         - y_3*(x_1+x_2)/((x_3-x_1)*(x_3-x_2)))

                    c = (y_1*x_2*x_3/((x_1-x_2)*(x_1-x_3))
                         + y_2*x_1*x_3/((x_2-x_1)*(x_2-x_3))
                         + y_3*x_1*x_2/((x_3-x_1)*(x_3-x_2)))

                    pred = a*(m**2) + b*m + c
                    x_grain = np.linspace(
                        x.val[indices[0]], x.val[indices[-1]], 20)
                    pred_grain = join_stats(
                        [a*(g**2)+b*g + c for g in x_grain]
                    )

                else:
                    print("Choose fittype linear or quadratic.")
                    break

                matrix[i, j] = pred.val
                errors[i, j] = pred.err

                if plot:
                    ax[i, j].errorbar(m,
                                      matrix[i, j],
                                      yerr=errors[i, j],
                                      c='r', fmt='o',
                                      label=err_disp(
                                          pred.val,
                                          pred.err),
                                      capsize=4)
                    ax[i, j].fill_between(x_grain,
                                          pred_grain.val+pred_grain.err,
                                          pred_grain.val-pred_grain.err,
                                          color='r', alpha=0.1,
                                          label=fittype)
                    ax[i, j].legend()

        if plot:
            call_PDF(filename)

        if self.mask[0, 0] == False:
            matrix[0, 0] = 1

        Z = stat(
            val=matrix,
            err=errors,
            btsp='fill'
        )

        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
            Z = stat(
                val=rot_mtx@Z.val@np.linalg.inv(rot_mtx),
                err='fill',
                btsp=np.array([rot_mtx@Z.btsp[k,]@np.linalg.inv(rot_mtx)
                               for k in range(N_boot)])
            )

        return Z

    def scale_evolve(self, mu2, mu1, **kwargs):
        Z1 = self.interpolate(mu1, type='mu', **kwargs)
        Z2 = self.interpolate(mu2, **kwargs)
        sigma = stat(
            val=Z2.val@np.linalg.inv(Z1.val),
            err='fill',
            btsp=np.array([Z2.btsp[k,]@np.linalg.inv(Z1.btsp[k,])
                           for k in range(N_boot)]))
        return sigma

    def plot_Z(self, xaxis='mu', filename='plots/Z_scaling.pdf',
               pass_plot=False, **kwargs):
        x = self.am if xaxis == 'am' else (self.am*self.ainv)

        fig, ax = plt.subplots(nrows=self.N_ops, ncols=self.N_ops,
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=self.Z.val[:, i, j],
                    err=self.Z.err[:, i, j],
                    btsp=self.Z.btsp[:, :, i, j]
                )
                ax[i, j].errorbar(x.val, y.val,
                                  yerr=y.err, xerr=x.err,
                                  fmt='o', capsize=4)

                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
                if i == 1 or i == 3:
                    ax[i, j].set_xticks([])
            else:
                ax[i, j].axis('off')
        plt.suptitle(
            r'$Z_{ij}^{'+self.ens+r'}/Z_{'+self.norm+r'}$ vs renormalisation scale $\mu$', y=0.9)

        if pass_plot:
            return fig, ax
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

        fig, ax = plt.subplots(nrows=self.N_ops, ncols=self.N_ops,
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

        call_PDF(filename)

    def load_fq_Z(self, norm='A', resid_mask=True, **kwargs):
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

        if resid_mask:
            Z_mask = np.zeros(shape=Z_ij_Z_q_2.val.shape)
            Z_mask_boot = np.zeros(shape=Z_ij_Z_q_2.btsp.shape)
            try:
                for m in range(self.N_mom):
                    Lambda = (np.linalg.inv(Z_ij_Z_q_2.val[m])@fq_gamma_F).T
                    Lambda = fq_mask*Lambda
                    Z_mask[m,] = fq_gamma_F@np.linalg.inv(Lambda.T)
                    for k in range(N_boot):
                        Lambda_k = (np.linalg.inv(Z_ij_Z_q_2.btsp[k,m])@fq_gamma_F).T
                        Lambda_k = fq_mask*Lambda_k
                        Z_mask_boot[k,m,] = fq_gamma_F@np.linalg.inv(Lambda_k.T)
                Z_ij_Z_q_2 = stat(
                        val=Z_mask,
                        err='fill',
                        btsp=Z_mask_boot
                        )
            except np.linalg.LinAlgError:
                print(self.mask)
                raise


        bl_data = h5py.File('bilinear_Z_gamma.h5', 'r')[
            str(self.action)][self.ens][str(self.masses)]

        if norm == 'bag':
            Z_ij_bag = Z_ij_Z_q_2.val
            Z_ij_bag[:, :, 0] = np.array(
                    [Z_ij_bag[m, :, 0] *
                     (bl_data['A']['central'][m]**(-2))
                     for m in range(self.N_mom)])
            Z_ij_bag[:, :, 1:] = np.array(
                    [Z_ij_bag[m, :, 1:] *
                     (bl_data['S']['central'][m]**(-2))
                     for m in range(self.N_mom)])

            Z_ij_bag_btsp = Z_ij_Z_q_2.btsp
            Z_ij_bag_btsp[:, :, :, 0] = np.array([
                [Z_ij_bag_btsp[k, m, :, 0] *
                 (bl_data['A']['bootstrap']
                  [k, m]**(-2))
                 for m in range(self.N_mom)]
                for k in range(N_boot)])
            Z_ij_bag_btsp[:, :, :, 1:] = np.array([
                [Z_ij_bag_btsp[k, m, :, 1:] *
                 (bl_data['S']['bootstrap']
                  [k, m]**(-2))
                 for m in range(self.N_mom)]
                for k in range(N_boot)])

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
        elif norm == '11/AS':
            self.Z = stat(
                val=[(Z_ij_Z_q_2.val[m, :, :]/Z_ij_Z_q_2.val[m, 0, 0]) *
                     (bl_data['A']['central'][m]/bl_data['S']['central'][m])**2
                     for m in range(self.N_mom)],
                err='fill',
                btsp=[[(Z_ij_Z_q_2.btsp[k, m, :, :] /
                       Z_ij_Z_q_2.btsp[k, m, 0, 0]) *
                       (bl_data['A']['bootstrap'][k, m] /
                        bl_data['S']['bootstrap'][k, m])**2
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
        else:
            print('Normalisation not recognised!')

    def closest_n_points(self, target, values, n, **kwargs):
        diff = np.abs(np.array(values)-np.array(target))
        sort = np.sort(diff)
        closest_idx = []
        for n_idx in range(n):
            nth_closest_point = list(diff).index(sort[n_idx])
            closest_idx.append(nth_closest_point)
        return closest_idx



class bag_analysis:
    def __init__(self, ensemble, obj='bag', action=(0, 0), **kwargs):
        self.ens = ensemble
        self.action = action
        self.ra = ratio_analysis(self.ens)
        self.obj = obj
        if obj == 'bag':
            norm = 'bag'
            self.bag = self.ra.B_N
        elif obj == 'ratio':
            norm = '11'
            self.bag = self.ra.ratio
        elif obj == 'ratio2':
            norm = '11/AS'
            self.bag = self.ra.ratio2

        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='seed', seed=ensemble_seeds[self.ens])
        self.a_sq = self.ainv**(-2)
        self.ms_phys = stat(
            val=params[self.ens]['ams_phys'],
            err=params[self.ens]['ams_phys_err'],
            btsp='seed', seed=ensemble_seeds[self.ens])*self.ainv

        self.ms_sea = self.ainv*params[self.ens]['ams_sea']
        self.ms_diff = (self.ms_sea-self.ms_phys)/self.ms_phys

        self.f_pi = f_pi_PDG/self.ainv

        self.m_pi = load_info('m_0', self.ens, meson='ll')
        if self.m_pi.N_boot != N_boot:
            self.m_pi = stat(val=self.m_pi.val, err=self.m_pi.err, btsp='fill')
        self.m_f_sq = (self.m_pi**2)/(self.f_pi**2)

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
            return f

        func = param[0] + param[1]*a_sq +\
            (mpi_dep(m_f_sq)-mpi_dep(PDG))
        if 'addnl_terms' in kwargs:
            if kwargs['addnl_terms'] == 'a4':
                func += param[3]*(a_sq**2)
            elif kwargs['addnl_terms'] == 'del_ms':
                func += param[3]*ms_diff
            elif kwargs['addnl_terms'] == 'log':
                chir_log_coeffs = chiral_logs(obj=self.obj, **kwargs)
                chir_log_coeffs = chir_log_coeffs/((4*np.pi)**2)
                log_ratio = m_f_sq*(f_pi_PDG.val**2)/(Lambda_QCD**2)
                log_term = chir_log_coeffs[op_idx]*np.log(log_ratio)
                func += param[0]*log_term*m_f_sq

        return func


class ratio_analysis:
    def __init__(self, ens, action=(0, 0), **kwargs):
        self.ens = ens
        self.op_recon()

    def op_recon(self, **kwargs):
        self.m_P = load_info('m_0', self.ens)
        self.ZP_L_0 = load_info('pLL_0', self.ens)
        self.ZA_L_0 = load_info('aLL_0', self.ens)
        if fit_file == 'Tobi':
            self.ZP_L_0 = self.ZP_L_0*(self.m_P)**0.5
            self.ZA_L_0 = self.ZA_L_0*(self.m_P)**0.5

        self.gr_O_gr = load_info('gr-O-gr', self.ens)
        self.N_boot = self.gr_O_gr.N_boot

        self.Ni = norm_factors()
        B1 = stat(
            val=self.gr_O_gr.val[0]/(
                self.Ni[0]*self.ZA_L_0.val[0]**2),
            err='fill',
            btsp=np.array([self.gr_O_gr.btsp[k, 0]/(
                self.Ni[0]*self.ZA_L_0.btsp[k, 0]**2)
                for k in range(self.N_boot)])
        )

        B2_5 = [
            stat(
                val=self.gr_O_gr.val[i]/(
                    self.Ni[i]*self.ZP_L_0.val[i]**2),
                err='fill',
                btsp=np.array([self.gr_O_gr.btsp[k, i]/(
                    self.Ni[i]*self.ZP_L_0.btsp[k, i]**2)
                    for k in range(self.N_boot)]))
            for i in range(1, len(operators))]

        Bs = [B1]+B2_5
        self.bag = stat(
            val=[B.val for B in Bs],
            err=[B.err for B in Bs],
            btsp=np.array([[B.btsp[k] for B in Bs]
                           for k in range(self.N_boot)])
        )
        Ni_diag = stat(
            val=np.diag(self.Ni),
            err='fill',
            btsp=np.array([np.diag(self.Ni) for k in range(self.N_boot)])
        )
        self.B_N = Ni_diag@self.bag
        self.ratio = self.gr_O_gr/self.gr_O_gr[0]
        if self.B_N.N_boot != N_boot:
            self.B_N = stat(
                    val=self.B_N.val,
                    err=self.B_N.err,
                    btsp='fill'
                    )
            self.ratio = stat(
                    val=self.ratio.val,
                    err=self.ratio.err,
                    btsp='fill'
                    )
        try:
            self.f_P = load_info('f_M', self.ens)
            K_exp_ratio = (m_K_PDG/f_K_PDG)**2
            P_lat_ratio = (self.m_P/self.f_P)**2
            self.ratio2 = P_lat_ratio*self.ratio/K_exp_ratio
        except KeyError:
            pass
