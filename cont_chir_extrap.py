import pdb
from coeffs import *
from bag_param_renorm import *

cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(bag_ensembles)), cmap.N)


def mpi_dep(params, m_f_sq, op_idx, **kwargs):
    f = params[2]*m_f_sq
    if 'addnl_terms' in kwargs:
        if kwargs['addnl_terms'] == 'm4':
            f += params[3]*(m_f_sq**2)
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


def chiral_continuum_ansatz(params, a_sq, m_f_sq, PDG, operator, **kwargs):
    op_idx = operators.index(operator)
    func = params[0] + params[1]*a_sq + \
        mpi_dep(params, m_f_sq, op_idx, **kwargs) - \
        mpi_dep(params, PDG, op_idx, **kwargs)
    if 'addnl_terms' in kwargs:
        if kwargs['addnl_terms'] == 'a4':
            func += params[3]*(a_sq**2)

    return func


class Z_fits:

    chiral_ens = [ens for ens in bag_ensembles if ens[0] == 'M']
    cont_ens = ['F1M', 'M0']
    mask = mask

    def __init__(self, ens_list, norm='V', **kwargs):
        self.ens_list = ens_list
        self.norm = norm
        if norm == '11':
            self.mask[0, 0] = False
        self.Z_dict = {e: Z_analysis(e, norm=self.norm)
                       for e in self.ens_list}
        self.colors = {list(self.Z_dict.keys())[k]: list(
            mc.TABLEAU_COLORS.keys())[k]
            for k in range(len(self.ens_list))}

    def chiral_ansatz(self, params, ens_list, fit='central', **kwargs):
        if fit == 'central':
            m_f_sq = np.array([self.Z_dict[e].m_f_sq.val
                              for e in ens_list])
            PDG = m_f_sq_PDG.val
        else:
            k = kwargs['k']
            m_f_sq = np.array([self.Z_dict[e].m_f_sq.btsp[k]
                              for e in ens_list])
            PDG = m_f_sq_PDG.btsp[k]

        return params[0] + params[1]*(m_f_sq-PDG)

    def continuum_ansatz(self, params, ens_list, fit='central', **kwargs):
        if fit == 'central':
            a_sq = np.array([self.Z_dict[e].a_sq.val for e in ens_list])
        else:
            k = kwargs['k']
            a_sq = np.array([self.Z_dict[e].a_sq.btsp[k] for e in ens_list])

        return params[0] + params[1]*a_sq

    def extrap_ij(self, mu2, mu1, i, j, guess=[1, 1e-1],
                  include_C=False, **kwargs):

        # ==== Step 1: chiral extrapolation of M ensembles=================
        sigma = stat(
            val=np.zeros(len(self.chiral_ens)),
            err=np.zeros(len(self.chiral_ens)),
            btsp=np.zeros(shape=(N_boot, len(self.chiral_ens)))
        )
        for e_idx, e in enumerate(self.chiral_ens):
            sig = self.Z_dict[e].scale_evolve(mu2, mu1, **kwargs)
            sigma.val[e_idx] = sig.val[i, j]
            sigma.err[e_idx] = sig.err[i, j]
            sigma.btsp[:, e_idx] = sig.btsp[:, i, j]

        COV = np.diag(sigma.err**2)
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def diff(sigma, params, **kwargs):
            return np.array(sigma) - self.chiral_ansatz(
                params, self.chiral_ens, **kwargs)

        def LD(params):
            return L.dot(diff(sigma.val, params, fit='central'))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))

        res_btsp = np.zeros(shape=(N_boot, len(res.x)))
        for k in range(N_boot):
            def LD_btsp(params):
                return L.dot(diff(sigma.btsp[k,], params,
                                  fit='btsp', k=k))
            res_k = least_squares(LD_btsp, guess,
                                  ftol=1e-10, gtol=1e-10)
            res_btsp[k, :] = res_k.x

        chiral_params = stat(
            val=res.x,
            err='fill',
            btsp=res_btsp
        )
        chiral_params.res = res

        # ==== Step 2: apply chiral slope to F1M to get "F0"=================
        sigma_F1 = self.Z_dict['F1M'].scale_evolve(mu2, mu1, **kwargs)
        m_f_sq_F1 = self.Z_dict['F1M'].m_f_sq
        sigma_F0 = stat(
            val=sigma_F1.val[i, j]-chiral_params.val[1] *
            (m_f_sq_F1.val-m_f_sq_PDG.val),
            err='fill',
            btsp=[sigma_F1.btsp[k, i, j]-chiral_params.btsp[k, 1]
                  * (m_f_sq_F1.btsp[k]-m_f_sq_PDG.btsp[k])
                  for k in range(N_boot)]
        )

        sigma_M0 = self.Z_dict['M0'].scale_evolve(mu2, mu1, **kwargs)
        sigma_M0 = stat(
            val=sigma_M0.val[i, j],
            err=sigma_M0.err[i, j],
            btsp=sigma_M0.btsp[:, i, j]
        )

        phys_sigma = [sigma_F0, sigma_M0]
        cont_ens = self.cont_ens.copy()
        if include_C:
            sigma_C0 = self.Z_dict['C0'].scale_evolve(mu2, mu1, **kwargs)
            sigma_C0 = stat(
                val=sigma_C0.val[i, j],
                err=sigma_C0.err[i, j],
                btsp=sigma_C0.btsp[:, i, j]
            )
            phys_sigma.append(sigma_C0)
            cont_ens.append('C0')

        # ==== Step 3: continuum extrapolate F0 and M0 (and maybe C0)=======================
        sigma = stat(
            val=[sig.val for sig in phys_sigma],
            err='fill',
            btsp=np.array([[sig.btsp[k] for sig in phys_sigma]
                           for k in range(N_boot)])
        )

        COV = np.diag(sigma.err**2)
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def diff(sigma, params, **kwargs):
            return np.array(sigma) - self.continuum_ansatz(
                params, cont_ens, **kwargs)

        def LD(params):
            return L.dot(diff(sigma.val, params, fit='central'))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))

        res_btsp = np.zeros(shape=(N_boot, len(res.x)))
        for k in range(N_boot):
            def LD_btsp(params):
                return L.dot(diff(sigma.btsp[k,], params,
                                  fit='btsp', k=k))
            res_k = least_squares(LD_btsp, guess,
                                  ftol=1e-10, gtol=1e-10)
            res_btsp[k, :] = res_k.x

        cont_params = stat(
            val=res.x,
            err='fill',
            btsp=res_btsp
        )
        cont_params.res = res

        return chiral_params, cont_params

    def extrap_sigma(self, mu2, mu1, **kwargs):

        N_ops = len(operators)
        sigma = np.zeros(shape=(N_ops, N_ops))
        sigma_err = np.zeros(shape=(N_ops, N_ops))
        sigma_btsp = np.zeros(shape=(N_boot, N_ops, N_ops))
        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                chiral_params, cont_params = self.extrap_ij(
                    mu2, mu1, i, j, **kwargs)
                sigma[i, j] = cont_params.val[0]
                sigma_err[i, j] = cont_params.err[0]
                sigma_btsp[:, i, j] = cont_params.btsp[:, 0]

        return stat(val=sigma, err=sigma_err, btsp=sigma_btsp)

    def plot_extrap(self, mu2, mu1, filename='plots/sigma_fits.pdf',
                    **kwargs):
        sigmas = {e: self.Z_dict[e].scale_evolve(mu2, mu1, **kwargs)
                  for e in self.ens_list}
        a_sqs = {e: self.Z_dict[e].a_sq for e in self.ens_list}
        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                chiral_params, cont_params = self.extrap_ij(
                    mu2, mu1, i, j, **kwargs)
                for e in self.ens_list:

                    ax[i, j].errorbar([a_sqs[e].val], [sigmas[e].val[i, j]],
                                      yerr=[sigmas[e].err[i, j]],
                                      fmt='o', capsize=4, c=self.colors[e])
                    ax[i, j].errorbar([0], [cont_params.val[0]],
                                      yerr=[cont_params.err[0]],
                                      fmt='o', capsize=4, c='k')

                    xmin, xmax = ax[i, j].get_xlim()
                    a_sq_grain = np.linspace(0, xmax, 100)
                    fit = stat(
                        val=cont_params.val[0] +
                        cont_params.val[1]*a_sq_grain,
                        err='fill',
                        btsp=np.array(
                            [cont_params.btsp[k, 0]+cont_params.btsp[k, 1] *
                             a_sq_grain for k in range(N_boot)])
                    )
                    ax[i, j].plot(a_sq_grain, fit.val, c='k')
                    ax[i, j].set_xlim([-xmax/50, xmax])
            else:
                ax[i, j].axis('off')
        plt.suptitle(r'$\sigma_{npt}('+str(np.around(mu2, 2)) +
                     ','+str(np.around(mu1, 2))+r')$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)

    def plot_sigma(self, mu1=None, mu2=3.0,
                   mom=np.linspace(2.0, 3.0, 10),
                   filename='plots/extrap_running.pdf',
                   **kwargs):
        x = mom

        if mu1 == None:
            sigmas = [self.extrap_sigma(mu2, mom, **kwargs)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{phys}('+str(mu2)+r'\leftarrow\mu)$'
        else:
            sigmas = [self.extrap_sigma(mom, mu1, **kwargs)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{phys}(\mu\leftarrow '+str(mu1)+r')$'

        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y = [sig.val[i, j] for sig in sigmas]
                yerr = [sig.err[i, j] for sig in sigmas]

                ax[i, j].errorbar(x, y, yerr=yerr,
                                  fmt='o', capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis('off')
        if self.norm == 'bag':
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_{A/P}^2$', y=0.9)
        elif self.norm in bilinear.currents:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_'+self.norm+r'^2$', y=0.9)
        else:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/'+self.norm+r'$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)

    def Z_chiral_extrap(self, mu, ens_list, **kwargs):
        chiral_data = [self.Z_dict[ens].interpolate(mu, **kwargs)
                       for ens in ens_list]
        m_sq = join_stats([self.Z_dict[ens].m_pi**2
                           for ens in ens_list])
        x = m_sq.val

        def ansatz(params, msq):
            return params[0] + params[1]*msq

        def diff(x, y, params):
            return y - ansatz(params, x)

        N_ops = len(operators)
        extrap = np.zeros(shape=(N_ops, N_ops))
        extrap_btsp = np.zeros(shape=(N_boot, N_ops, N_ops))
        extrap_map = np.empty(shape=(N_ops, N_ops), dtype=object)

        param_save = np.zeros(shape=(2, N_ops, N_ops))
        param_save_btsp = np.zeros(shape=(N_boot, 2, N_ops, N_ops))
        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y = np.array([cd.val[i, j] for cd in chiral_data])
                y_err = np.array([cd.err[i, j]
                                  for cd in chiral_data])
                COV = np.diag(y_err**2)
                L_inv = np.linalg.cholesky(COV)
                L = np.linalg.inv(L_inv)

                # ==central fit====
                def LD(params):
                    return L.dot(diff(x, y, params))
                res = least_squares(LD, [1, 0.1], ftol=1e-10, gtol=1e-10)
                param_save[:, i, j] = res.x

                # ==btsp fit======
                for k in range(N_boot):
                    x_k = m_sq.btsp[k]
                    y_k = np.array([cd.btsp[k, i, j]
                                    for cd in chiral_data])

                    def LD_k(params):
                        return L.dot(diff(x_k, y_k, params))

                    res_k = least_squares(
                        LD_k, [1, 0.1], ftol=1e-10, gtol=1e-10)
                    param_save_btsp[k, :, i, j] = res_k.x

                extrap[i, j] = ansatz(param_save[:, i, j], 0.0)
                extrap_btsp[:, i, j] = np.array([ansatz(
                    param_save_btsp[k, :, i, j], 0.0)
                    for k in range(N_boot)])

        fit_params = stat(
            val=param_save,
            err='fill',
            btsp=param_save_btsp
        )

        def mapping(i, j, fit_params, m_sq):
            if type(m_sq) is not stat:
                m_sq = stat(
                    val=m_sq,
                    btsp='fill'
                )
            Z_map = stat(
                val=ansatz(fit_params.val[:, i, j], m_sq.val),
                err='fill',
                btsp=np.array([ansatz(
                    fit_params.btsp[k, :, i, j], m_sq.btsp[k])
                    for k in range(N_boot)])
            )
            return Z_map

        return stat(val=extrap, err='fill', btsp=extrap_btsp), mapping, fit_params

    def Z_chiral_extrap_phys_handler(self, mu, exclude_phys=True,
                                     passonly=False, **kwargs):
        ainv = None
        ens_list = self.ens_list.copy()
        try:
            phys_idx = [index for index, ens in enumerate(
                self.ens_list) if '0' in ens][0]
            phys_Z = self.Z_dict[self.ens_list[0]].interpolate(mu, **kwargs)
            if exclude_phys:
                ens_list.pop(phys_idx)
                ainv = self.Z_dict[self.ens_list[-1]].ainv
                if not passonly:
                    print(
                        f'Excluding {self.ens_list[phys_idx]} from chiral fit.')
            extrap, mapping, fit_params = self.Z_chiral_extrap(
                mu, ens_list, **kwargs)
            chiral_data = [self.Z_dict[ens].interpolate(
                mu, ainv=ainv, **kwargs) for ens in self.ens_list]

            N_ops = len(operators)
            expanded_errors = np.zeros(shape=(N_ops, N_ops))
            for i, j in itertools.product(range(N_ops), range(N_ops)):
                if self.mask[i, j]:
                    y_chiral = mapping(i, j, fit_params, 0.0)
                    y = np.array([cd.val[i, j] for cd in chiral_data])
                    yerr = np.array([cd.err[i, j] for cd in chiral_data])
                    y_top = np.max([y_chiral.val+y_chiral.err,
                                    y[phys_idx]+yerr[phys_idx],
                                    phys_Z.val[i, j]+phys_Z.err[i, j]])
                    y_bot = np.min([y_chiral.val-y_chiral.err,
                                    y[phys_idx]-yerr[phys_idx],
                                    phys_Z.val[i, j]-phys_Z.err[i, j]])
                    expanded_errors[i, j] = np.max([np.abs(
                        y_top-y_chiral.val), np.abs(
                            y_chiral.val-y_bot)])
            return extrap, mapping, fit_params, chiral_data, phys_idx, phys_Z, expanded_errors

        except IndexError:
            print('No physical point ensembles included.')
            extrap, mapping, fit_params = self.Z_chiral_extrap(
                mu, ens_list, **kwargs)
            chiral_data = [self.Z_dict[ens].interpolate(
                mu, ainv=ainv, **kwargs) for ens in self.ens_list]
            return extrap, mapping, fit_params, chiral_data, None

    def Z_chiral_extrap_plot(self, mu, normalise=False,
                             filename='plots/Z_chiral_extrap.pdf',
                             **kwargs):

        quantities = self.Z_chiral_extrap_phys_handler(mu, **kwargs)
        extrap, mapping, fit_params, chiral_data, phys_idx = quantities[:5]
        if phys_idx != None:
            phys_Z, expanded_errors = quantities[5:]

        m_sq = join_stats([self.Z_dict[ens].m_pi**2
                           for ens in self.ens_list])
        x = m_sq.val
        xerr = m_sq.err

        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y_chiral = mapping(i, j, fit_params, 0.0)
                divider = y_chiral.val if normalise else 1
                y = np.array([cd.val[i, j] for cd in chiral_data])
                yerr = np.array([cd.err[i, j] for cd in chiral_data])

                label = r'$a^{-1}('+self.ens_list[-1]+')$'
                ax[i, j].errorbar(x, y/divider,
                                  yerr=yerr/divider,
                                  xerr=xerr, fmt='o',
                                  capsize=4, label=label)

                xmin, xmax = ax[i, j].get_xlim()
                msq_grain = np.linspace(0.0, xmax, 50)
                y_grain = mapping(i, j, fit_params, msq_grain)
                ax[i, j].fill_between(msq_grain,
                                      (y_grain.val-y_grain.err)/divider,
                                      (y_grain.val+y_grain.err)/divider,
                                      alpha=0.1, color='k')
                ax[i, j].axvline(0.0, c='k', linestyle='dashed', alpha=0.1)
                ax[i, j].errorbar([0.0], y_chiral.val/divider,
                                  yerr=y_chiral.err/divider,
                                  color='0.1', fmt='o', capsize=4)
                if phys_idx != None:
                    ax[i, j].errorbar([m_sq.val[phys_idx]],
                                      phys_Z.val[i, j]/divider,
                                      yerr=phys_Z.err[i, j]/divider,
                                      xerr=[m_sq.err[phys_idx]],
                                      color='r', fmt='o', capsize=4)

                    ax[i, j].errorbar([m_sq.val[phys_idx]/2],
                                      y_chiral.val/divider,
                                      yerr=expanded_errors[i, j]/divider,
                                      color='k', fmt='o', capsize=4)

                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
                if i == 1 or i == 3:
                    ax[i, j].set_xticks([])
            else:
                ax[i, j].axis('off')
        plt.suptitle(r'$Z_{ij}^{'+','.join(self.ens_list) +
                     '}(\mu='+str(mu)+'$ GeV$)/Z_{'+self.norm+r'}$ vs $(am_\pi)^2$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)


M_ens = ['M0', 'M1', 'M2', 'M3']
M = Z_fits(M_ens[1:], norm='bag')
C_ens = ['C0', 'C1', 'C2']
C = Z_fits(C_ens[1:], norm='bag')


class bag_fits:
    mask = mask
    operators = operators

    def __init__(self, ens_list, obj='bag', **kwargs):
        self.ens_list = ens_list
        self.obj = obj
        if self.obj == 'ratio':
            self.operators = self.operators[1:]
        self.bag_dict = {e: bag_analysis(e, obj=obj)
                         for e in self.ens_list}
        self.colors = {list(self.bag_dict.keys())[k]: list(
            mc.TABLEAU_COLORS.keys())[k]
            for k in range(len(self.ens_list))}

    def load_bag(self, mu, ens_list, chiral_extrap=False,
                 expanded_extrap=False, **kwargs):
        if not chiral_extrap:
            bags = [self.bag_dict[e].interpolate(mu, **kwargs)
                    for e in ens_list]
        else:
            M_quantities = M.Z_chiral_extrap_phys_handler(
                mu, passonly=True, chiral_extrap=chiral_extrap,
                expanded_extrap=expanded_extrap, **kwargs)
            M_extrap, M_mapping, M_fit_params = M_quantities[:3]

            C_quantities = C.Z_chiral_extrap_phys_handler(
                mu, passonly=True, chiral_extrap=chiral_extrap,
                expanded_extrap=expanded_extrap, **kwargs)
            C_extrap, C_mapping, C_fit_params = C_quantities[:3]

            if expanded_extrap and 'C0' in ens_list and 'M0' in ens_list:
                print('Using expanded extrap')
                M_expanded_errs = M_quantities[-1]
                M_extrap = stat(
                    val=M_extrap.val,
                    err=M_expanded_errs,
                    btsp='fill'
                )
                C_expanded_errs = C_quantities[-1]
                C_extrap = stat(
                    val=C_extrap.val,
                    err=C_expanded_errs,
                    btsp='fill'
                )
            Z_dict = {}
            for ens in ens_list:
                if ens in M_ens[1:]:
                    Z = M_extrap
                elif ens in C_ens[1:]:
                    Z = C_extrap
                elif ens[1] == '0':
                    if expanded_extrap:
                        Z = C_extrap if ens == 'C0' else M_extrap
                    else:
                        print(
                            f'using slope from {ens[0]} ensembles to chirally extrapolate {ens}')
                        params = C_fit_params if ens == 'C0' else M_fit_params
                        slope = stat(
                            val=params.val[1],
                            err=params.err[1],
                            btsp=params.btsp[:, 1]
                        )
                        Z0 = self.bag_dict[ens].Z_info.interpolate(
                            mu, **kwargs)
                        Z = Z0 - slope*(self.bag_dict[ens].m_pi**2)
                elif ens == 'F1M':
                    # chose slope from M ensembles for F1M
                    params = M_fit_params
                    slope = stat(
                        val=params.val[1],
                        err=params.err[1],
                        btsp=params.btsp[:, 1]
                    )
                    Z0 = self.bag_dict[ens].Z_info.interpolate(
                        mu, **kwargs)
                    Z = Z0 - slope*(self.bag_dict[ens].m_pi**2)

                else:
                    Z = self.bag_dict[ens].Z_info.interpolate(mu, **kwargs)
                Z_dict[ens] = Z

            if 'rotate' in kwargs:
                rot_mtx = kwargs['rotate']
            else:
                rot_mtx = np.eye(len(operators))

            rot_mtx = stat(
                val=rot_mtx,
                btsp='fill'
            )

            bags = [Z_dict[e]@(rot_mtx@self.bag_dict[e].bag)
                    for e in ens_list]

        self.bag = stat(
            val=[b.val for b in bags],
            err=[b.err for b in bags],
            btsp=np.array([b.btsp for b in bags])
        )
        self.bag.btsp = self.bag.btsp.swapaxes(0, 1)

    def fit_operator(self, mu, operator, ens_list=None,
                     guess=[1e-1, 1e-2, 1e-3], title='',
                     Z=None, plot=False, open=False,
                     **kwargs):

        if ens_list == None:
            ens_list = self.ens_list

        self.load_bag(mu, ens_list, **kwargs)

        op_idx = operators.index(operator)
        dof = len(ens_list)-len(guess)

        def ansatz(x, param, **kwargs):
            p = [self.bag_dict[e].ansatz(param, operator, **kwargs)
                 for e in ens_list]
            return np.array(p)

        y = stat(
            val=self.bag.val[:, op_idx],
            err=self.bag.err[:, op_idx],
            btsp=self.bag.btsp[:, :, op_idx]
        )
        x = y
        guess = np.array(guess)

        res = fit_func(x, y, ansatz, guess, **kwargs)
        cc_coeffs = stat(
            val=res.val[1:]/res.val[0],
            err='fill',
            btsp=np.array([res.btsp[k, 1:]/res.btsp[k, 0]
                           for k in range(N_boot)])
        )

        y_phys = stat(
            val=res.val[0],
            err=res.err[0],
            btsp=res.btsp[:, 0]
        )

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            label = operator
            if 'rotate' in kwargs:
                N = len(operators)
                rot_str = []
                for c in range(N):
                    val = np.around(kwargs['rotate'][op_idx, c], 2)
                    if val != 0.0 and val != 1.0:
                        rot_str.append('('+str(val)+')*'+operators[c])
                    elif val == 1.0:
                        rot_str.append(operators[c])
                label = ' + '.join(rot_str)

            ax[0].set_title(label)
            ax[1].set_title(label)

            ax[0].errorbar([0], [y_phys.val], yerr=[y_phys.err],
                           color='k', fmt='o', capsize=4, label='phys')
            ax[1].errorbar([(m_pi_PDG**2).val], [y_phys.val],
                           xerr=[(m_pi_PDG**2).err],
                           yerr=[y_phys.err], color='k',
                           fmt='o', capsize=4, label='phys')
            ax[1].axvline(m_pi_PDG.val**2, color='k',
                          linestyle='dashed')

            x_asq = np.linspace(0, (1/1.7)**2, 50)
            x_msq = np.linspace(0.01, 0.2, 50)
            for ens_idx, e in enumerate(ens_list):
                m_pi = self.bag_dict[e].m_pi
                m_f_sq = self.bag_dict[e].m_f_sq
                a_sq = self.bag_dict[e].a_sq

                y_asq = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            res.val, x, m_f_sq.val,
                            m_f_sq_PDG.val,
                            operator, **kwargs)
                         for x in x_asq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], x, m_f_sq.btsp[k],
                        m_f_sq_PDG.btsp[k],
                        operator, **kwargs) for x in x_asq]
                        for k in range(N_boot)])
                )

                ax[0].plot(x_asq, y_asq.val, color=self.colors[e])
                ax[0].fill_between(x_asq, y_asq.val+y_asq.err,
                                   y_asq.val-y_asq.err,
                                   color=self.colors[e], alpha=0.2)

                y_mpi = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            res.val, a_sq.val,
                            x/(f_pi_PDG.val**2),
                            m_f_sq_PDG.val,
                            operator, **kwargs) for x in x_msq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], a_sq.btsp[k],
                        x/(f_pi_PDG.btsp[k]**2),
                        m_f_sq_PDG.btsp[k],
                        operator, **kwargs) for x in x_msq]
                        for k in range(N_boot)])
                )

                ax[1].plot(x_msq, y_mpi.val, color=self.colors[e])
                ax[1].fill_between(x_msq, y_mpi.val+y_mpi.err,
                                   y_mpi.val-y_mpi.err,
                                   color=self.colors[e], alpha=0.2)

                ax[0].errorbar([a_sq.val],
                               [self.bag.val[ens_idx, op_idx]],
                               yerr=[self.bag.err[ens_idx, op_idx]],
                               xerr=[a_sq.err],
                               fmt='o', label=e,
                               capsize=4, color=self.colors[e],
                               mfc='None')

                mpi_phys = stat(
                    val=m_pi.val**2/a_sq.val,
                    err='fill',
                    btsp=[(m_pi.btsp[k]**2)/a_sq.btsp[k]
                          for k in range(N_boot)]
                )
                ax[1].errorbar([mpi_phys.val],
                               [self.bag.val[ens_idx, op_idx]],
                               yerr=[self.bag.err[ens_idx, op_idx]],
                               xerr=[mpi_phys.err],
                               fmt='o', label=e, capsize=4,
                               color=self.colors[e],
                               mfc='None')

            ax[0].legend()
            ax[0].set_xlabel(r'$a^2$ (GeV${}^{-2}$)')
            ax[1].legend()
            ax[1].set_xlabel(r'$m_{\pi}^2$ (GeV${}^2$)')

            ax[0].text(0.4, 0.05, r'$\chi^2$/DOF:'+'{:.3f}'.format(
                res.chi_sq/res.DOF), transform=ax[0].transAxes)
            ax[1].text(0.4, 0.05, r'$p$-value:'+'{:.3f}'.format(
                res.pvalue), transform=ax[1].transAxes)

            title += r', $\mu='+str(np.around(mu, 2))+'$ GeV'
            plt.suptitle(title, y=0.95)

            if 'filename' in kwargs:
                filename = kwargs['filename']
            else:
                filename = f'plots/bag_fits_{operator}.pdf'
            call_PDF(filename, open=open)

        return y_phys, cc_coeffs, res.chi_sq/res.DOF, res.pvalue
