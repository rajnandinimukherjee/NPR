import pdb
from coeffs import *
from bag_param_renorm import *

Z_dict = {ens: Z_analysis(ens) for ens in bag_ensembles}
bag_dict = {ens: bag_analysis(ens) for ens in bag_ensembles}


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


def rotation_mtx(theta, phi, **kwargs):
    num_ops = len(bag_analysis.operators)
    A = np.zeros(shape=(num_ops, num_ops))
    A[0, 0] = 1
    ct, st = np.cos(theta), np.sin(theta)
    A[1:3, 1:3] = np.array([[ct, -st], [st, ct]])
    cp, sp = np.cos(phi), np.sin(phi)
    A[3:, 3:] = np.array([[cp, -sp], [sp, cp]])
    return A


class bag_fits:

    def __init__(self, ens_list, **kwargs):
        self.ens_list = ens_list
        self.bag_dict = {e: bag_analysis(e)
                         for e in self.ens_list}
        self.colors = {list(self.bag_dict.keys())[k]: list(
                       mc.TABLEAU_COLORS.keys())[k]
                       for k in range(len(self.ens_list))}

    def fit_operator(self, operator, mu, ens_list=None,
                     guess=[1e-1, 1e-2, 1e-3], **kwargs):
        if ens_list == None:
            ens_list = self.ens_list

        op_idx = bag_analysis.operators.index(operator)
        dof = len(ens_list)-len(guess)

        def pred(params, operator, **kwargs):
            p = [self.bag_dict[e].ansatz(params, operator, **kwargs)
                 for e in ens_list]
            return p

        def diff(bag, params, operator, **kwargs):
            return np.array(bag) - np.array(pred(params, operator, **kwargs))

        bags = [self.bag_dict[e].interpolate(mu, **kwargs)
                for e in ens_list]
        bags_op = stat(
            val=[bag.val[op_idx] for bag in bags],
            err=[bag.err[op_idx] for bag in bags],
            btsp=np.array([bag.btsp[:, op_idx] for bag in bags]).T
        )

        COV = np.diag(bags_op.err**2)
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params):
            return L.dot(diff(bags_op.val, params, operator,
                              fit='central', **kwargs))

        guess = np.array(guess)
        res = least_squares(LD, guess, ftol=1e-15, gtol=1e-15,
                            max_nfev=1e+5, xtol=1e-15)
        chi_sq = LD(res.x).dot(LD(res.x))
        pvalue = gammaincc(dof/2, chi_sq/2)

        res_btsp = np.zeros(shape=(N_boot, len(res.x)))
        for k in range(N_boot):
            def LD_btsp(params):
                return L.dot(diff(bags_op.btsp[k, :], params, operator,
                                  fit='btsp', k=k, **kwargs))
            res_k = least_squares(LD_btsp, guess,
                                  ftol=1e-10, gtol=1e-10)
            res_btsp[k, :] = res_k.x

        params = stat(
            val=res.x,
            err='fill',
            btsp=res_btsp
        )
        params.res = res

        return params, chi_sq/dof, pvalue

    def plot_fits(self, mu, ops=operators, ens_list=None,
                  title='', filename='plots/bag_fits.pdf', open=False,
                  passvals=True, save=True, **kwargs):

        if ens_list == None:
            ens_list = self.ens_list

        num_ops = len(ops)

        fig, ax = plt.subplots(num_ops, 2, figsize=(15, 6*num_ops))
        for i in range(num_ops):
            if num_ops == 1:
                ax0, ax1 = ax[0], ax[1]
            else:
                ax0, ax1 = ax[i, 0], ax[i, 1]

            operator = ops[i]
            op_idx = operators.index(operator)
            if 'rotate' in kwargs:
                N = len(operators)
                rot_str = np.empty(N, dtype=object)
                for r in range(N):
                    rot_str[r] = []
                    for c in range(N):
                        val = np.around(kwargs['rotate'][r, c], 2)
                        if val != 0.0 and val != 1.0:
                            rot_str[r].append('('+str(val)+')*'+operators[c])
                        elif val == 1.0:
                            rot_str[r].append(operators[c])
                    rot_str[r] = ' + '.join(rot_str[r])
                ax0.title.set_text(rot_str[op_idx])
                ax1.title.set_text(rot_str[op_idx])
            else:
                ax0.title.set_text(operator)
                ax1.title.set_text(operator)
            if 'names' in kwargs:
                ax0.title.set_text(kwargs['names'][i])
                ax1.title.set_text(kwargs['names'][i])

            params, chi_sq_dof, pvalue = self.fit_operator(
                operator, mu, ens_list=ens_list, **kwargs)

            cc_coeffs = stat(
                val=params.val[1:3]/params.val[0],
                err='fill',
                btsp=np.array([params.btsp[k, 1:3]/params.btsp[k, 0]
                               for k in range(N_boot)])
            )

            y_phys = stat(
                val=params.val[0],
                err=params.err[0],
                btsp=params.btsp[:, 0]
            )

            ax0.errorbar([0], [y_phys.val], yerr=[y_phys.err],
                         color='k', fmt='o', capsize=4, label='phys')
            ax1.errorbar([m_pi_PDG.val**2], [y_phys.val], yerr=[y_phys.err],
                         color='k', fmt='o', capsize=4, label='phys')
            ax1.axvline(m_pi_PDG.val**2, color='k', linestyle='dashed')

            x_asq = np.linspace(0, (1/1.7)**2, 50)
            x_msq = np.linspace(0.01, 0.2, 50)
            for e in ens_list:
                m_pi = self.bag_dict[e].m_pi
                m_f_sq = self.bag_dict[e].m_f_sq
                a_sq = self.bag_dict[e].a_sq

                y_asq = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            params.val, x, m_f_sq.val,
                            m_f_sq_PDG.val,
                            operator, **kwargs)
                         for x in x_asq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        params.btsp[k, :], x, m_f_sq.btsp[k],
                        m_f_sq_PDG.btsp[k],
                        operator, **kwargs) for x in x_asq]
                        for k in range(N_boot)])
                )

                ax0.plot(x_asq, y_asq.val, color=self.colors[e])
                ax0.fill_between(x_asq, y_asq.val+y_asq.err, y_asq.val-y_asq.err,
                                 color=self.colors[e], alpha=0.2)

                y_mpi = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            params.val, a_sq.val,
                            x/(f_pi_PDG.val**2),
                            m_f_sq_PDG.val,
                            operator, **kwargs) for x in x_msq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        params.btsp[k, :], a_sq.btsp[k],
                        x/(f_pi_PDG.btsp[k]**2),
                        m_f_sq_PDG.btsp[k],
                        operator, **kwargs) for x in x_msq]
                        for k in range(N_boot)])
                )

                ax1.plot(x_msq, y_mpi.val, color=self.colors[e])
                ax1.fill_between(x_msq, y_mpi.val+y_mpi.err, y_mpi.val-y_mpi.err,
                                 color=self.colors[e], alpha=0.2)

                bag = self.bag_dict[e].interpolate(mu, **kwargs)
                ax0.errorbar([a_sq.val], [bag.val[op_idx]],
                             yerr=[bag.err[op_idx]],
                             xerr=[a_sq.err],
                             fmt='o', label=e,
                             capsize=4, color=self.colors[e], mfc='None')

                mpi_phys = stat(
                    val=m_pi.val**2/a_sq.val,
                    err='fill',
                    btsp=[(m_pi.btsp[k]**2)/a_sq.btsp[k]
                          for k in range(N_boot)]
                )
                ax1.errorbar([mpi_phys.val], [bag.val[op_idx]],
                             yerr=[bag.err[op_idx]],
                             xerr=[mpi_phys.err],
                             fmt='o', label=e, capsize=4,
                             color=self.colors[e],
                             mfc='None')

            ax0.legend()
            ax0.set_xlabel(r'$a^2$ (GeV${}^{-2}$)')
            ax1.legend()
            ax1.set_xlabel(r'$m_{\pi}^2$ (GeV${}^2$)')

            ax0.text(0.4, 0.05, r'$\chi^2$/DOF:'+'{:.3f}'.format(chi_sq_dof),
                     transform=ax0.transAxes)
            ax1.text(0.4, 0.05, r'$p$-value:'+'{:.3f}'.format(pvalue),
                     transform=ax1.transAxes)

        if title != '':
            title += r', '
        title += r'$\mu='+str(np.around(mu, 2))+'$ GeV'
        plt.suptitle(title, y=0.95)

        if save:
            pp = PdfPages(filename)
            fig_nums = plt.get_fignums()
            figs = [plt.figure(n) for n in fig_nums]
            for fig in figs:
                fig.savefig(pp, format='pdf')
            pp.close()
            plt.close('all')

            if open:
                print(f'Saved plot to {filename}.')
                os.system("open "+filename)
        if passvals and num_ops == 1:
            return y_phys, cc_coeffs, chi_sq_dof, pvalue


class Z_fits:

    chiral_ens = [ens for ens in bag_ensembles if ens[0] == 'M']
    cont_ens = ['F1M', 'M0']

    def __init__(self, ens_list, bag=True, **kwargs):
        self.ens_list = ens_list
        self.bag = bag
        self.Z_dict = {e: Z_analysis(e, bag=self.bag)
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

        # ==== Step 3: continuum extrapolate F0 and M0=======================
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
            if mask[i, j]:
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
            if mask[i, j]:
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
            if mask[i, j]:
                y = [sig.val[i, j] for sig in sigmas]
                yerr = [sig.err[i, j] for sig in sigmas]

                ax[i, j].errorbar(x, y, yerr=yerr,
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
