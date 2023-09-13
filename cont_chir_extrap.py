import pdb
from coeffs import *
from bag_param_renorm import *

Z_dict = {ens: Z_analysis(ens) for ens in bag_ensembles}
bag_dict = {ens: bag_analysis(ens) for ens in bag_ensembles}


cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(bag_ensembles)), cmap.N)


def chiral_continuum_ansatz(params, a_sq, mpi_f_m_sq, operator, **kwargs):
    op_idx = bag_analysis.operators.index(operator)
    func = params[0]*(1+params[1]*a_sq + params[2]*mpi_f_m_sq)
    if 'addnl_terms' in kwargs.keys():
        if kwargs['addnl_terms'] == 'a4':
            func += params[0]*(params[3]*(a_sq**2))
        elif kwargs['addnl_terms'] == 'm4':
            func += params[0]*(params[3]*(mpi_f_m_sq**2))
        elif kwargs['addnl_terms'] == 'log':
            # chiral log term included
            chir_log_coeffs = np.array([-0.5, -0.5, -0.5, 0.5, 0.5])
            chir_log_coeffs = chir_log_coeffs/((4*np.pi)**2)
            Lambda_QCD = 1.0
            log_ratio = mpi_f_m_sq*(f_pi_PDG**2)/(Lambda_QCD**2)
            log_term = chir_log_coeffs[op_idx]*np.log(log_ratio)
            func += params[0]*log_term*mpi_f_m_sq
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

        def pred(params, operator, **akwargs):
            p = [self.bag_dict[e].ansatz(params, operator, **kwargs)
                 for e in ens_list]
            return p

        def diff(bag, params, operator, **kwargs):
            return np.array(bag) - np.array(pred(params, operator, **kwargs))

        bags_central = np.array([self.bag_dict[e].interpolate(
            mu, **kwargs)[0][op_idx] for e in ens_list])
        COV = np.diag([self.bag_dict[e].interpolate(
            mu, **kwargs)[1][op_idx]**2 for e in ens_list])
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params):
            return L.dot(diff(bags_central, params, operator,
                              fit='central', **kwargs))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        pvalue = gammaincc(dof/2, chi_sq/2)

        res_btsp = np.zeros(shape=(N_boot, len(res.x)))
        bag_btsp = np.array([self.bag_dict[e].interpolate(
            mu, **kwargs)[2][:, op_idx] for e in ens_list])
        for k in range(N_boot):

            def LD_btsp(params):
                return L.dot(diff(bag_btsp[:, k], params, operator,
                                  fit='btsp', k=k, **kwargs))
            res_k = least_squares(LD_btsp, guess,
                                  ftol=1e-10, gtol=1e-10)
            res_btsp[k, :] = res_k.x

        res_err = [((res_btsp[:, i]-res.x[i]).dot(
            res_btsp[:, i]-res.x[i])/N_boot)**0.5
            for i in range(len(res.x))]

        return res.x, chi_sq/dof, pvalue, res_err, res_btsp

    def plot_fits(self, mu, ops=None, ens_list=None, cont_adjust=False,
                  title='', filename='plots/bag_fits.pdf', open=False,
                  passvals=True, save=True, **kwargs):
        if ens_list == None:
            ens_list = self.ens_list

        if ops == None:
            ops = operators
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

            params, chi_sq_dof, pvalue, err, btsp = self.fit_operator(
                operator, mu, ens_list=ens_list, **kwargs)

            cc_coeffs = params[1:3]
            cc_coeffs_err = err[1:3]

            x_asq = np.linspace(0, (1/1.7)**2, 50)
            x_mpi_sq = np.linspace(0.01, 0.2, 50)

            y_phys = chiral_continuum_ansatz(params, 0,
                                             (mpi_PDG/f_pi_PDG)**2, operator)
            y_phys_btsp = np.array([chiral_continuum_ansatz(btsp[k, :], 0,
                                   (mpi_PDG/f_pi_PDG)**2, operator)
                for k in range(N_boot)])
            y_phys_err = st_dev(y_phys_btsp, mean=y_phys)

            subtract = params[0]*params[2] * \
                (mpi_PDG/f_pi_PDG)**2 if cont_adjust else 1
            scaling = 1/(y_phys-subtract) if cont_adjust else 1
            y_phys_a = 1 if cont_adjust else y_phys

            ax0.errorbar([0], [y_phys_a], yerr=[y_phys_err],
                         color='k', fmt='o', capsize=4, label='phys')
            ax1.errorbar([mpi_PDG**2], [y_phys], yerr=[y_phys_err],
                         color='k', fmt='o', capsize=4, label='phys')
            ax1.axvline(mpi_PDG**2, color='k', linestyle='dashed')

            for e in ens_list:
                a_sq = self.bag_dict[e].ainv**(-2)
                mpi_sq = self.bag_dict[e].mpi**2
                f_m_sq = self.bag_dict[e].f_m_ren**2
                mpi_f_m_sq = mpi_sq/f_m_sq

                subtract = params[0]*params[2]*mpi_f_m_sq if cont_adjust else 0

                y_asq = np.array([chiral_continuum_ansatz(params, x,
                                 mpi_f_m_sq, operator, **kwargs) for x in x_asq])
                y_asq_btsp = np.array([[chiral_continuum_ansatz(btsp[k, :],
                                      x_asq[a], mpi_f_m_sq, operator, **kwargs)
                    for k in range(N_boot)]
                    for a in range(len(x_asq))])
                y_asq_err = np.array([st_dev(y_asq_btsp[a, :], mean=y_asq[a])
                                     for a in range(len(x_asq))])
                y_asq = (y_asq-subtract)*scaling
                ax0.plot(x_asq, y_asq, color=self.colors[e])
                ax0.fill_between(x_asq, y_asq+y_asq_err, y_asq-y_asq_err,
                                 color=self.colors[e], alpha=0.2)

                y_mpi = np.array([chiral_continuum_ansatz(params, a_sq,
                                 x*a_sq/f_m_sq, operator, **kwargs)
                                 for x in x_mpi_sq])
                y_mpi_btsp = np.array([[chiral_continuum_ansatz(btsp[k, :], a_sq,
                                      x_mpi_sq[p]*a_sq/f_m_sq, operator, **kwargs)
                    for k in range(N_boot)]
                    for p in range(len(x_mpi_sq))])
                y_mpi_err = np.array([st_dev(y_mpi_btsp[p, :], mean=y_mpi[p])
                                     for p in range(len(x_mpi_sq))])
                ax1.plot(x_mpi_sq, y_mpi, color=self.colors[e])
                ax1.fill_between(x_mpi_sq, y_mpi+y_mpi_err, y_mpi-y_mpi_err,
                                 color=self.colors[e], alpha=0.2)

                bag = self.bag_dict[e].interpolate(mu, **kwargs)[0][op_idx]
                bag_err = self.bag_dict[e].interpolate(mu, **kwargs)[1][op_idx]
                ax0.errorbar([a_sq], [(bag-subtract)*scaling],
                             yerr=[bag_err], fmt='o', label=e,
                             capsize=4, color=self.colors[e], mfc='None')
                ax1.errorbar([mpi_sq/a_sq], [bag], yerr=[bag_err], fmt='o',
                             label=e, capsize=4, color=self.colors[e],
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
            return y_phys, y_phys_err, y_phys_btsp, \
                cc_coeffs, cc_coeffs_err, chi_sq_dof, pvalue


class Z_fits:

    def __init__(self, ens_list, bag=False, **kwargs):
        self.ens_list = ens_list
        self.Z_dict = {e: Z_analysis(e, bag=bag)
                       for e in self.ens_list}
        self.colors = {list(self.Z_dict.keys())[k]: list(
                       mc.TABLEAU_COLORS.keys())[k]
                       for k in range(len(self.ens_list))}

    def plot_sigma(self, mu2, mu1, i, j,
                   filename='plots/sigma_fits.pdf', **kwargs):
        fig = plt.figure(figsize=(4, 4))
        for e in self.ens_list:
            a_sq = self.Z_dict[e].ainv**(-2)
            sig, sig_err, sig_btsp = self.Z_dict[e].scale_evolve(mu2, mu1)
            sig, sig_err = sig[i, j], sig_err[i, j]
            plt.errorbar([a_sq], [sig], yerr=[sig_err],
                         fmt='o', capsize=4, c=self.colors[e])
        plt.xlabel(r'$a^2$ (GeV${}^2$)')
        plt.title(r'$\sigma_{npt}('+str(np.around(mu2, 2)) +
                  ','+str(np.around(mu1, 2))+r')['+str(i)+','+str(j)+r']$')

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)
