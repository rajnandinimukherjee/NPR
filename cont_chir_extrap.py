import pdb
from coeffs import *
from bag_param_renorm import *

cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(bag_ensembles)), cmap.N)
ens_colors = {bag_ensembles[k]: list(mc.TABLEAU_COLORS.keys())[
    k] for k in range(len(bag_ensembles))}

M_ens = ['M0', 'M1', 'M2', 'M3']
C_ens = ['C0', 'C1', 'C2']


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
        elif kwargs['addnl_terms'] == 'del_ms':
            func += params[3]*kwargs['ms_diff']

    return func


class Z_fits:
    mask = mask
    N_ops = len(operators)

    def __init__(self, ens_list, norm='V', **kwargs):
        self.ens_list = ens_list
        self.norm = norm
        if norm == '11':
            self.mask[0, 0] = False
        self.Z_dict = {e: Z_analysis(e, norm=self.norm)
                       for e in bag_ensembles+['C1M']}

    def Z_assignment(self, mu, chiral_extrap=False, **kwargs):
        if not chiral_extrap:
            assignment = {}
            for e in self.ens_list:
                Z_linear = self.Z_dict[e].interpolate(
                    mu, fittype='linear', **kwargs)
                Z_quadratic = self.Z_dict[e].interpolate(
                    mu, fittype='quadratic', **kwargs)
                stat_err = np.array([[max(Z_linear.err[i, j], Z_quadratic.err[i, j])
                                      for j in range(self.N_ops)]
                                     for i in range(self.N_ops)])
                sys_err = np.array([[np.abs(Z_linear.val[i, j]-Z_quadratic.val[i, j])/2
                                     for j in range(self.N_ops)]
                                    for i in range(self.N_ops)])
                Z = stat(
                    val=(Z_linear.val+Z_quadratic.val)/2,
                    err=(stat_err**2+sys_err**2)**0.5,
                    btsp='fill'
                )
                Z.stat_err = stat_err
                Z.sys_err = sys_err
                assignment[e] = Z
            return assignment
        else:
            extrap_assign = {}
            for fittype in ['linear', 'quadratic']:
                extrap_assign[fittype] = {}
                CS_extrap, CS_fit_params = self.Z_chiral_extrap(
                    ['C1', 'C2'], mu, fittype=fittype, **kwargs)
                CM_extrap, CM_fit_params = self.Z_chiral_extrap(
                    ['C0', 'C1M'], mu, fittype=fittype, **kwargs)
                MS_extrap, MS_fit_params = self.Z_chiral_extrap(
                    ['M1', 'M2', 'M3'], mu, fittype=fittype, **kwargs)

                for ens in self.ens_list:
                    m_pi = self.Z_dict[ens].m_pi
                    if ens in C_ens[1:]:
                        Z = CS_extrap
                    elif ens in M_ens[1:]:
                        Z = MS_extrap
                    elif ens == 'C0':
                        Z = CM_extrap
                    elif ens == 'M0':
                        slope = MS_fit_params[1]
                        Z0 = self.Z_dict[ens].interpolate(mu, **kwargs)
                        Z = Z0 - slope*(m_pi**2)
                    elif ens == 'F1M':
                        C_slope = CM_fit_params[1]
                        M_slope = MS_fit_params[1]
                        Z0 = self.Z_dict[ens].interpolate(mu, **kwargs)

                        Z_C = Z0 - C_slope*(m_pi**2)
                        Z_M = Z0 - M_slope*(m_pi**2)
                        stat_err = np.array([[max(Z_C.err[i, j], Z_M.err[i, j])
                                            for j in range(5)] for i in range(5)])
                        sys_err = np.abs(Z_C.val-Z_M.val)/2
                        Z = stat(
                            val=(Z_C.val+Z_M.val)/2,
                            err=(stat_err**2 + sys_err**2)**0.5,
                            btsp='fill'
                        )
                        Z.sys_err = sys_err
                    extrap_assign[fittype][ens] = Z

            extrap_assign['both'] = {}
            for ens in self.ens_list:
                Z_linear = extrap_assign['linear'][ens]
                Z_quadratic = extrap_assign['quadratic'][ens]
                stat_err = np.array([[max(Z_linear.err[i, j], Z_quadratic.err[i, j])
                                      for j in range(self.N_ops)]
                                     for i in range(self.N_ops)])
                sys_err = np.array([[np.abs(Z_linear.val[i, j]-Z_quadratic.val[i, j])/2
                                     for j in range(self.N_ops)]
                                    for i in range(self.N_ops)])
                Z = stat(
                    val=(Z_linear.val+Z_quadratic.val)/2,
                    err=(stat_err**2+sys_err**2)**0.5,
                    btsp='fill'
                )
                Z.stat_err = stat_err
                Z.sys_err = sys_err
                extrap_assign['both'][ens] = Z

            return extrap_assign['both']

    def Z_chiral_extrap(self, ens_list, mu, plot=False,
                        extrap_label='0',
                        filename='plots/Z_chiral_extrap.pdf',
                        fittype='linear', **kwargs):

        chiral_data = join_stats([self.Z_dict[ens].interpolate(
            mu, fittype=fittype, **kwargs)
            for ens in ens_list])
        x = join_stats([self.Z_dict[ens].m_pi**2
                        for ens in ens_list])

        def ansatz(msq, param, **kwargs):
            return param[0] + param[1]*msq

        extrap = np.zeros(shape=(self.N_ops, self.N_ops), dtype=object)
        param_save = np.zeros(shape=(2, self.N_ops, self.N_ops))
        param_save_btsp = np.zeros(shape=(N_boot, 2, self.N_ops, self.N_ops))

        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=chiral_data.val[:, i, j],
                    err=chiral_data.err[:, i, j],
                    btsp=chiral_data.btsp[:, :, i, j])
                res = fit_func(x, y, ansatz, [1, 0.1], **kwargs)

                param_save[:, i, j] = res.val
                param_save_btsp[:, :, i, j] = res.btsp
                extrap[i, j] = res.mapping(0.0)
            else:
                extrap[i, j] = stat(val=0, btsp='fill')

        fit_params = stat(
            val=param_save,
            err='fill',
            btsp=param_save_btsp
        )
        extrap = stat(
            val=[[extrap[i, j].val for j in range(self.N_ops)]
                 for i in range(self.N_ops)],
            err='fill',
            btsp=[[[extrap[i, j].btsp[k] for j in range(self.N_ops)]
                   for i in range(self.N_ops)] for k in range(N_boot)]
        )

        def mapping(i, j, m_sq):
            if type(m_sq) is not stat:
                m_sq = stat(
                    val=m_sq,
                    btsp='fill'
                )
            Z_map = stat(
                val=ansatz(m_sq.val, fit_params.val[:, i, j]),
                err='fill',
                btsp=[ansatz(m_sq.btsp[k], fit_params.btsp[k, :, i, j])
                      for k in range(N_boot)]
            )
            return Z_map

        if plot:
            fig, ax = plt.subplots(nrows=self.N_ops, ncols=self.N_ops,
                                   figsize=(16, 16))
            plt.subplots_adjust(hspace=0, wspace=0)

            for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                if self.mask[i, j]:
                    y = stat(
                        val=chiral_data.val[:, i, j],
                        err=chiral_data.err[:, i, j],
                        btsp=chiral_data.btsp[:, :, i, j])

                    ax[i, j].errorbar(x.val, y.val,
                                      yerr=y.err,
                                      xerr=x.err, fmt='o',
                                      capsize=4)
                    xmin, xmax = ax[i, j].get_xlim()
                    msq_grain = np.linspace(0.0, xmax, 50)
                    y_grain = mapping(i, j, msq_grain)
                    ax[i, j].fill_between(msq_grain,
                                          (y_grain.val-y_grain.err),
                                          (y_grain.val+y_grain.err),
                                          alpha=0.1, color='k')
                    ax[i, j].axvline(0.0, c='k', linestyle='dashed', alpha=0.1)
                    ax[i, j].errorbar([0.0], extrap.val[i, j],
                                      yerr=extrap.err[i, j],
                                      color='0.1', fmt='o', capsize=4)
                    if j == 2 or j == 4:
                        ax[i, j].yaxis.tick_right()
                    if i == 1 or i == 3:
                        ax[i, j].set_xticks([])

                    if i == 0 or i == 1 or i == 3:
                        ax_twin = ax[i, j].twiny()
                        ax_twin.set_xlim(ax[i, j].get_xlim())
                        ax_twin.set_xticks([0]+list(x.val))
                        ax_twin.set_xticklabels([extrap_label]+ens_list)
                else:
                    ax[i, j].axis('off')
            plt.suptitle(r'$Z_{ij}^{'+','.join(ens_list) +
                         '}(\mu='+str(mu)+'$ GeV$)/Z_{'+self.norm+r'}$ vs $(am_\pi)^2$', y=0.9)
            call_PDF(filename)

        return extrap, fit_params


class sigma:
    relevant_ensembles = ['C1', 'C0', 'M1', 'M0', 'F1M']
    N_ops = len(operators)
    mask = mask

    def __init__(self, norm='V', **kwargs):
        self.norm = norm
        if self.norm == '11':
            self.mask[0, 0] = False

        self.Z_fits = Z_fits(bag_ensembles, norm=self.norm)
        self.Z_dict = {e: Z_analysis(e, norm=self.norm)
                       for e in self.relevant_ensembles}

    def calc_running(self, mu2, mu1, plot=False, include_C=True,
                     filename='plots/Z_running.pdf', **kwargs):
        if not include_C:
            ens_list = self.relevant_ensembles[2:]
        else:
            ens_list = self.relevant_ensembles.copy()

        Z_dict_mu1 = self.Z_fits.Z_assignment(mu1, **kwargs)
        Z_dict_mu2 = self.Z_fits.Z_assignment(mu2, **kwargs)
        sigmas = join_stats([stat(
            val=Z_dict_mu2[e].val@np.linalg.inv(Z_dict_mu1[e].val),
            err='fill',
            btsp=np.array([Z_dict_mu2[e].btsp[k]@np.linalg.inv(
                Z_dict_mu1[e].btsp[k]) for k in range(N_boot)])
        ) for e in ens_list])

        def ansatz(asq, param, **kwargs):
            return param[0] + param[1]*asq

        x = join_stats([self.Z_dict[e].a_sq for e in ens_list])

        extrap = np.zeros(shape=(self.N_ops, self.N_ops), dtype=object)
        param_save = np.zeros(shape=(2, self.N_ops, self.N_ops))
        param_save_btsp = np.zeros(shape=(N_boot, 2, self.N_ops, self.N_ops))
        GOF = np.zeros(shape=(self.N_ops, self.N_ops), dtype=object)
        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=sigmas.val[:, i, j],
                    err=sigmas.err[:, i, j],
                    btsp=np.array([sigmas.btsp[k, :, i, j]
                                   for k in range(N_boot)])
                )

                res = fit_func(x, y, ansatz, [1, 1e-1], **kwargs)
                param_save[:, i, j] = res.val
                param_save_btsp[:, :, i, j] = res.btsp
                extrap[i, j] = res.mapping(0.0)
                GOF[i, j] = str(np.around(res.pvalue, 3))
            else:
                extrap[i, j] = stat(val=0, btsp='fill')

        fit_params = stat(
            val=param_save,
            err='fill',
            btsp=param_save_btsp
        )
        extrap = stat(
            val=[[extrap[i, j].val for j in range(self.N_ops)]
                 for i in range(self.N_ops)],
            err='fill',
            btsp=[[[extrap[i, j].btsp[k] for j in range(self.N_ops)]
                   for i in range(self.N_ops)] for k in range(N_boot)]
        )

        def mapping(i, j, a_sq):
            if type(a_sq) is not stat:
                a_sq = stat(
                    val=a_sq,
                    btsp='fill'
                )
            sig_map = stat(
                val=ansatz(a_sq.val, fit_params.val[:, i, j]),
                err='fill',
                btsp=[ansatz(a_sq.btsp[k], fit_params.btsp[k, :, i, j])
                      for k in range(N_boot)]
            )
            return sig_map

        if plot:
            fig, ax = plt.subplots(nrows=self.N_ops, ncols=self.N_ops,
                                   figsize=(16, 16))
            plt.subplots_adjust(hspace=0, wspace=0)
            for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                if self.mask[i, j]:
                    y = stat(
                        val=sigmas.val[:, i, j],
                        err=sigmas.err[:, i, j],
                        btsp=np.array([sigmas.btsp[k, :, i, j]
                                       for k in range(N_boot)])
                    )
                    ax[i, j].errorbar(x.val, y.val,
                                      yerr=y.err,
                                      xerr=x.err, fmt='o',
                                      capsize=4)
                    xmin, xmax = ax[i, j].get_xlim()
                    asq_grain = np.linspace(0.0, xmax, 50)
                    y_grain = mapping(i, j, asq_grain)
                    ax[i, j].fill_between(asq_grain,
                                          (y_grain.val-y_grain.err),
                                          (y_grain.val+y_grain.err),
                                          alpha=0.1, color='k')
                    ax[i, j].axvline(0.0, c='k', linestyle='dashed', alpha=0.1)
                    ax[i, j].errorbar([0.0], extrap.val[i, j],
                                      yerr=extrap.err[i, j],
                                      color='0.1', fmt='o', capsize=4)
                    ax[i, j].text(0.5, 0.1, r'$p$-val:'+GOF[i, j]+r'',
                                  va='center', ha='center',
                                  transform=ax[i, j].transAxes)
                    if j == 2 or j == 4:
                        ax[i, j].yaxis.tick_right()

                    if i == 1 or i == 3:
                        ax[i, j].set_xticks([])

                else:
                    ax[i, j].axis('off')
            plt.suptitle(r'$\sigma^{'+','.join(ens_list) +
                         '}_{'+self.norm+r'}('+str(mu1)+','+str(mu2)+r')$ vs $a^2$', y=0.9)
            call_PDF(filename)

        return extrap


class bag_fits:
    operators = operators

    def __init__(self, ens_list, obj='bag', **kwargs):
        self.ens_list = ens_list
        self.obj = obj
        self.mu = 0
        if obj == 'ratio':
            self.norm = '11'
            self.operators = operators[1:]
        elif obj == 'ratio2':
            self.norm = '11/AS'
            self.operators = operators[1:]
        elif obj == 'bag':
            self.norm = 'bag'

        self.bag_dict = {e: bag_analysis(e, obj=obj)
                         for e in self.ens_list}
        self.Z_fits = Z_fits(self.ens_list, norm=self.norm)

    def load_bag(self, mu, chiral_extrap=False,
                 **kwargs):

        Z_dict = self.Z_fits.Z_assignment(mu, chiral_extrap=chiral_extrap,
                                          **kwargs)
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
        else:
            rot_mtx = np.eye(len(operators))

        rot_mtx = stat(
            val=rot_mtx,
            btsp='fill'
        )

        self.mu = mu
        self.bag = join_stats([Z_dict[e]@(rot_mtx@self.bag_dict[e].bag)
                               for e in self.ens_list])

    def fit_operator(self, mu, operator,
                     ens_list=None,
                     guess=[1e-1, 1e-2, 1e-3],
                     plot=False, open=False,
                     rescale=False,
                     title='',
                     figsize=(15, 6),
                     legend_axis=1,
                     **kwargs):

        if self.mu != mu:
            self.load_bag(mu, **kwargs)

        if ens_list == None:
            ens_list = self.ens_list
            bag = self.bag
        else:
            ens_indices = [self.ens_list.index(ens)
                           for ens in ens_list]
            bag = stat(
                val=self.bag.val[ens_indices, :],
                err=self.bag.err[ens_indices, :],
                btsp=self.bag.btsp[:, ens_indices, :]
            )

        op_idx = operators.index(operator)
        dof = len(ens_list)-len(guess)

        def ansatz(x, param, **kwargs):
            p = [self.bag_dict[e].ansatz(param, operator, **kwargs)
                 for e in ens_list]
            return np.array(p)

        y = stat(
            val=bag.val[:, op_idx],
            err=bag.err[:, op_idx],
            btsp=bag.btsp[:, :, op_idx]
        )
        if rescale and self.obj == 'bag':
            norm = norm_factors(**kwargs)[op_idx]
            y = y/norm

        x = y
        guess = np.array(guess)

        res = fit_func(x, y, ansatz, guess, **kwargs)
        cc_coeffs = res[1:]/res[0]
        y_phys = res[0]

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=figsize,
                                   sharey=True)
            plt.subplots_adjust(wspace=0)

            if 'label' in kwargs:
                label = kwargs['label']
            else:
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

            x_asq = np.linspace(0, (1/1.7)**2, 50)
            x_msq = np.linspace(0.01, 0.2, 50)

            for ens_idx, e in enumerate(ens_list):
                m_pi = self.bag_dict[e].m_pi
                m_f_sq = self.bag_dict[e].m_f_sq
                a_sq = self.bag_dict[e].a_sq
                ms_diff = self.bag_dict[e].ms_diff

                y_asq = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            res.val, x, m_f_sq.val,
                            m_f_sq_PDG.val,
                            operator, ms_diff=0,  # ms_diff.val,
                            **kwargs)
                         for x in x_asq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], x, m_f_sq.btsp[k],
                        m_f_sq_PDG.btsp[k], operator,
                        ms_diff=0,  # ms_diff.btsp[k],
                        **kwargs) for x in x_asq]
                        for k in range(N_boot)])
                )

                ax[0].plot(x_asq, y_asq.val, color=ens_colors[e])
                ax[0].fill_between(x_asq, y_asq.val+y_asq.err,
                                   y_asq.val-y_asq.err,
                                   color=ens_colors[e], alpha=0.1)

                y_mpi = stat(
                    val=np.array(
                        [chiral_continuum_ansatz(
                            res.val, a_sq.val,
                            x/(f_pi_PDG.val**2),
                            m_f_sq_PDG.val,
                            operator, ms_diff=0,  # ms_diff.val,
                            **kwargs) for x in x_msq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], a_sq.btsp[k],
                        x/(f_pi_PDG.btsp[k]**2),
                        m_f_sq_PDG.btsp[k], operator,
                        ms_diff=0,  # ms_diff.btsp[k],
                        **kwargs) for x in x_msq]
                        for k in range(N_boot)])
                )

                ax[1].plot(x_msq, y_mpi.val, color=ens_colors[e])
                ax[1].fill_between(x_msq, y_mpi.val+y_mpi.err,
                                   y_mpi.val-y_mpi.err,
                                   color=ens_colors[e], alpha=0.1)

                y_ens = y[ens_idx]

                if 'addnl_terms' in kwargs:
                    if kwargs['addnl_terms'] == 'del_ms':
                        y_ens = y_ens - res[-1]*ms_diff

                ax[0].errorbar([a_sq.val], [y_ens.val],
                               yerr=[y_ens.err],
                               xerr=[a_sq.err],
                               fmt='o', label=e,
                               capsize=4, color=ens_colors[e],
                               mfc='None', zorder=10, clip_on=False)

                mpi_phys = stat(
                    val=m_pi.val**2/a_sq.val,
                    err='fill',
                    btsp=[(m_pi.btsp[k]**2)/a_sq.btsp[k]
                          for k in range(N_boot)]
                )
                ax[1].errorbar([mpi_phys.val], [y_ens.val],
                               yerr=[y_ens.err],
                               xerr=[mpi_phys.err],
                               fmt='o', label=e, capsize=4,
                               color=ens_colors[e],
                               mfc='None')

            ax[0].errorbar([0], [y_phys.val], yerr=[y_phys.err],
                           color='k', fmt='o', capsize=4, label='phys')
            ax[1].errorbar([(m_pi_PDG**2).val], [y_phys.val],
                           xerr=[(m_pi_PDG**2).err],
                           yerr=[y_phys.err], color='k',
                           fmt='o', capsize=4, label='phys')
            ax[1].axvline(m_pi_PDG.val**2, color='k',
                          linestyle='dashed')
            ax[0].axvline(0, color='k',
                          linestyle='dashed')

            ax[0].set_xlim(x_asq[0]-0.01, x_asq[-1])
            ax[1].set_xlim(x_msq[0], x_msq[-1])

            ax[0].set_xlabel(r'$a^2$ (GeV${}^{-2}$)')
            ax[1].set_xlabel(r'$m_{\pi}^2$ (GeV${}^2$)')
            ax[legend_axis].legend(ncol=2, columnspacing=0.6)

            ax[0].text(0.4, 0.05, r'$\chi^2$/DOF:'+'{:.3f}'.format(
                res.chi_sq/res.DOF), transform=ax[0].transAxes)
            ax[1].text(0.4, 0.05, r'$p$-value:'+'{:.3f}'.format(
                res.pvalue), transform=ax[1].transAxes)

            if title != '':
                title += r', $\mu='+str(np.around(mu, 2))+'$ GeV'
            else:
                if 'custom_title' in kwargs:
                    title = kwargs['custom_title']

            plt.suptitle(title, y=0.95 if figsize == (15, 6) else 1)

            if 'filename' in kwargs:
                filename = kwargs['filename']
            else:
                filename = f'plots/{self.obj}_fits_{operator}.pdf'
            call_PDF(filename, open=open)

        return res
