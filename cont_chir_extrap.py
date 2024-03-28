import pdb
from coeffs import *
from bag_param_renorm import *
scheme = 'gamma'
#scheme = 'qslash'
expand_err = False

cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(bag_ensembles)), cmap.N)
#ens_colors = {bag_ensembles[k]: list(mc.TABLEAU_COLORS.keys())[
#    k] for k in range(len(bag_ensembles))}
ens_colors = {'C0':[1,0,0],
              'C1':[250./256,128./256,114./256],
              'C1M':[128./256,128./256,128./256],
              'C2':[240./256,190./256,128./256],
              'M0':[0,0,1.],
              'M1':[30./256,144./256,1],
              'M1M':[128./256,128./256,128./256],
              'M2':[135./256,206./256,250./256],
              'M3':[185./256,206./256,256./256],
              'F1M':[154./256,205./256,50./256]}

def ens_symbols(ens):
    if ens[0]=='C' and ens[-1]!='M':
        return "s"
    elif ens[0]=='M' and ens[-1]!='M':
        return "o"
    else:
        return "D"


CS = ['C1', 'C2']
CM = ['C0', 'C1M']

MS = ['M1', 'M2', 'M3']
MM = ['M0', 'M1M']

def mpi_dep(params, m_f_sq, op_idx, **kwargs):
    f = params[2]*m_f_sq
    if 'addnl_terms' in kwargs:
        if kwargs['addnl_terms'] == 'm4':
            f += params[3]*(m_f_sq**2)
    return f


def chiral_continuum_ansatz(params, a_sq, m_f_sq, PDG, operator,
                            obj='bag', log=False, **kwargs):
    op_idx = operators.index(operator)
    func = params[0] + params[1]*a_sq + \
        mpi_dep(params, m_f_sq, op_idx, **kwargs) - \
        mpi_dep(params, PDG, op_idx, **kwargs)
    if 'addnl_terms' in kwargs:
        if kwargs['addnl_terms'] == 'a4':
            func += params[3]*(a_sq**2)
        elif kwargs['addnl_terms'] == 'del_ms':
            func += params[3]*kwargs['ms_diff']
    if log:
        chir_log_coeffs = chiral_logs(obj=obj, **kwargs)
        chir_log_coeffs = chir_log_coeffs/((4*np.pi)**2)
        log_ratio = m_f_sq*(f_pi_PDG.val**2)/(Lambda_QCD**2)
        log_term = chir_log_coeffs[op_idx]*np.log(log_ratio)
        func += params[0]*log_term*m_f_sq

    return func


class Z_fits:
    N_ops = len(operators)

    def __init__(self, ens_list, norm='V', mask=fq_mask.copy(), **kwargs):
        self.ens_list = ens_list
        self.norm = norm
        self.mask = mask

        if norm == '11':
            self.mask[0, 0] = False

        self.Z_dict = {e: Z_analysis(e, norm=self.norm, mask=self.mask, 
                                     sea_mass_idx=1 if (e in CS+MS or e=='F1M') else 0, 
                                     **kwargs)
                       for e in bag_ensembles+['C1M', 'M1M']}

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
                CS_extrap, CS_fit_params = self.Z_chiral_extrap(CS, mu, fittype=fittype, **kwargs)
                CM_extrap, CM_fit_params = self.Z_chiral_extrap(CM, mu, fittype=fittype, **kwargs)
                MS_extrap, MS_fit_params = self.Z_chiral_extrap(MS, mu, fittype=fittype, **kwargs)
                MM_extrap, MM_fit_params = self.Z_chiral_extrap(MM, mu, fittype=fittype, **kwargs)

                for ens in self.ens_list:
                    m_pi = self.Z_dict[ens].m_pi
                    if ens in CS:
                        Z = CS_extrap
                    elif ens in MS:
                        Z = MS_extrap
                    elif ens in CM:
                        Z = CM_extrap
                    elif ens in MM:
                        Z = MM_extrap
                    elif ens == 'F1M':
                        Z0 = self.Z_dict[ens].interpolate(mu, **kwargs)

                        Z_CM = Z0 - CM_fit_params[1]*(m_pi**2)
                        Z_MM = Z0 - MM_fit_params[1]*(m_pi**2)
                        Z_CS = Z0 - CS_fit_params[1]*(m_pi**2)
                        Z_MS = Z0 - MS_fit_params[1]*(m_pi**2)

                        all_4 = [Z_CM, Z_MM, Z_CS, Z_MS]

                        stat_err = np.array([[max(z.err[i,j] for z in all_4)
                                            for j in range(self.N_ops)]
                                             for i in range(self.N_ops)])
                        sys_err = np.array([[np.abs(
                            max(z.val[i,j] for z in all_4)-min(z.val[i,j] for z in all_4))/2
                                             for j in range(self.N_ops)]
                                            for i in range(self.N_ops)])
                        all_sum = (Z_CM+Z_MM+Z_CS+Z_MS)/4
                        Z = stat(
                            val=all_sum.val,
                            err=stat_err,
                            btsp='seed',
                            seed = 1
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
                if ens=='F1M':
                    other_sys_error = np.array([[max(
                        Z_linear.sys_err[i,j], Z_quadratic.sys_err[i,j])
                                                 for j in range(self.N_ops)]
                                                for i in range(self.N_ops)])
                    sys_err = (sys_err**2 + other_sys_error**2)**0.5

                Z = stat(
                    val=(Z_linear.val+Z_quadratic.val)/2,
                    err=(stat_err**2+sys_err**2)**0.5,
                    btsp='seed',
                    seed=ensemble_seeds[ens]
                )
                Z.stat_err = stat_err
                Z.sys_err = sys_err
                extrap_assign['both'][ens] = Z

            return extrap_assign['both']

    def Z_chiral_extrap(self, ens_list, mu, plot=False,
                        extrap_label='0',
                        filename='plots/Z_chiral_extrap.pdf',
                        fittype='linear', 
                        separate=False, 
                        **kwargs):

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
                if i==0 and j==0:
                    extrap[i,j] = stat(val=1, err=0, btsp='fill')
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

        fit_params.map = mapping

        if plot:
            if not separate:
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
            else:
                for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                    if self.mask[i, j]:
                        fig, ax = plt.subplots(figsize=(2.5,2))
                        y = stat(
                            val=chiral_data.val[:, i, j],
                            err=chiral_data.err[:, i, j],
                            btsp=chiral_data.btsp[:, :, i, j])

                        ax.errorbar(x.val, y.val,
                                    yerr=y.err,
                                    xerr=x.err, fmt='o',
                                    capsize=4)
                        xmin, xmax = ax.get_xlim()
                        msq_grain = np.linspace(0.0, xmax, 50)
                        y_grain = mapping(i, j, msq_grain)
                        ax.fill_between(msq_grain,
                                        (y_grain.val-y_grain.err),
                                        (y_grain.val+y_grain.err),
                                        alpha=0.1, color='k')
                        ax.axvline(0.0, c='k', linestyle='dashed', alpha=0.1)
                        ax.errorbar([0.0], extrap.val[i, j],
                                    yerr=extrap.err[i, j],
                                    color='0.1', fmt='o', capsize=4)
                        if self.norm=='bag':
                            if i==0 and j==0:
                                denom = 'A'
                            else:
                                denom = 'S'
                            label = '$Z_{'+str(i+1)+str(j+1)+r'}/Z_'+denom+r'^2'
                            label += r'(\mu = '+str(mu)+r'\,\mathrm{GeV})$'

                        ax.set_title(label, fontsize=12)
                        ax.set_xlabel(r'$(am_\pi)^2$', fontsize=12)

                        ax_twin = ax.twiny()
                        ax_twin.set_xlim(ax.get_xlim())
                        ax_twin.set_xticks([0]+list(x.val))
                        ens_list_mod = []
                        for e in ens_list:
                            if e[-1]=='0':
                                ens_list_mod.append(e+'M')
                            elif e[-1]=='M':
                                ens_list_mod.append(e)
                            else:
                                ens_list_mod.append(e+'S')

                        ax_twin.set_xticklabels([extrap_label]+ens_list_mod)

                call_PDF(filename)

        return extrap, fit_params

class sigma:
    relevant_ensembles = ['C1', 'C0', 'M1', 'M0', 'F1M']
    N_ops = len(operators)

    def __init__(self, norm='V', mask=fq_mask.copy(), **kwargs):
        self.norm = norm
        self.mask=mask
        if self.norm == '11':
            self.mask[0, 0] = False

        self.Z_fits = Z_fits(bag_ensembles, norm=self.norm, mask=mask, **kwargs)

    def calc_running(self, mu1, mu2, plot=False, include_C=True,
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

        x = join_stats([self.Z_fits.Z_dict[e].a_sq for e in ens_list])

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
                if i==0 and j==0:
                    extrap[i, j] = stat(val=1, btsp='fill')
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

    def compare_scaling(self, mu1, mu2, npoints=11, save=True,
                        filename='running_pt_v_npt.pdf',
                        open_file=False, figsize=(5,2.5), fs=10,
                        separate=False, **kwargs):

        mu_grain = np.linspace(mu1, mu2, npoints)

        npt_filename = f'npt_scaling_{int(mu1*10)}_{int(mu2*10)}_{npoints}.p'
        if os.path.isfile(npt_filename):
            npt_scaling = pickle.load(open(npt_filename, 'rb'))
        else:
            npt_scaling = join_stats([self.calc_running(mu_grain[m_idx], mu2, **kwargs)
                                     for m_idx in tqdm(range(len(mu_grain)))])
            pickle.dump(npt_scaling, open(npt_filename, 'wb'))

        LO_scaling = np.array([expm(gamma_0*np.log(g(mu2)/g(m))/Bcoeffs(3)[0])
                      for m in mu_grain]) 
        NLO_scaling = np.array([K(mu2)@LO_scaling[m_idx]@np.linalg.inv(K(mu_grain[m_idx]))
                       for m_idx in range(len(mu_grain))])

        for i,j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i,j]:
                fig, ax = plt.subplots(figsize=figsize)

                ax.errorbar(mu_grain, npt_scaling.val[:,i,j],
                            yerr=npt_scaling.err[:,i,j], 
                            fmt='o:', capsize=4,
                            label='npt')
                ax.plot(mu_grain, LO_scaling[:,i,j],
                         '--', label='LO')
                ax.plot(mu_grain, NLO_scaling[:,i,j],
                            '-', label='NLO')
                ax.set_xlabel(r'$\mu\,\mathrm{[GeV]}$', fontsize=fs)
                ax.set_ylabel(r'$\sigma_{'+str(i+1)+str(j+1)+r'}('+\
                        str(mu2)+r'\,\mathrm{GeV}, \mu)$', fontsize=fs)
                ax.yaxis.set_ticks_position('both')
                ax.legend()
                if separate:
                    call_PDF(filename[:-4]+f'_{i+1}{j+1}'+filename[-4:], open=open_file)

        if not separate:
            call_PDF(filename, open=open_file)



class bag_fits:
    operators = operators.copy()

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
        self.Z_fits = Z_fits(self.ens_list, norm=self.norm, **kwargs)

    def load_bag(self, mu, chiral_extrap=False,
                 **kwargs):

        self.Z_dict = self.Z_fits.Z_assignment(mu, chiral_extrap=chiral_extrap,
                                          **kwargs)
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
        else:
            rot_mtx = np.eye(len(self.operators))

        rot_mtx = stat(
            val=rot_mtx,
            btsp='fill'
        )

        self.mu = mu
        self.bag = join_stats([self.Z_dict[e]@(rot_mtx@self.bag_dict[e].bag)
                               for e in self.ens_list])

    def fit_operator(self, mu, operator,
                     ens_list=None,
                     guess=[1e-1, 1e-2, 1e-3],
                     plot=False,
                     open=False,
                     rescale=False,
                     title='',
                     figsize=(15, 6),
                     legend_axis=1,
                     fs=10,
                     expand_err=False,
                     print_F1M_recons=False,
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

        # in this case ansatz loads correct x dependence
        # so just passing dummy x
        x = y
        guess = np.array(guess)

        res = fit_func(x, y, ansatz, guess, **kwargs)
        cc_coeffs = res[1:]/res[0]
        y_phys = res[0]

        if 'log' in kwargs:
            if kwargs['log']:
                log_term = chiral_logs(obj=self.obj, **kwargs)[op_idx]/((4*np.pi)**2)
                log_term = m_f_sq_PDG*((m_pi_PDG**2)/(Lambda_QCD**2)).use_func(np.log)*log_term
                y_phys = y_phys + y_phys*log_term

        if expand_err:
            a_sq_F1M = self.bag_dict['F1M'].a_sq
            F1M_recons = stat(
                    val=chiral_continuum_ansatz(res.val, a_sq_F1M.val, m_f_sq_PDG.val,
                                                m_f_sq_PDG.val, operator, ms_diff=0,
                                                obj=self.obj, **kwargs),
                    err='fill',
                    btsp=np.array([chiral_continuum_ansatz(res.btsp[k,], a_sq_F1M.btsp[k],
                                                           m_f_sq_PDG.btsp[k], m_f_sq_PDG.btsp[k],
                                                           operator, ms_diff=0, obj=self.obj, **kwargs)
                                   for k in range(N_boot)])
                    )
            if print_F1M_recons:
                print(f'{self.obj} {op_idx+1}',err_disp(F1M_recons.val, F1M_recons.err))

            cont_err = np.abs(y_phys.val-F1M_recons.val)/2
            if y_phys.err < cont_err:
                y_phys.err = cont_err
                y_phys.btsp = stat(val=y_phys.val, err=y_phys.err, btsp='fill').btsp

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
                            obj=self.obj,
                            **kwargs)
                         for x in x_asq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], x, m_f_sq.btsp[k],
                        m_f_sq_PDG.btsp[k], operator,
                        ms_diff=0,  # ms_diff.btsp[k],
                        obj=self.obj,
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
                            obj=self.obj,
                            **kwargs) for x in x_msq]),
                    err='fill',
                    btsp=np.array([[chiral_continuum_ansatz(
                        res.btsp[k, :], a_sq.btsp[k],
                        x/(f_pi_PDG.btsp[k]**2),
                        m_f_sq_PDG.btsp[k], operator,
                        ms_diff=0,  # ms_diff.btsp[k],
                        obj=self.obj,
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

                e_label = e
                if e in CM or e in MM:
                    e_label += 'M'
                elif e in CS or e in MS:
                    e_label += 'S'

                ax[0].errorbar([a_sq.val], [y_ens.val],
                               yerr=[y_ens.err],
                               xerr=[a_sq.err],
                               fmt=ens_symbols(e),
                               label=e_label,
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
                               fmt=ens_symbols(e),
                               label=e_label, capsize=4,
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
            ax[0].yaxis.set_ticks_position('both')
            ax[1].yaxis.set_ticks_position('both')

            ax[0].set_xlabel(r'$a^2\,[\mathrm{GeV}^{-2}]$', fontsize=fs)
            ax[1].set_xlabel(r'$m_{\pi}^2\,[\mathrm{GeV}^{2}]$', fontsize=fs)
            

            if self.obj=='bag':
                ylabel = r'$B' if not rescale else r'$\mathcal{B}'
            else:
                ylabel = r'$R' 
            ylabel += f'_{op_idx+1}'+r'(\mu= '+str(mu)+r'\,\mathrm{GeV})$'
            ax[0].set_ylabel(ylabel, fontsize=fs)
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

            plt.suptitle(title, y=0.95 if figsize == (15, 6) else 1, fontsize=fs)

            if 'filename' in kwargs:
                filename = kwargs['filename']
            else:
                filename = f'plots/{self.obj}_fits_{operator}.pdf'
            call_PDF(filename, open=open)

        if 'log' in kwargs:
            if kwargs['log']:
                res.val[0], res.err[0], res.btsp[:,0] = y_phys.val, y_phys.err, y_phys.btsp


        return res
