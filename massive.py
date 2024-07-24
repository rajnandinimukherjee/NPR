from NPR_classes import *
from valence import *
from coeffs import *

eta_PDG = stat(
    val=2983.9/1000,
    err=0.5/1000,
    btsp='fill')

mc_PDG = stat(
        val=1.2730,
        err=0.0046,
        btsp='fill'
        )

def closest_n_points(target, values, n, **kwargs):
    diff = np.abs(np.array(values)-np.array(target))
    sort = np.sort(diff)
    closest_idx = []
    for n_idx in range(n):
        nth_closest_point = list(diff).index(sort[n_idx])
        closest_idx.append(nth_closest_point)
    return closest_idx

class Z_bl_analysis:
    grp = 'bilinear'

    def __init__(self, ens, action=(0, 0), scheme='qslash',
                 renorm='mSMOM', **kwargs):

        self.ens = ens
        self.renorm = renorm

        a1, a2 = action
        datafile = f'{datapath}/action{a1}_action{a2}/'
        datafile += '__'.join(['NPR', self.ens, params[self.ens]['baseactions'][a1],
                              params[self.ens]['baseactions'][a2]])
        datafile += f'_{renorm}.h5'
        self.data = h5py.File(datafile, 'r')

        info = params[self.ens]
        self.ainv = stat(
            val=info['ainv'],
            err=info['ainv_err'],
            btsp='fill')

        self.all_masses = ['{:.4f}'.format(m) for m in info['safe_masses']]
        self.valence = valence(self.ens)
        self.valence.calc_all(load=False)

        self.momenta, self.Z = {}, {}

        for m in self.data.keys():
            masses = (m[2:8], m[12:18])
            self.momenta[masses] = stat(
                val=self.data[m][self.grp]['ap'][:],
                btsp='fill')

            self.Z[masses] = {}
            for key in self.data[m][self.grp].keys():
                if key != 'ap':
                    self.Z[masses][key] = stat(
                        val=self.data[m][self.grp][key]['central'][:],
                        err=self.data[m][self.grp][key]['errors'][:],
                        btsp=self.data[m][self.grp][key]['bootstrap'][:]
                    )
        df = pd.DataFrame(0, index=self.all_masses,
                          columns=['amres', 'aM', 'Z_A',
                                   'Z_m(2.0)', 'Z_m(2.5)', 'Z_m(3.0)'])
        df['amres'] = [err_disp(self.valence.amres[m].val, 
                                self.valence.amres[m].err)
                       for m in self.all_masses]
        df['aM'] = [err_disp(self.valence.eta_h_masses[m].val, 
                             self.valence.eta_h_masses[m].err)
                    for m in self.all_masses]
        df['Z_A'] = [err_disp(self.valence.Z_A[m].val, 
                              self.valence.Z_A[m].err)
                    for m in self.all_masses]
        for mu in [2.0, 2.5, 3.0]:
            df['Z_m('+str(mu)+')'] = [err_disp(self.interpolate(mu, (m,m), 'm').val,
                                       self.interpolate(mu, (m,m), 'm').err)
                         for m in self.all_masses]
        df.index.name = self.ens
        self.data_df = df

    def plot_mass_dependence(self, mu, M, Mbar,
                             key='m', start=0,
                             stop=None, open_file=True,
                             filename='', add_pdg=False,
                             pass_vals=False, key_only=False,
                             normalise=False, alt_fit=False,
                             Z_only=False,
                             **kwargs):

        if filename=='':
            filename = f'{self.renorm}_{self.ens}_vs_amq.pdf'

        if stop==None:
            stop = len(self.all_masses)
        x = join_stats([self.valence.amres[m]+eval(m)
                        for m in self.all_masses[start:stop]])
        if key_only:
            fig, ax = plt.subplots(figsize=(2,3))
            if key=='m' and self.renorm=='SMOM':
                n_pts = 4
                y = join_stats([self.interpolate(mu, (m,m), 'S')
                                for m in self.all_masses[start:stop]])**(-1)
                ax.set_ylabel(r'$Z_m(\mu='+str(mu)+r'\,\mathrm{GeV}) = 1/Z_S$')
                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err,
                            capsize=4, fmt='o')
                ax.errorbar(x.val[:n_pts], y.val[:n_pts],
                            xerr=x.err[:n_pts], yerr=y.err[:n_pts],
                            capsize=4, fmt='o', c='r')
                
                def chiral_extrap_ansatz(am, param, **kwargs):
                    return param[0] + param[1]*am
                res = fit_func(x,y,chiral_extrap_ansatz,[1,0.1],end=n_pts)
                y0 = res[0]
                ax.errorbar([0], [y0.val], yerr=[y0.err],
                            capsize=4, fmt='o', c='k',
                            label='$Z_m^\mathrm{SMOM}:$'+\
                                    err_disp(y0.val, y0.err))
                ymin, ymax = ax.get_ylim()
                ax.vlines(x=0, ymin=ymin, ymax=ymax,
                          color='k', linestyle='dashed')

                xmin, xmax = ax.get_xlim()
                xrange = np.linspace(xmin, x.val[n_pts-1], 100)
                yrange = res.mapping(xrange)
                ax.fill_between(xrange, yrange.val+yrange.err,
                                yrange.val-yrange.err,
                                color='k', alpha=0.1)
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.legend(frameon=False)
                ax.set_xlabel(r'$am_q+am_\mathrm{res}$')
                
                if pass_vals:
                    plt.close(fig)
                    return y0
                else:
                    call_PDF(filename, open=open_file)
                
            else:
                y = join_stats([self.interpolate(mu, (m,m), key)
                                for m in self.all_masses[start:stop]])
                ax.set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+r'\,\mathrm{GeV},\, am_h)$')
                if normalise:
                    q = join_stats([self.interpolate(mu, (m,m), 'q')
                                    for m in self.all_masses[start:stop]])
                    y = y/q
                    ax.set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+r'\,\mathrm{GeV})/Z_q$')

                ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err,
                            capsize=4, fmt='o')
                ax.text(0.5, 0.1, self.ens,
                           va='center', ha='center',
                           transform=ax.transAxes)
                ax.set_xlabel(r'$am_h$')
                call_PDF(filename, open=open_file)

        else:
            if stop==None:
                stop = len(self.all_masses)

            nrows = 2 if Z_only else 3
            figsize = (3,7.5) if Z_only else (3,10)
            fig, ax = plt.subplots(nrows=nrows, ncols=1, 
                                   sharex='col',
                                   figsize=figsize)
            plt.subplots_adjust(hspace=0, wspace=0)

            y = join_stats([self.valence.eta_h_masses[m]*self.ainv
                            for m in self.all_masses[start:stop]])
            def eta_mass_ansatz(am, param, **kwargs):
                return param[0] + param[1]*am**0.5 + param[2]*am
            ax[0].text(1.05, 0.5,
                       r'$\alpha + \beta\sqrt{am_q} + \gamma\,am_q$',
                       va='center', ha='center', rotation=90,
                       color='k', alpha=0.3,
                       transform=ax[0].transAxes)

            res = fit_func(x, y, eta_mass_ansatz, [0.1,1,1])
            def pred_amq(eta_mass, param, **kwargs):
                a, b, c = param
                root = (-b+(b**2 + 4*c*(eta_mass-a))**0.5)/(2*c)
                return root**2
            am_c = stat(
                    val=pred_amq(M.val, res.val),
                    err='fill',
                    btsp=[pred_amq(M.btsp[k], res.btsp[k])
                          for k in range(N_boot)]
                    )
            am_star = stat(
                       val=pred_amq(Mbar.val, res.val),
                       err='fill',
                       btsp=[pred_amq(Mbar.btsp[k], res.btsp[k])
                             for k in range(N_boot)]
                       )

            if alt_fit:
                variations, names = self.fit_variations(y, x, M)
                am_c = stat(
                        val=variations[2].val,
                        err=(max([var.err for var in variations])**2 +\
                                np.abs((max(var.val for var in variations)-\
                                min(var.val for var in variations))/2)**2)**0.5,
                        btsp='fill'
                        )
                variations, names = self.fit_variations(y, x, Mbar)
                am_star = stat(
                        val=variations[2].val,
                        err=(max([var.err for var in variations])**2 +\
                                np.abs((max(var.val for var in variations)-\
                                min(var.val for var in variations))/2)**2)**0.5,
                        btsp='fill'
                        )

            ax[0].errorbar(x.val, y.val, yerr=y.err,
                        capsize=4, fmt='o')
            if add_pdg:
                ymin, ymax = ax[0].get_ylim()
                ax[0].hlines(y=M.val, xmin=0, xmax=am_c.val, color='k')
                ax[0].text(0.05, M.val-0.05, r'$M$',
                           va='top', ha='center', color='k')
                ax[0].fill_between(np.linspace(0,am_c.val,100),
                                   M.val+M.err,
                                   M.val-M.err,
                                   color='k', alpha=0.2)
                ax[0].vlines(x=am_c.val, ymin=ymin, ymax=M.val,
                             color='k', linestyle='dashed')
                ax[0].text(am_c.val, 0.5, r'$am_c$', rotation=90,
                           va='center', ha='right', color='k')
                ax[0].fill_between(np.linspace(am_c.val-am_c.err,
                                               am_c.val+am_c.err,100),
                                   ymin, M.val+M.err,
                                   color='k', alpha=0.2)

                ax[0].hlines(y=Mbar.val, xmin=0, xmax=am_star.val, color='r')
                ax[0].text(0.05, Mbar.val-0.05, r'$\overline{M}$',
                           va='top', ha='center', color='r')
                ax[0].fill_between(np.linspace(0,am_star.val,100),
                                   Mbar.val+Mbar.err,
                                   Mbar.val-Mbar.err,
                                   color='r', alpha=0.2)
                ax[0].vlines(x=am_star.val, ymin=ymin, ymax=Mbar.val,
                             color='r', linestyle='dashed')
                ax[0].text(am_star.val, 0.5, r'$am^\star$', rotation=90,
                           va='center', ha='right', color='r')
                ax[0].fill_between(np.linspace(am_star.val-am_star.err,
                                               am_star.val+am_star.err,100),
                                   ymin, Mbar.val+Mbar.err,
                                   color='r', alpha=0.2)

                discard, new_ymax = ax[0].get_ylim()
                ax[0].set_ylim([ymin, new_ymax])

            xmin, xmax = ax[0].get_xlim()
            xrange = np.linspace(x.val[0], xmax, 100)
            yrange = res.mapping(xrange)
            ax[0].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                    str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                    ha='center', va='center',
                    transform=ax[0].transAxes)
            ax[0].fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color='k', alpha=0.1)
            ax[0].set_ylabel(r'$M_{\eta_h}\,[\mathrm{GeV}]$')
            ax[0].set_xlim(0, xmax)
            ax[0].set_title(f'{self.ens}')



            y = join_stats([self.interpolate(mu, (m,m), key)
                            for m in self.all_masses[start:stop]])
            if not Z_only:
                y = y*am_c*self.ainv

            def Z_m_ansatz(am, param, **kwargs):
                return param[0]/am + param[1] + param[2]*am + param[3]*am**2
            ax[1].text(1.05, 0.5,
                       r'$\alpha/am_q + \beta + \gamma\,am_q + \delta\,(am_q)^2$',
                       va='center', ha='center', rotation=90,
                       color='k', alpha=0.3,
                       transform=ax[1].transAxes)

            res = fit_func(x, y, Z_m_ansatz, [0.1,0.1,0.1,0.1],
                           pause=False, verbose=False)
            Z_m_amstar = res.mapping(am_star)

            ax[1].errorbar(x.val, y.val, xerr=x.err,
                           yerr=y.err, capsize=4, fmt='o')
            if add_pdg:
                ymin, ymax = ax[1].get_ylim()
                ax[1].errorbar([am_star.val], [Z_m_amstar.val],
                               xerr=[am_star.err], yerr=[Z_m_amstar.err],
                               fmt='o', capsize=4, color='r')
                ax[1].vlines(x=am_star.val, ymin=ymin, ymax=ymax,
                             color='r', linestyle='dashed')
                ax[1].fill_between(np.linspace(am_star.val-am_star.err,
                                               am_star.val+am_star.err,100),
                                   ymin, Mbar.val+Mbar.err,
                                   color='r', alpha=0.2)

                ax[1].set_ylim([ymin, ymax])

            xmin, xmax = ax[1].get_xlim()
            xrange = np.linspace(x.val[0], xmax, 100)
            yrange = res.mapping(xrange)
            ax[1].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                    str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                    ha='center', va='center',
                    transform=ax[1].transAxes)
            ax[1].fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color='k', alpha=0.1)

            SMOM = Z_bl_analysis(self.ens, renorm='SMOM')
            Z_m_amc_SMOM = SMOM.plot_mass_dependence(mu, M, Mbar,
                                                 key=key, stop=stop,
                                                 open_file=False,
                                                 key_only=True,
                                                 pass_vals=True)
            if not Z_only:
                Z_m_amc_SMOM = Z_m_amc_SMOM*am_c*self.ainv
            ax[1].axhspan(Z_m_amc_SMOM.val+Z_m_amc_SMOM.err,
                          Z_m_amc_SMOM.val-Z_m_amc_SMOM.err,
                          color='r', alpha=0.1)

            label = r'\,\mathrm{GeV})$' if Z_only else\
                    r'\,\mathrm{GeV})\cdot am_c\cdot a^{-1}\, [\mathrm{GeV}]$'
            ax[1].set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+label)
            ax[1].set_xlim(0, xmax)

            if alt_fit:
                variations, names = self.fit_variations(x, y, am_star)
                variations = join_stats([Z_m_amstar]+variations)
                names = ['all-pts']+names
                fit_vals = [fit.val for fit in variations]
                stat_err = max([fit.err for fit in variations])
                sys_err = (max(fit_vals)-min(fit_vals))/2
                Z_m_amstar = stat(
                        val=fit_vals[2],
                        err=(stat_err**2+sys_err**2)**0.5,
                        btsp='fill'
                        )
                var = inset_axes(ax[1], width="40%", height="15%", loc=10)
                var.errorbar(np.arange(1,len(names)+1), variations.val,
                             yerr=variations.err, fmt='o', capsize=4,
                             color='k')
                var.tick_params(axis='y', labelsize=8)
                var.set_xticks(np.arange(1,len(names)+1))
                var.set_xticklabels(names, rotation=90, fontsize=8)

            if Z_only:
                ax[1].set_xlabel(r'$am_q+am_\mathrm{res}$')
                Z_m_amstar = Z_m_amstar*am_c*self.ainv
                mbar = Z_m_amstar*am_star*self.ainv

            if not Z_only:
                y = (x*y)/am_c
                def Z_m_m_q_ansatz(am, param, **kwargs):
                    return param[0]*am + param[1] + param[2]*am**2 + param[3]*am**3
                ax[2].text(1.05, 0.5,
                        r'$\alpha+\beta\,am_q+\gamma\,(am_q)^2 +\delta\,(am_q)^3$',
                           va='center', ha='center', rotation=90,
                           color='k', alpha=0.3,
                           transform=ax[2].transAxes)

                res = fit_func(x, y, Z_m_m_q_ansatz, [1,1e-1,1e-1,1e-2],
                               verbose=False)
                mbar = res.mapping(am_star) 

                ax[2].errorbar(x.val, y.val, xerr=x.err,
                               yerr=y.err,
                            capsize=4, fmt='o')
                if add_pdg:
                    ymin, ymax = ax[2].get_ylim()
                    ax[2].errorbar([am_star.val], [mbar.val],
                                   xerr=[am_c.err], yerr=[mbar.err],
                                   fmt='o', capsize=4, color='r')
                    ax[2].vlines(x=am_star.val, ymin=ymin, ymax=ymax,
                                 color='r', linestyle='dashed')
                    ax[2].fill_between(np.linspace(am_star.val-am_star.err,
                                                   am_star.val+am_star.err,100),
                                       ymin, Mbar.val+Mbar.err,
                                       color='r', alpha=0.2)

                    ax[2].set_ylim([ymin, ymax])
                xmin, xmax = ax[2].get_xlim()
                xrange = np.linspace(x.val[0], xmax, 100)
                yrange = res.mapping(xrange)
                ax[2].text(0.5, 0.05, r'$\chi^2/\mathrm{DOF}:'+\
                        str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                        ha='center', va='center',
                        transform=ax[2].transAxes)
                ax[2].fill_between(xrange, yrange.val+yrange.err,
                                yrange.val-yrange.err,
                                color='k', alpha=0.1)
                ax[2].set_ylabel(r'$Z_'+key+'(\mu='+str(mu)+\
                        '\,\mathrm{GeV})\cdot am_q\cdot a^{-1}\, [\mathrm{GeV}]$')
                ax[2].set_xlabel(r'$am_q+am_\mathrm{res}$')

                variations, names = self.fit_variations(x, y, am_star)
                variations = join_stats([mbar]+variations)
                names = ['all-pts']+names
                if alt_fit:
                    fit_vals = [fit.val for fit in variations]
                    stat_err = max([fit.err for fit in variations])
                    sys_err = (max(fit_vals)-min(fit_vals))/2
                    mbar = stat(
                            val=np.mean(fit_vals),
                            err=(stat_err**2+sys_err**2)**0.5,
                            btsp='fill'
                            )

                    var = inset_axes(ax[2], width="40%", height="15%", loc=2)
                    var.errorbar(np.arange(1,len(names)+1), variations.val,
                                 yerr=variations.err, fmt='o', capsize=4,
                                 color='k')
                    var.tick_params(axis='y', labelsize=8)
                    var.set_xticks(np.arange(1,len(names)+1))
                    var.set_xticklabels(names, rotation=90, fontsize=8)
                    var.yaxis.tick_right()

            if pass_vals:
                plt.close(fig)
                return Z_m_amstar, Z_m_amc_SMOM, mbar
            else:
                call_PDF(filename, open=open_file)

    def fit_variations(self, x, y, am_star, **kwargs):
        def two_pt_ansatz(am, param, **kwargs):
            return param[0] + param[1]*am
        indices = np.sort(closest_n_points(
            am_star.val, x.val, n=2))
        res = fit_func(x[indices], y[indices],
                       two_pt_ansatz, [1,1e-1])
        fit_2 = res.mapping(am_star)

        def three_pt_ansatz(am, param, **kwargs):
            return param[0] + param[1]*am + param[2]*(am**2)
        indices = np.sort(closest_n_points(
            am_star.val, x.val, n=3))
        res = fit_func(x[indices], y[indices],
                       three_pt_ansatz, [1,1e-1,1e-2])
        fit_3 = res.mapping(am_star)
        farthest_point = indices[-1]

        indices = list(np.sort(closest_n_points(
            am_star.val, x.val, n=4)))
        indices.remove(farthest_point)
        res = fit_func(x[indices], y[indices],
                       three_pt_ansatz, [1,1e-1,1e-2])
        fit_3_alt = res.mapping(am_star)

        def four_pt_ansatz(am, param, **kwargs):
            return param[0] + param[1]*am +\
                    param[2]*(am**2) + param[3]*(am**3)
        indices = np.sort(closest_n_points(
            am_star.val, x.val, n=4))
        res = fit_func(x[indices], y[indices],
                       four_pt_ansatz, [1,1e-1,1e-2,1e-3])
        fit_4 = res.mapping(am_star)
        farthest_point = indices[-1]

        indices = list(np.sort(closest_n_points(
            am_star.val, x.val, n=5)))
        indices.remove(farthest_point)
        res = fit_func(x[indices], y[indices],
                       four_pt_ansatz, [1,1e-1,1e-2,1e-3])
        fit_4_alt = res.mapping(am_star)

        fits = [fit_2, fit_3, fit_3_alt, fit_4]#, fit_4_alt]
        names = ['2-pt', '3-pt', '3-pt(alt)', '4-pt']#, '4-pt(alt)']
        return fits, names

    def interpolate(self, mu, masses, key,
                    plot=False, pass_plot=False,
                    axis=None, limit=None, **kwargs):

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        linear_Z = self.fit_momentum_dependence(mu,
                masses, key, fittype='linear', **kwargs)
        quadratic_Z = self.fit_momentum_dependence(mu,
                masses, key, fittype='quadratic', **kwargs)

        stat_err = max(linear_Z.err, quadratic_Z.err)
        sys_err = np.abs(quadratic_Z.val-linear_Z.val)/2
        Z_mu = stat(
                val=(linear_Z.val+quadratic_Z.val)/2,
                err=(stat_err**2+sys_err**2)**0.5,
                btsp='fill'
                )

        if plot:
            if pass_plot:
                ax = axis
            else:
                fig, ax = plt.subplots(figsize=(2,3))

            ax.errorbar([mu], Z_mu.val, yerr=Z_mu.err,
                         capsize=4, fmt='o', color='k')
            if pass_plot:
                return Z_mu, plt
            else:
                if limit==None:
                    limit = x.val[-1]

                plot_idx = np.where(x.val<=limit)[0]
                ax.errorbar(x.val[plot_idx], y.val[plot_idx],
                            xerr=x.err[plot_idx], yerr=y.err[plot_idx],
                            capsize=4, fmt='o')
                ax.text(0.5, 0.1, self.ens,
                           va='center', ha='center',
                           transform=ax.transAxes)
                ax.set_xlabel(r'$\sqrt{p^2}\,[\mathrm{GeV}]$')
                ax.set_ylabel(r'$Z_{'+key+r'}(a|p|, am_h='+masses[0]+r')$')
                ymin, ymax = ax.get_ylim()
                ax.vlines(x=mu, ymin=ymin, ymax=ymax,
                             color='k', linestyle='dashed')
                ax.set_ylim([ymin, ymax])

                filename = f'plots/Z_{key}_v_ap.pdf'
                call_PDF(filename)

        return Z_mu

    def fit_momentum_dependence(self, mu, masses, key,
                                plot=False, fittype='quadratic',
                                pass_plot=False, normalise=False,
                                **kwargs):

        x = self.momenta[masses]*self.ainv
        y = self.Z[masses][key]

        if fittype == 'linear':
            indices = np.sort(
                closest_n_points(mu, x.val, n=2))
            x_1, y_1 = x[indices[0]], y[indices[0]]
            x_2, y_2 = x[indices[1]], y[indices[1]]
            slope = (y_2-y_1)/(x_2-x_1)
            intercept = y_1 - slope*x_1

            pred = intercept + slope*mu

        elif fittype == 'quadratic':
            indices = np.sort(
                closest_n_points(mu, x.val, n=3))
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

            pred = a*(mu**2) + b*mu + c

        return pred

    def plot_momentum_dependence(self, masses, key, x_sq=False,
                                 pass_fig=False, normalise=False,
                                 **kwargs):
        plt.figure()

        x = self.momenta[masses]*self.ainv
        if x_sq:
            x = x**2
        y = self.Z[masses][key]
        if normalise:
            y = y/self.Z[masses]['q']

        plt.errorbar(x.val, y.val,
                     xerr=x.err, yerr=y.err,
                     capsize=4, fmt='o:')
        xlabel = r'$\mu^2\, \mathrm{[GeV]}^2$' if x_sq else r'$\mu$ (GeV)'
        plt.xlabel(xlabel)
        ylabel = r'$Z_'+key+r'/Z_q$' if normalise else r'$Z_'+key+r'$'
        plt.ylabel(ylabel)

        if pass_fig:
            return plt
        else:
            filename = f'plots/Z_{key}_v_ap.pdf'
            call_PDF(filename)

    def save_data(self, filename='mNPR_data_for_tables.h5', **kwargs):
        file = h5py.File(filename, 'a')
        name = self.ens+'S' if self.ens[-1]=='1' else self.ens
        if name in file:
            del file[name]

        grp = file.create_group(name)
        am_q = grp.create_dataset('am_q', data=self.all_masses)

        amres = join_stats([self.valence.amres[m] for m in self.all_masses])
        grp.create_dataset('am_res/central', data=amres.val)
        grp.create_dataset('am_res/error', data=amres.err)
        grp.create_dataset('am_res/Bootstraps', data=amres.btsp)

        aM = join_stats([self.valence.eta_h_masses[m] for m in self.all_masses])
        grp.create_dataset('aM/central', data=aM.val)
        grp.create_dataset('aM/error', data=aM.err)
        grp.create_dataset('aM/Bootstraps', data=aM.btsp)

        Z_A = join_stats([self.valence.Z_A[m] for m in self.all_masses])
        grp.create_dataset('Z_A/central', data=Z_A.val)
        grp.create_dataset('Z_A/error', data=Z_A.err)
        grp.create_dataset('Z_A/Bootstraps', data=Z_A.btsp)

        SMOM_data = Z_bl_analysis(self.ens, renorm='SMOM')

        for mu in [2.0, 2.5, 3.0]:
            Z = join_stats([self.interpolate(mu, (m,m), 'm')
                                 for m in self.all_masses])
            grp.create_dataset(f'mSMOM/Z_m({str(mu)})/central', data=Z.val)
            grp.create_dataset(f'mSMOM/Z_m({str(mu)})/error', data=Z.err)
            grp.create_dataset(f'mSMOM/Z_m({str(mu)})/Bootstraps', data=Z.btsp)


            Z = join_stats([self.interpolate(mu, (m,m), 'S')
                            for m in self.all_masses[:4]])**(-1)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/central', data=Z.val)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/error', data=Z.err)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/Bootstraps', data=Z.btsp)

            SMOM_fit = SMOM_data.plot_mass_dependence(
                    mu,1,1,pass_vals=True,key_only=True)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/extrap/central', data=SMOM_fit.val)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/extrap/error', data=SMOM_fit.err)
            grp.create_dataset(f'SMOM/Z_m({str(mu)})/extrap/Bootstraps', data=SMOM_fit.btsp)

        file.close()
        print(f'Saved {self.ens} data to {filename}')

    def print_ens_Z_table(self, key='m', mus=[2.0, 2.5, 3.0], stop=None, **kwarsg):
        if stop==None:
            stop = len(self.all_masses)
        am = join_stats([self.valence.amres[m]+eval(m)
                         for m in self.all_masses[:stop]])
        N_mu = len(mus)

        rv = [r'\begin{tabular}{c|'+''.join(['c']*N_mu)+r'}']
        rv += [r'\hline']
        rv += [r'\hline']

        rv += [r'$am_h$ & '+' & '.join([r'$Z_{'+key+'}^\mathrm{'+self.renorm+\
                r'}('+str(m)+r'\,\mathrm{GeV})$' for m in mus])+r'\\']
        rv += [r'\hline']
        for idx, am in enumerate(am):
            if key=='m' and self.renorm=='SMOM':
                Zs = join_stats([self.interpolate(
                    mu, (self.all_masses[idx],self.all_masses[idx]), 'S')
                                for mu in mus])**(-1)
            else:
                Zs = join_stats([self.interpolate(
                    mu, (self.all_masses[idx],self.all_masses[idx]), key)
                                for mu in mus])
            rv += [r'$'+'$ & $'.join([err_disp(am.val, am.err)]+\
                    [err_disp(z.val, z.err) for z in Zs])+r'$ \\']

        rv += [r'\hline']
        rv += [r'\hline']
        rv += [r'\end{tabular}']

        ens = self.ens+'S' if self.ens[-1]=='1' else self.ens
        filename = f'/Users/rajnandinimukherjee/PhD/thesis/inputs/MassiveNPR/'+\
                f'tables/{ens}_Z_{key}_{self.renorm}.tex'

        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z_{key} table written to {filename}')

class cont_extrap:
    def __init__(self, ens_list):
        self.ens_list = ens_list
        self.mSMOM_dict = {ens: Z_bl_analysis(ens, renorm='mSMOM')
                          for ens in ens_list}
        self.SMOM_dict = {ens: Z_bl_analysis(ens, renorm='SMOM')
                          for ens in ens_list}

        self.a_sq = join_stats([val.ainv**(-2)
                                for key, val in self.mSMOM_dict.items()])
        self.extrap_values = {}
        if ens_list == ['F1S', 'F1M', 'M1', 'M1M', 'C1', 'C1M']:
            self.no_C1S = cont_extrap(['F1S', 'F1M', 'M1', 'M1M', 'C1M'])
            self.M_only = cont_extrap(['F1M', 'M1M', 'C1M'])
            self.S_only = cont_extrap(['F1S', 'M1', 'C1'])

    def load_renorm_masses(self, mu, M, Mbar, **kwargs):
        m_c_ren_mSMOM = []
        m_c_ren_SMOM = []
        m_bar = []

        for ens in self.ens_list:
            quantities = self.mSMOM_dict[ens].plot_mass_dependence(
                    mu, M, Mbar, key='m', 
                    open_file=False, pass_vals=True, **kwargs)

            m_c_ren_mSMOM.append(quantities[0])
            m_c_ren_SMOM.append(quantities[1])
            m_bar.append(quantities[2])

        return join_stats(m_c_ren_mSMOM), join_stats(m_c_ren_SMOM),\
                join_stats(m_bar)

    def plot_cont_extrap(self, mu, M, Mbar, with_amres=False, 
                         M_label='', Mbar_label='', quad=False,
                         pass_axis=None, axis=None, plot=True, plot_amres=True,
                         plot_SMOM=True, filename='', open_file=True,
                         verbose=False, mSMOM_color='tab:blue', 
                         SMOM_color='tab:orange', mSMOM_label='',
                         amres_invert=False, phys_color=None, **kwargs):

        ansatz, guess = self.ansatz(choose='quadratic' if quad else 'linear', **kwargs)
        if M.val>eta_PDG.val*0.75 or Mbar.val>eta_PDG.val*0.75:
            if M.val>eta_PDG.val*0.81 or Mbar.val>eta_PDG.val*0.81:
                stop = -2 if len(self.ens_list)>3 else -1
            else:
                stop = -1
        else:
            stop = len(self.ens_list) 
        amres = join_stats(
                [self.mSMOM_dict[ens].valence.interpolate_amres(M)
                 for ens in self.ens_list[:stop]])

        if with_amres and len(self.ens_list)>3:
            def ansatz(asq, param, **kw):
                if kw['fit']=='central':
                    am = amres.val
                elif kw['fit']=='recon':
                    am = asq*0
                else:
                    am = amres.btsp[kw['k'],:]
                    
                if quad:
                    return param[0] + param[1]*asq + param[2]*asq**2 + param[3]*am
                else:
                    return param[0] + param[1]*asq + param[2]*am

            ansatz_name = r'$\alpha+\beta\,a^2+\gamma\,am_\mathrm{res}(M)$'
            guess.append(1e-1)
            if quad:
                guess.append(1e-1)
                ansatz_name = r'$\alpha+\beta\,a^2+\gamma\,a^4+'+\
                        r'\delta\,am_\mathrm{res}(M)$'
        else:
            ansatz_name = r'$\alpha+\beta\,a^2+\gamma\,a^4$' if quad else r'$\alpha+\beta\,a^2$'


        x = self.a_sq
        y_mc_mSMOM, y_mc_SMOM, y_mbar = self.load_renorm_masses(
                mu, M, Mbar)

        res_mbar = fit_func(x, y_mbar, ansatz, guess, end=stop)
        y_mbar_phys = res_mbar[0]
        res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess, end=stop)
        if verbose:
            print('mSMOM: '+' '.join([err_disp(f.val, f.err) for f in res_mc_mSMOM]))
        y_mc_mSMOM_phys = res_mc_mSMOM[0]
        res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess, end=stop)
        if verbose:
            print('SMOM: '+' '.join([err_disp(f.val, f.err) for f in res_mc_SMOM]))
        y_mc_SMOM_phys = res_mc_SMOM[0]


        if not pass_axis:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(4,3))
        else:
            ax = axis

        plt.subplots_adjust(hspace=0, wspace=0)

        amres = join_stats(
                [self.mSMOM_dict[ens].valence.interpolate_amres(M)
                 for ens in self.ens_list])
        if with_amres:
            if plot_amres:
                ax.errorbar(x.val, y_mc_mSMOM.val,
                            xerr=x.err, yerr=y_mc_mSMOM.err,
                            fmt='o', capsize=4, 
                            color=mSMOM_color,
                            mfc='None', alpha=1 if amres_invert else 0.3)

            y_mc_mSMOM = y_mc_mSMOM-amres.val*res_mc_mSMOM[-1].val

        cDOF = '{:.3f}'.format(res_mc_mSMOM.chi_sq/res_mc_mSMOM.DOF)
        pval = '{:.2f}'.format(res_mc_mSMOM.pvalue)
        label = r'mSMOM' if mSMOM_label=='' else mSMOM_label
        label += '($p$-val:'+pval+')' 
        ax.errorbar(x.val, y_mc_mSMOM.val,
                    xerr=x.err, yerr=y_mc_mSMOM.err,
                    fmt='o', mfc='None', capsize=4, 
                    color=phys_color if amres_invert else mSMOM_color,
                    alpha=0.3 if amres_invert else 1)

        xmin, xmax = ax.get_xlim()
        if stop!=len(self.ens_list):
            xrange = np.linspace(0, x[stop-1].val)
        else:
            xrange = np.linspace(0, xmax)

        if phys_color==None:
            phys_color=mSMOM_color
        yrange = res_mc_mSMOM.mapping(xrange)
        ax.errorbar([0.0], [y_mc_mSMOM_phys.val],
                    xerr=[0.0],
                    yerr=[y_mc_mSMOM_phys.err],
                    capsize=4, fmt='o', mfc='None', 
                    color=phys_color, label=label)
        ax.fill_between(xrange, yrange.val+yrange.err,
                        yrange.val-yrange.err,
                        color=phys_color, alpha=0.1)


        if with_amres:
            if plot_SMOM and plot_amres:
                ax.errorbar(x.val, y_mc_SMOM.val, 
                            xerr=x.err, yerr=y_mc_SMOM.err,
                            fmt='d', capsize=4, color=SMOM_color,
                            mfc='None', alpha=0.3)
            y_mc_SMOM = y_mc_SMOM-amres.val*res_mc_SMOM[-1].val

        if plot_SMOM:
            cDOF = '{:.3f}'.format(res_mc_SMOM.chi_sq/res_mc_SMOM.DOF)
            pval = '{:.2f}'.format(res_mc_SMOM.pvalue)
            label = r'SMOM ($p$-val:'+pval+')'
            ax.errorbar(x.val, y_mc_SMOM.val, 
                        xerr=x.err, yerr=y_mc_SMOM.err,
                        fmt='d', capsize=4, mfc='None',
                        color=SMOM_color,
                        label=label)

            yrange = res_mc_SMOM.mapping(xrange)
            ax.errorbar([0.0], [y_mc_SMOM_phys.val],
                        yerr=[y_mc_SMOM_phys.err], mfc='None',
                        capsize=4, fmt='d', color=SMOM_color)
            ax.fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color=SMOM_color, alpha=0.1)

        label1, label2 = r'$M=$', r'$\overline{M}=$'
        label1 += err_disp(M.val, M.err) if\
                M_label=='' else M_label
        label2 += err_disp(Mbar.val, Mbar.err) if\
                Mbar_label=='' else Mbar_label
        ax.set_title(label1+r', '+label2)
        ax.set_ylabel(r'$m_R(\mu='+str(mu)+r'\,\mathrm{GeV}, \overline{m}_R)\, [\mathrm{GeV}]$')
        ax.set_xlabel(r'$a^2\,[\mathrm{GeV}^{-2}]$')
        #ax.text(1.05, 0.5, ansatz_name,
        #           va='center', ha='center', rotation=90,
        #           color='k', alpha=0.3,
        #           transform=ax.transAxes)

        ymin, ymax = ax.get_ylim()
        ax.vlines(x=0.0, ymin=ymin, ymax=ymax,
                     color='k', linestyle='dashed')
        ax.set_ylim([ymin, ymax])
        ax.legend(fontsize=10)

        if plot:
            if pass_axis:
                return {'y_mc_mSMOM': y_mc_mSMOM,
                        'res_mc_mSMOM': res_mc_mSMOM,
                        'y_mc_SMOM': y_mc_SMOM,
                        'res_mc_SMOM': res_mc_SMOM,
                        'xrange': xrange,
                        'end':stop}
            else:
                if filename=='':
                    filename = '_'.join(self.ens_list)+'_cont_extrap.pdf'
                call_PDF(filename, open=open_file)
        else:
            plt.close(fig)
            return {'y_mc_mSMOM': y_mc_mSMOM,
                    'res_mc_mSMOM': res_mc_mSMOM,
                    'y_mc_SMOM': y_mc_SMOM,
                    'res_mc_SMOM': res_mc_SMOM,
                    'xrange': xrange,
                    'end':stop}

    def plot_M_variations(self, mu, M_list, Mbar, data='mSMOM',
                          name_list=[], Mbar_label='', xticks=[],
                          filename='', verbose=False, 
                          ranges=[(0,None)], pass_only=False,
                          range_names = ['(0.6,1.0)'],
                          **kwargs):
        x = self.a_sq
        figure, ax = plt.subplots(nrows=2, ncols=1,
                               sharex='col',
                               figsize=(3,5.5),
                               gridspec_kw={'height_ratios':[2,1]})
        plt.subplots_adjust(hspace=0, wspace=0)
        y_phys_mSMOM = []
        y_phys_SMOM = []
        for M_idx in tqdm(range(len(M_list)), leave=False):
            M = M_list[M_idx]
            color = color_list[M_idx]
            y_mc_mSMOM_phys, y_mc_SMOM_phys, fit_dict = self.CL_variations(
                    mu, M, Mbar, plot=False, fit_all=True, **kwargs)

            end = fit_dict['end']
            y_mc_mSMOM = fit_dict['y_mc_mSMOM'][:end]
            if len(name_list)!=len(M_list):
                label = r'$M=$'+err_disp(M.val, M.err)
            else:
                label = r'$M='+name_list[M_idx]+r'\,M_{\eta_c}^\mathrm{PDG}$'
            ax[0].errorbar(x.val[:end], y_mc_mSMOM.val, 
                           xerr=x.err[:end], yerr=y_mc_mSMOM.err,
                           fmt='o', capsize=4, color=color, 
                           mfc='None', label=label)

            y_phys_mSMOM.append(y_mc_mSMOM_phys)

            xrange = fit_dict['xrange']
            yrange = fit_dict['res_mc_mSMOM'].mapping(xrange)

            ax[0].errorbar([0.0], [y_mc_mSMOM_phys.val],
                           yerr=[y_mc_mSMOM_phys.err],
                           capsize=4, fmt='o', color=color)
            ax[0].fill_between(xrange, yrange.val+yrange.err,
                               yrange.val-yrange.err,
                               color=color, alpha=0.2)

            y_mc_SMOM = fit_dict['y_mc_SMOM'][:end]
            ax[0].errorbar(x.val[:end], y_mc_SMOM.val, 
                           xerr=x.err[:end], yerr=y_mc_SMOM.err,
                           fmt='d', capsize=4, color=color,
                           mfc='None')
            y_phys_SMOM.append(y_mc_SMOM_phys)

            yrange = fit_dict['res_mc_SMOM'].mapping(xrange)
            ax[0].errorbar([0.0], [y_mc_SMOM_phys.val],
                           yerr=[y_mc_SMOM_phys.err],
                           capsize=4, fmt='d', color=color)
            ax[0].fill_between(xrange, yrange.val+yrange.err,
                               yrange.val-yrange.err,
                               color=color, alpha=0.1)

        y_mbar_phys, discard, mbar_dict = self.CL_variations(
                mu, Mbar, Mbar, plot=False, fit_all=True, **kwargs)

        end = mbar_dict['end']
        xrange = mbar_dict['xrange']
        xmax = xrange[-1]

        y_mbar = mbar_dict['y_mc_mSMOM'][:end]
        ax[1].errorbar(x.val[:end], y_mbar.val, 
                       xerr=x.err[:end], yerr=y_mbar.err,
                       fmt='o', capsize=4, color='k',
                       mfc='None', label=label)

        yrange = mbar_dict['res_mc_mSMOM'].mapping(xrange)
        ax[1].errorbar([0.0], [y_mbar_phys.val],
                       yerr=[y_mbar_phys.err],
                       capsize=4, fmt='o', color='k')
        ax[1].fill_between(xrange, yrange.val+yrange.err,
                           yrange.val-yrange.err,
                           color='k', alpha=0.2)
        ax[1].set_xlim([-0.02, xmax])

        if Mbar_label=='':
            Mbar_label = r'$\overline{M}:'+err_disp(Mbar.val, Mbar.err)+r'$'
        ax[0].set_title(Mbar_label)
        ax[0].set_ylabel(r'$m^R(\mu='+str(mu)+r'\,\mathrm{GeV}, \overline{m})$')
        ax[1].set_ylabel(r'$\overline{m}(\mu='+str(mu)+r'\,\mathrm{GeV})$')
        ax[1].set_xlabel(r'$a^2\,[\mathrm{GeV}^{-2}]$')
        ax[0].text(-0.05, 1.0, r'[GeV]',
                va='bottom', ha='right',
                transform=ax[0].transAxes)

        ax[0].legend(bbox_to_anchor=(1, 0.4))

        for idx in range(2):
            ymin, ymax = ax[idx].get_ylim()
            ax[idx].set_ylim([ymin, ymax])
            ax[idx].vlines(x=0.0, ymin=ymin, ymax=ymax,
                         color='k', linestyle='dashed')
            ax[idx].set_ylim([ymin, ymax])

        if pass_only:
            plt.close(figure)
        else:
            filename_ = '_'.join(self.ens_list)+'_M_variations.pdf'
            call_PDF(filename_)

        #=============================================================

        x = join_stats(M_list)
        if data=='mSMOM':
            y = join_stats(y_phys_mSMOM)/x
        else:
            y = join_stats(y_phys_SMOM)/x

        def ansatz(m, param, **kwargs):
            return param[0]/m + param[1] + param[2]*m
        guess = [1, 1e-1, 1e-2]

        ymin, ymax = 0.24, 0.40
        xmin, xmax = 1.4, 3.1

        mc_ren = [y[-1]]
        pvalues = []
        for range_ in ranges:
            figure, ax = plt.subplots(figsize=(3,2))
            start, stop = range_
            if stop==None:
                stop = len(x.val)
            fitrange = np.arange(start, stop)

            res = fit_func(x, y, ansatz, guess, start=start, end=stop)
            if verbose:
                print(data+': '+' '.join([err_disp(
                    res.val[i],res.err[i]) for i in range(len(guess))]))
            mc_ren.append(res.mapping(eta_PDG))
            pvalues.append(res.pvalue)

            label = r'$p$-val:'+'{:.2f}'.format(res.pvalue)
            ax.text(0.5, 0.1, label,
                    va='center', ha='center',
                    transform=ax.transAxes)
            ax.errorbar(x.val[fitrange], y.val[fitrange],
                        xerr=x.err[fitrange], yerr=y.err[fitrange],
                        fmt='o', capsize=4, c='tab:blue',
                        label=label)
            ax.errorbar(x.val[~fitrange], y.val[~fitrange],
                        xerr=x.err[~fitrange], yerr=y.err[~fitrange],
                        fmt='o', mfc='None', capsize=4, c='tab:blue')
            ax.set_ylabel(r'$m^R/M$')
            ax.set_xlabel(r'$M\,[\mathrm{GeV}]$')

            ax.vlines(x=eta_PDG.val, ymin=ymin, ymax=ymax,
                      color='k', linestyle='dashed')
            ax.set_ylim([ymin, ymax])
            ax.text(0.5, 0.9, Mbar_label,
                    va='center', ha='center',
                    transform=ax.transAxes)

            xrange = np.linspace(xmin, xmax)
            yrange = res.mapping(xrange)
            ax.fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color='tab:blue', alpha=0.1)
            ax_twin = ax.twiny()
            ax_twin.set_xlim(ax.get_xlim())
            ax_twin.set_xticks([eta_PDG.val])
            ax_twin.set_xticklabels([r'$M_{\eta_c}^\mathrm{PDG}$'])

            ax.set_xlim([xmin, xmax])
            if pass_only:
                plt.close(figure)

        vals = [m.val for m in mc_ren]
        sys_err = (max(vals)-min(vals))/2
        stat_only = mc_ren[0]*eta_PDG
        sys_only = stat(
                val=mc_ren[0].val,
                err=sys_err,
                btsp='fill'
                )*eta_PDG
        print('mbar = '+err_disp(y_mbar_phys.val,
                                y_mbar_phys.err)+\
                ',\t m_c^R('+str(mu)+') = '+\
                err_disp(stat_only.val,
                         stat_only.err,
                         sys_err=sys_only.err))
        mc_final = stat(
                val=mc_ren[0].val,
                err=(mc_ren[0].err**2+sys_err**2)**0.5,
                btsp='fill'
                )

        if not pass_only:
            filename = '_'.join(self.ens_list)+'_M_dependence.pdf'
            call_PDF(filename)

            figure, ax = plt.subplots(nrows=1, ncols=2,
                                      sharey=True, figsize=(3.5,2),
                                      gridspec_kw={'width_ratios':[2,1]})
            plt.subplots_adjust(hspace=0, wspace=0)
            x = x/eta_PDG
            ax[0].errorbar(x.val, y.val,
                        xerr=x.err, yerr=y.err,
                        fmt='o', capsize=4, c='k',
                        label=label)
            ax[0].set_ylabel(r'$m_R/M$')
            ax[0].set_xlabel(r'$M/M_{\eta_c}^\mathrm{PDG}$')
            if len(xticks)==0:
                xticks = [eval(f) for f in name_list]
            ax[0].set_xticks(xticks)
            ax[0].set_xticklabels(name_list, fontsize=8)

            ax[0].vlines(x=1, ymin=ymin, ymax=ymax,
                      color='k', linestyle='dashed')
            ax[0].set_ylim([ymin, ymax])
            ax[0].text(0.5, 0.9, Mbar_label,
                    va='center', ha='center',
                    transform=ax[0].transAxes)
            ax_twin = ax[0].twiny()
            ax_twin.set_xlim(ax[0].get_xlim())
            ax_twin.set_xticks([1])
            ax_twin.set_xticklabels([r'$M_{\eta_c}^\mathrm{PDG}$'],
                                    fontsize=8)

            ax[0].set_xlim([0.58, 1.02])

            n_fits = len(mc_ren)
            mc_vars = join_stats(mc_ren)

            ax[1].errorbar(np.arange(n_fits), mc_vars.val,
                           yerr=mc_vars.err, fmt='o', capsize=4,
                           color='tab:gray', alpha=0.5)
            ax[1].errorbar([0], [mc_vars.val[0]],
                           yerr=[mc_vars.err[0]],
                           fmt='o', capsize=4,
                           color='k')
            ax[1].errorbar([n_fits], [mc_final.val],
                           yerr=[mc_final.err], fmt='o', capsize=4,
                           color='tab:red')
            ax[1].axhspan(mc_final.val-mc_final.err,
                          mc_final.val+mc_final.err,
                          facecolor='tab:red', alpha=0.2)

            ax[1].yaxis.tick_right()
            ax[1].set_xlim([-1,n_fits+1])
            ax[1].set_xticks(np.arange(n_fits+1))
            ax[1].set_xticklabels(['[1.0,1.0]']+range_names+['final'],
                                  fontsize=8)
            ax[1].tick_params(axis='x', labelrotation=90)

            for fit_idx in range(n_fits-1):
                label = r'$p$-val:'+'{:.2f}'.format(
                        pvalues[fit_idx])
                ax[1].text(fit_idx+1, 0.25, label,
                           va='bottom', ha='center',
                           rotation=90, fontsize=8)

            filename = '_'.join(self.ens_list)+'_M_final.pdf'
            call_PDF(filename)

        mc_final = mc_final*eta_PDG
        mc_final.stat_only, mc_final.sys_only = stat_only, sys_only
        return mc_final, y_mbar_phys

    def final_numbers(self, Mbar_list, mu=2.0, run=False,
                      plot=True, data='mSMOM', 
                      M_list=[eta_PDG*0.6, eta_PDG*0.7, eta_PDG*0.75,
                              eta_PDG*0.8, eta_PDG*0.9, eta_PDG],
                      name_list=[r'0.6', r'0.7', r'0.75',
                                 r'0.8', r'0.9', r'1.0'],
                      ranges=[(0,None), (1,None), (0,5), (0,4)],
                      verbose=False,
                      **kwargs):

        N_Mbar = len(Mbar_list)
        def get_mcmc(obj):
            return stat(
                    val=mcmc(obj.val, mu, f=4, N_f=4),
                    err='fill',
                    btsp=np.array([mcmc(obj.btsp[k],mu,f=4,N_f=4)
                                   for k in range(N_boot)])
                    )

        if run:
            if mu in self.extrap_values and data=='mSMOM':
                mbars = self.extrap_values[mu]['mbars']
                mc_mus = self.extrap_values[mu]['mc_mus']
            else:
                mbars = []
                mc_mus = []

                for Mbar_idx in tqdm(range(len(Mbar_list))):
                    Mbar = Mbar_list[Mbar_idx]
                    mc_ren, mbar = self.plot_M_variations(
                            mu, M_list, Mbar, pass_only=True, 
                            ranges=ranges, data=data, **kwargs)

                    mbars.append(mbar)
                    mc_mus.append(mc_ren)

            if data=='mSMOM':
                mbars = join_stats(mbars)
                mbars.disp = [err_disp(mbar.val, mbar.err) for mbar in mbars]
                R_conv_mu = stat(
                        val=[R_mSMOM_to_MSbar(mu, mbar.val)
                             for mbar in mbars],
                        err='fill',
                        btsp=np.array([[R_mSMOM_to_MSbar(mu, mbar.btsp[k])
                                        for mbar in mbars]
                                       for k in range(N_boot)])
                        )
            else:
                R_conv_mu = [R_mSMOM_to_MSbar(mu, 0)]

            mc_mus_MS = join_stats(mc_mus)*R_conv_mu
            mc_mus_MS.stat_only = join_stats(
                    [mc_mus[i].stat_only*R_conv_mu[i]
                     for i in range(N_Mbar)])
            mc_mus_MS.sys_only = join_stats(
                    [mc_mus[i].sys_only*R_conv_mu[i]
                     for i in range(N_Mbar)])
            mc_mus_MS.PT_only = join_stats(
                    [stat(val=mc_mus_MS[i].val,
                          err=R_m_PT_err(mu)*mc_mus_MS[i].val,
                          btsp='fill')
                     for i in range(N_Mbar)]
                    )
            mc_mus_MS.disp = [err_disp(
                mc_mus_MS.val[i], mc_mus_MS.stat_only.err[i],
                sys_err=mc_mus_MS.sys_only.err[i])+\
                        err_disp(mc_mus_MS.val[i],
                                 mc_mus_MS.PT_only.err[i], n=1)[-3:]
                              for i in range(N_Mbar)]

            stat_only = join_stats([mc_mus[i].stat_only for i in range(N_Mbar)])
            sys_only = join_stats([mc_mus[i].sys_only for i in range(N_Mbar)])
            mc_mus = join_stats(mc_mus)
            mc_mus.stat_only = stat_only
            mc_mus.sys_only = sys_only
            mc_mus.disp = [err_disp(mc_mus.val[i], mc_mus.stat_only.err[i],
                                    sys_err=mc_mus.sys_only.err[i]) for i in range(N_Mbar)]

            mc_mcs_MS = join_stats([get_mcmc(mc_mus_MS[i]) for i in range(N_Mbar)])
            mc_mcs_MS.stat_only = join_stats([get_mcmc(mc_mus_MS.stat_only[i])
                                              for i in range(N_Mbar)])
            mc_mcs_MS.sys_only = join_stats([get_mcmc(mc_mus_MS.sys_only[i])
                                              for i in range(N_Mbar)])
            mc_mcs_MS.PT_only = join_stats([get_mcmc(mc_mus_MS.PT_only[i])
                                              for i in range(N_Mbar)])
            mc_mcs_MS.disp = [err_disp(
                mc_mcs_MS.val[i], mc_mcs_MS.stat_only.err[i],
                sys_err=mc_mcs_MS.sys_only.err[i])+\
                        err_disp(mc_mcs_MS.val[i],
                                 mc_mcs_MS.PT_only.err[i], n=1)[-3:]
                              for i in range(N_Mbar)]

            if data=='mSMOM':
                self.extrap_values[mu] = {'mbars':mbars,
                                          'mSMOM_mc_mu':mc_mus,
                                          'mSMOM_mc_mu_MS':mc_mus_MS,
                                          'mSMOM_mc_mc_MS':mc_mcs_MS}
            else:
                self.extrap_values[mu].update({'SMOM_mc_mu':mc_mus,
                                               'SMOM_mc_mu_MS':mc_mus_MS,
                                               'SMOM_mc_mc_MS':mc_mcs_MS})

        if plot:
            fig, ax = plt.subplots(figsize=(3,2))
            x = join_stats(Mbar_list)/eta_PDG
            y = self.extrap_values[mu]['mSMOM_mc_mu_MS']
            ax.errorbar(x.val, y.val, xerr=x.err,
                        yerr=(y.err**2+y.PT_only.err**2)**0.5,
                        fmt='o', capsize=4, color='r')
            ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err,
                        fmt='o', capsize=4, color='k')
            ax.set_xlabel(r'$\overline{M}/M_{\eta_c}^\mathrm{PDG}$')
            ax.set_ylabel(
                    r'$m_{c,R}^{\overline{\mathrm{MS}}}(\mu='+\
                            str(mu)+'\,\mathrm{GeV})\,[\mathrm{GeV}]$')

            call_PDF('mcMSbar_v_Mbar_'+str(int(mu*10))+'.pdf')

        if verbose:
            print(f'mu = {mu}\nmbars: '+' '.join(self.extrap_values[mu]['mbars'].disp)+\
                        '\nm_c^mSMOM('+str(mu)+'): '+' '.join(self.extrap_values[mu]['mSMOM_mc_mu'].disp)+\
                        '\nm_c^MS<-mSMOM('+str(mu)+'): '+' '.join(self.extrap_values[mu]['mSMOM_mc_mu_MS'].disp)+\
                        '\nm_c^MS<-mSMOM(m_c): '+' '.join(self.extrap_values[mu]['mSMOM_mc_mc_MS'].disp)+\
                        '\nm_c^SMOM('+str(mu)+'): '+' '.join(self.extrap_values[mu]['SMOM_mc_mu'].disp)+\
                        '\nm_c^MS<-SMOM('+str(mu)+'): '+' '.join(self.extrap_values[mu]['SMOM_mc_mu_MS'].disp)+\
                        '\nm_c^MS<-SMOM(m_c): '+' '.join(self.extrap_values[mu]['SMOM_mc_mc_MS'].disp)
                  )

    def final_plot(self, mu_list=[2.0, 2.5, 3.0],
                   Mrat_list=[0.6, 0.7, 0.75], N_f=4, f=4, **kwargs):

        fig, ax = plt.subplots(nrows=1, ncols=2, 
                               sharey=True, figsize=(3,2),
                               gridspec_kw={'width_ratios':(1,2.5)})
        plt.subplots_adjust(hspace=0, wspace=0)
        
        x = np.array(Mrat_list)

        for idx, mu in enumerate(mu_list):
            y = self.extrap_values[mu]['mSMOM_mc_mu_MS']
            y_stat = y.stat_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            y_sys = y.sys_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            y_PT = y.PT_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            y = stat(
                    val=y.val*MSbar_m_running(mu,3.0,N_f=N_f,f=f), 
                    err=(y_stat.err**2+y_sys.err**2+y_PT.err**2)**0.5,
                    btsp='fill'
                    )
            y.disp = [err_disp(y.val[i],y_stat.err[i],sys_err=y_sys.err[i])+\
                    err_disp(y.val[i],y_PT.err[i],n=1)[-3:] for i in range(len(x))]

            ax[1].errorbar(x+(idx-1)/120, y.val, yerr=y.err,
                           label=r'$\mu='+str(mu)+r'\,\mathrm{GeV}$',
                           fmt=marker_list[idx], capsize=4, mfc='None',
                           color=color_list[idx])

            SMOM = self.extrap_values[mu]['SMOM_mc_mu_MS']
            SMOM_stat = SMOM.stat_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            SMOM_sys = SMOM.sys_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            SMOM_PT = SMOM.PT_only*MSbar_m_running(mu,3.0,N_f=N_f,f=f)
            SMOM = stat(
                    val=SMOM.val*MSbar_m_running(mu,3.0,N_f=N_f,f=f),
                    err=(SMOM_stat.err**2+SMOM_sys.err**2+SMOM_PT.err**2)**0.5,
                    btsp='fill'
                    )
            SMOM.disp = [err_disp(SMOM.val[i],SMOM_stat.err[i],sys_err=SMOM_sys.err[i])+\
                    err_disp(SMOM.val[i],SMOM_PT.err[i],n=1)[-3:] for i in range(1)]

            ax[0].errorbar([0.0+(idx-1)/120], SMOM.val, yerr=SMOM.err,
                           fmt=marker_list[idx], capsize=4, mfc='None',
                           color=color_list[idx])
            print(str(mu)+f'->3.0GeV: mSMOM={y.disp}\nSMOM={SMOM.disp}.')

        ymin, ymax = ax[0].get_ylim()
        ax[0].set_ylim((ymin, ymax*1.03))
        ax[1].set_xlim([0.57,0.78])
        ax[0].set_xlim([-0.03,0.03])
        ax[1].set_xlabel(r'$\overline{M}/M_{\eta_c}^\mathrm{PDG}$')
        ax[0].set_title(r'SMOM', fontsize=10)
        ax[1].set_title(r'mSMOM', fontsize=10)
        ax[0].set_ylabel(
                r'$m_{c,R}^{\overline{\mathrm{MS}}}(3\,\mathrm{GeV}\leftarrow\mu)\,[\mathrm{GeV}]$')
        ax[1].set_xticks(Mrat_list)
        ax[1].set_xticklabels([str(m) for m in Mrat_list])
        ax[0].set_xticks([0])
        ax[0].set_xticklabels(['0'])
        ax[1].yaxis.tick_right()
        ax[1].legend(bbox_to_anchor=(1.1,1), ncol=len(mu_list),
                     columnspacing=0.2, fontsize=8, framealpha=1)

        call_PDF('mcMSbar_v_Mbar_all_mu.pdf')

    def mu_variations(self, mu_list, Mbar_list=[0.6, 0.7, 0.75],
                      **kwargs):
        mc_mus = []
        for mu in mu_list:
            Mbar_list_eta = [eta_PDG*f for f in Mbar_list]
            mc_mus.append(self.final_numbers(
                Mbar_list=Mbar_list_eta, mu=mu,
                plot=False, verbose=False))

        x = stat(
                val=mu_list,
                err=np.zeros(len(mu_list)),
                btsp='fill'
                )
        fig, ax = plt.subplots(figsize=(3,3))
        ansatz, guess = self.ansatz(choose='linear')
        for Mbar_idx, Mbar in enumerate(Mbar_list):
            y = stat(
                    val=[mcs.val[Mbar_idx] for mcs in mc_mus],
                    err=[mcs.err[Mbar_idx] for mcs in mc_mus],
                    btsp=np.array([[mcs.btsp[k,Mbar_idx]
                                    for mcs in mc_mus]
                                   for k in range(N_boot)])
                    )
            label = r'$\overline{M}='+str(Mbar)+\
                    r'\,M_{\eta_c}^\mathrm{PDG}$'
            ax.errorbar(x.val, y.val, yerr=y.err, 
                        capsize=4, fmt='o',
                        mfc='None', label=label,
                        color=color_list[Mbar_idx])
            
            res = fit_func(x, y, ansatz, guess)
            mc_mc = res.mapping(mc_PDG)
            print(f'Mbar/M_eta_c={Mbar},\t mc(mc)={err_disp(mc_mc.val, mc_mc.err)}')

            ax.errorbar(mc_PDG.val, mc_mc.val, 
                        xerr=mc_PDG.err, yerr=mc_mc.err, 
                         capsize=4, fmt='o',
                        color=color_list[Mbar_idx])

            xmin, xmax = ax.get_xlim()
            xrange = np.linspace(xmin, xmax,50)
            yrange = res.mapping(xrange)
            ax.fill_between(xrange, yrange.val+yrange.err,
                            yrange.val-yrange.err,
                            color=color_list[Mbar_idx],
                            alpha=0.1)
            ax.set_xlim([xmin, xmax])

            ymin, ymax = ax.get_ylim()
            ax.vlines(x=mc_PDG.val, ymin=ymin, ymax=ymax,
                      color='k', linestyle='dashed')
            ax.set_ylim([ymin, ymax])

            ax_twin = ax.twiny()
            ax_twin.set_xlim(ax.get_xlim())
            ax_twin.set_xticks([mc_PDG.val])
            ax_twin.set_xticklabels([r'$m_c^\mathrm{PDG}$'])


        ax.set_xlabel(r'$\mu\,[\mathrm{GeV}]$')
        ax.set_ylabel(r'$m_c^{\overline{\mathrm{MS}}}(\mu)\,[\mathrm{GeV}]$')
        ax.legend(fontsize=8)

        call_PDF('mu_variations.pdf')

    def plot_Mbar_variations(self, mu, M, Mbar_list,
                             filename='', M_label='',
                             name_list=[],
                             **kwargs):
        ansatz, guess = self.ansatz(**kwargs)
        x = self.a_sq
        figure, ax = plt.subplots(figsize=(4,4.4))
        plt.subplots_adjust(hspace=0, wspace=0)
        for Mbar_idx in tqdm(range(len(Mbar_list))):
            Mbar = Mbar_list[Mbar_idx]
            color = color_list[Mbar_idx]
            fit_dict = self.plot_cont_extrap(
                    mu, M, Mbar, plot=False, **kwargs)
            end = fit_dict['end']
            xrange = fit_dict['xrange']
            if Mbar_idx==0:
                xmax_store = xrange[-1]
                y_mc_SMOM = fit_dict['y_mc_SMOM'][:end]
                ax.errorbar(x.val[:end], y_mc_SMOM.val, 
                               xerr=x.err[:end], yerr=y_mc_SMOM.err,
                               fmt='d', capsize=4, color='k',
                               mfc='None', label='SMOM')
                res_mc_SMOM = fit_dict['res_mc_SMOM']
                y_mc_SMOM_phys = res_mc_SMOM[0]

                yrange = res_mc_SMOM.mapping(xrange)
                ax.errorbar([0.0], [y_mc_SMOM_phys.val],
                               yerr=[y_mc_SMOM_phys.err],
                               capsize=4, fmt='d', color='k')
                ax.fill_between(xrange, yrange.val+yrange.err,
                                   yrange.val-yrange.err,
                                   color='k', alpha=0.1)

            y_mc_mSMOM = fit_dict['y_mc_mSMOM'][:end]
            if len(name_list)!=len(Mbar_list):
                label = r'$\overline{M}$='+err_disp(Mbar.val, Mbar.err)
            else:
                label = r'$\overline{M}='+name_list[Mbar_idx]+\
                        r'M_{\eta_c}^\mathrm{PDG}$'
            ax.errorbar(x.val[:end], y_mc_mSMOM.val, 
                        xerr=x.err[:end], yerr=y_mc_mSMOM.err,
                        fmt='o', capsize=4, color=color, 
                        mfc='None', label=label)

            res_mc_mSMOM = fit_dict['res_mc_mSMOM']
            y_mc_mSMOM_phys = res_mc_mSMOM[0]
            yrange = res_mc_mSMOM.mapping(xrange)
            ax.errorbar([0.0], [y_mc_mSMOM_phys.val],
                           yerr=[y_mc_mSMOM_phys.err],
                           capsize=4, fmt='o', color=color)
            ax.fill_between(xrange, yrange.val+yrange.err,
                               yrange.val-yrange.err,
                               color=color, alpha=0.2)

        ax.set_xlim([-0.02, xmax_store])
        M_label = r'$M:'+err_disp(M.val, M.err)+r'$'\
                if M_label=='' else M_label
        ax.set_title(M_label)
        ax.set_ylabel(r'$m_R(\mu='+str(mu)+\
                r'\,\mathrm{GeV}, \overline{m}_R)\, [\mathrm{GeV}]$')
        ax.set_xlabel(r'$a^2\,[\mathrm{GeV}^{-2}]$')

        ax.legend(fontsize=8)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        ax.vlines(x=0.0, ymin=ymin, ymax=ymax,
               color='k', linestyle='dashed')
        ax.set_ylim([ymin, ymax])

        if filename=='':
            filename = '_'.join(self.ens_list)+'_cont_extrap.pdf'
        call_PDF(filename)

    def ansatz(self, choose=None, **kwargs):
        def linear_ansatz(asq, param, **kwargs):
            return param[0] + param[1]*asq
        def quadratic_ansatz(asq, param, **kwargs):
            return param[0] + param[1]*asq + param[2]*(asq**2)

        ansatz = linear_ansatz if len(self.ens_list)==2 else quadratic_ansatz
        guess = [1, 1e-1] if len(self.ens_list)==2 else [1,1e-1,1e-2]
        if choose=='linear':
            return linear_ansatz, [1, 1e-1]
        elif choose=='quadratic':
            return quadratic_ansatz, [1, 1e-1, 1e-2]
        else:
            return ansatz, guess 

    def CL_variations(self, mu, M, Mbar, plot=False, 
                      pass_axis=False, axis=None, **kwargs):
        fit_dicts = []
        if plot:
            if pass_axis:
                axs = axis
            else:
                figs, axs = plt.subplots(figsize=(5,3.5))
            kwargs.update({'plot_SMOM':False,
                           'pass_axis':True,
                           'axis':axs,
                           'mSMOM_color':'k'
                           })

        fit_dicts.append(self.plot_cont_extrap(
                mu, M, Mbar, with_amres=True, fit_alt=True, plot=plot, 
                phys_color=color_list[0], mSMOM_label=r'$a^2+am_\mathrm{res}$ ',
                plot_amres=True, amres_invert=True, **kwargs))
        fit_dicts.append(self.M_only.plot_cont_extrap(
                mu, M, Mbar, fit_alt=True, phys_color=color_list[1], plot=plot, 
                mSMOM_label=r'$a^2\,\mathrm{(M\"{o}bius}\,\mathrm{only)}$',
                plot_amres=False, **kwargs))

        if M.val<0.81*eta_PDG.val and Mbar.val<0.81*eta_PDG.val:
            fit_dicts.append(self.no_C1S.plot_cont_extrap(
                    mu, M, Mbar, fit_alt=True, phys_color=color_list[2],
                    plot=plot, mSMOM_label=r'$a^2$ (no C1S) ', 
                    plot_amres=False, **kwargs))
        else:
            fit_dicts.append(self.S_only.plot_cont_extrap(
                    mu, M, Mbar, fit_alt=True, phys_color=color_list[2],
                    mSMOM_label=r'$a^2\,\mathrm{(Shamir}\,\mathrm{only)}$',
                    plot=plot, plot_amres=False, **kwargs))


        mSMOM_phys_vals = [d['res_mc_mSMOM'][0] for d in fit_dicts]
        mSMOM_errors = [p.err for p in mSMOM_phys_vals]
        mSMOM_vals = [p.val for p in mSMOM_phys_vals]
        stat_err = mSMOM_errors[0]
        sys_err = (max(mSMOM_vals)-min(mSMOM_vals))/2
        mSMOM_number = stat(
                val=mSMOM_phys_vals[0].val,
                err=(stat_err**2+sys_err**2)**0.5,
                btsp='fill'
                )

        SMOM_phys_vals = [d['res_mc_SMOM'][0] for d in fit_dicts]
        SMOM_errors = [p.err for p in SMOM_phys_vals]
        SMOM_vals = [p.val for p in SMOM_phys_vals]
        stat_err = SMOM_errors[0]
        sys_err = (max(SMOM_vals)-min(SMOM_vals))/2
        SMOM_number = stat(
                val=SMOM_phys_vals[0].val,
                err=(stat_err**2+sys_err**2)**0.5,
                btsp='fill'
                )

        if plot:
            ylow = min(mSMOM_vals)-max(mSMOM_errors)
            xmax = fit_dicts[0]['xrange'][-1]
            axs.set_xlim((-0.02, xmax))
            ymin, ymax = axs.get_ylim()
            axs.set_ylim((ylow, ymax))
            call_PDF('CL_variations.pdf')

        return mSMOM_number, SMOM_number, fit_dicts[0]



