from massive import *

mu_chosen = 3.0
ens_list = valence_ens
mNPR_dict = {ens: mNPR(ens, mu=mu_chosen)
             for ens in ens_list}

h, w = 4, 2.75
num_ens = len(ens_list)

# ===plot 3: Zms vs am for all with SMOM==============
fig, axes = plt.subplots(1, num_ens, sharey=True,
                         figsize=(w*num_ens, h))
plt.subplots_adjust(hspace=0, wspace=0)
for ens in ['C1', 'M1', 'F1S']:
    idx = ens_list.index(ens)
    ax = axes[idx]

    e = mNPR_dict[ens]
    x, y = e.load_mSMOM(key='m')

    ax.errorbar(x.val, y.val, yerr=y.err,
                fmt='o', capsize=4, label='mSMOM',
                color=color_list[idx])
    ax.axhspan(e.Z_SMOM.val-e.Z_SMOM.err,
               e.Z_SMOM.val+e.Z_SMOM.err,
               color='k', alpha=0.1)
    ax.axhline(e.Z_SMOM.val, color='k', label=r'SMOM $Z_m$')

    ax.axhspan(e.Z_P_inv_SMOM.val-e.Z_P_inv_SMOM.err,
               e.Z_P_inv_SMOM.val+e.Z_P_inv_SMOM.err,
               color='r', alpha=0.1)
    ax.axhline(e.Z_P_inv_SMOM.val, color='r', label=r'SMOM $1/Z_P$')

    ax.legend(loc='center right')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([-0.05, xmax])
    ax.set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')
    ax.set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)


# ===plot 1: Zms and M_eta_C vs am============================
axes_info = np.zeros(shape=(2, num_ens, 2, 2))
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))
plt.subplots_adjust(hspace=0, wspace=0)
for ens in ['C1', 'M1', 'F1S']:
    idx = ens_list.index(ens)
    e = mNPR_dict[ens]

    axes[0, idx].errorbar(e.eta_ax.val, e.eta_y.val,
                          yerr=e.eta_y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    xmin0, xmax0 = axes[0, idx].get_xlim()
    axes[0, 0].set_ylabel(r'$M_{\eta_h}$ (GeV)', fontsize=18)
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(
        e.ainv.val)+r'$ GeV)')
    axes_info[0, idx, 1, :] = axes[0, idx].get_ylim()

    x, y = e.load_mSMOM(key='m')
    axes[1, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)
    xmin1, xmax1 = axes[1, idx].get_xlim()
    axes_info[1, idx, 0, :] = np.min(
        [xmin0, xmin1]), np.max([xmax0, xmax1])
    axes_info[0, idx, 0, :] = axes_info[1, idx, 0, :]
    axes[0, idx].set_xlim([-0.05, np.max([xmax0, xmax1])])
    axes_info[1, idx, 1, :] = axes[1, idx].get_ylim()

axes_info_combined = np.array([[[np.min(axes_info[row_num, :, axis, 0]),
                                np.max(axes_info[row_num, :, axis, 1])]
                               for axis in range(2)]
                              for row_num in range(2)])
xmax = np.max(axes_info_combined[:, 0, 1])
axes_info_combined[:, 0, 1] = xmax, xmax
plt.close()

eta_C_star_idx = 1
eta_C_star = eta_stars[eta_C_star_idx]

# ===plot 3: Zms interpolate for some choice M_eta_C_star===


fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))

plt.subplots_adjust(hspace=0, wspace=0)

for idx, ens in enumerate(['C1', 'M1', 'F1S']):
    e = mNPR_dict[ens]
    am_star = e.interpolate_eta_c(eta_C_star)

    axes[0, idx].errorbar(e.eta_ax.val, e.eta_y.val,
                          yerr=e.eta_y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(
        e.ainv.val)+r'$ GeV)')

    xmin, xmax = axes[0, idx].get_xlim()
    axes[0, idx].hlines(eta_C_star, xmin, xmax,
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].axvspan(am_star.val-am_star.err,
                         am_star.val+am_star.err,
                         color='k', alpha=0.1)

    ymin, ymax = axes_info_combined[0, 1, :]
    axes[0, idx].vlines(am_star.val, ymin, eta_C_star,
                        linestyle='dashed',
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].text(am_star.val*1.01, 0.3,
                      r'$a_{'+ens+r'}m^\star$', rotation=270, fontsize=18)

    axes[0, idx].set_xlim(xmin, xmax)
    axes[0, idx].set_ylim(axes_info_combined[0, 1, :])
    axes[0, idx].set_title(
        ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')

    x, y = e.load_mSMOM(key='m')
    axes[1, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

    am_grain = stat(
        val=np.linspace(0.0001, xmax+0.5, 100),
        err=np.zeros(100),
        btsp='fill')

    if ens != 'F1S':
        guess = [1, 1e-1]

        def Z_m_ansatz(am, params):
            return params[0] + params[1]/am
    else:
        guess = [1, 1e-1, 1e-2]

        def Z_m_ansatz(am, params):
            return params[0] + params[1]/am + params[2]*am

    # guess = [1, 1e-1, 1e-2]

    # def Z_m_ansatz(am, params):
    #    return params[0] + params[1]/am + params[2]*np.log(am)*(am**2)

    e.Z_m_mSMOM_fit = fit_func(x, y, Z_m_ansatz, guess)
    Zm_grain = e.Z_m_mSMOM_fit.mapping(am_grain)
    Zm_grain_up = Zm_grain.val+Zm_grain.err
    Zm_grain_dn = Zm_grain.val-Zm_grain.err
    axes[1, idx].fill_between(am_grain.val, Zm_grain_up, Zm_grain_dn,
                              color='k', alpha=0.3, label='fit', zorder=0)

    Zm_star = e.Z_m_mSMOM_fit.mapping(am_star)
    axes[1, idx].errorbar([am_star.val], [Zm_star.val],
                          xerr=[am_star.err],
                          yerr=[Zm_star.err],
                          fmt='o', capsize=4,
                          color=color_list[3+eta_C_star_idx],
                          zorder=2,)

    chi_sq_DOF = np.around(e.Z_m_mSMOM_fit.chi_sq/e.Z_m_mSMOM_fit.DOF, 3)
    axes[1, idx].text(0.5, 0.1, r'$\chi^2$/DOF: '+str(chi_sq_DOF),
                      ha='center', va='center',
                      transform=axes[1, idx].transAxes)

    axes[1, idx].set_xlim(xmin, xmax)
    axes[1, idx].set_ylim(axes_info_combined[1, 1, :])

for ens_idx, ens in enumerate(ens_list):
    e = mNPR_dict[ens]
    am_star = e.interpolate_eta_c(eta_C_star)
    for ax_idx in [1]:
        ymin, ymax = axes[ax_idx, ens_idx].get_ylim()
        axes[ax_idx, ens_idx].vlines(
            am_star.val, ymin, ymax,
            linestyle='dashed',
            color=color_list[3+eta_C_star_idx], zorder=0)
        axes[ax_idx, ens_idx].axvspan(am_star.val-am_star.err,
                                      am_star.val+am_star.err,
                                      color='k', alpha=0.1)
        axes[ax_idx, ens_idx].set_ylim(ymin, ymax)


axes[0, 0].set_ylabel(r'$M_{\eta_h}$ (GeV)', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[0, 0].text(axes_info_combined[0, 0, 0]+0.1, eta_C_star *
                1.03, r'$M^\star$', fontsize=16)


# ===plot 3: Zms and Z_mm_q interpolate for some choice M_eta_C_star===
fig, axes = plt.subplots(nrows=3, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*3),
                         gridspec_kw={'width_ratios': [1, 1, 1]})

plt.subplots_adjust(hspace=0, wspace=0)

for idx, ens in enumerate(['C1', 'M1', 'F1S']):
    e = mNPR_dict[ens]
    am_star = e.interpolate_eta_c(eta_C_star)

    axes[0, idx].errorbar(e.eta_ax.val, e.eta_y.val,
                          yerr=e.eta_y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(
        e.ainv.val)+r'$ GeV)')

    xmin, xmax = axes[0, idx].get_xlim()
    xmax = max(e.am_C.val*1.05, xmax)
    axes[0, idx].hlines(eta_C_star, xmin, xmax,
                        color=color_list[3+eta_C_star_idx])

    axes[0, idx].hlines(eta_PDG.val, xmin, xmax,
                        color='0.5')
    axes[0, idx].axhspan(eta_PDG.val-eta_PDG.err,
                         eta_PDG.val+eta_PDG.err,
                         xmin, xmax,
                         color='0.7')

    ymin, ymax = axes_info_combined[0, 1, :]
    axes[0, idx].vlines(am_star.val, ymin, eta_C_star,
                        linestyle='dashed',
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].fill_between(np.linspace(am_star.val-am_star.err,
                              am_star.val+am_star.err, 10),
                              ymin, eta_C_star,
                              color=color_list[3+eta_C_star_idx],
                              alpha=0.1)
    axes[0, idx].text(am_star.val*0.97, 0.5, r'$am^\star$',
                      color=color_list[3+eta_C_star_idx],
                      ha='right', va='center',
                      rotation=270, fontsize=18)

    axes[0, idx].vlines(e.am_C.val, ymin, eta_PDG.val,
                        linestyle='dashed',
                        color='0.5')
    axes[0, idx].fill_between(np.linspace(e.am_C.val-e.am_C.err,
                              e.am_C.val+e.am_C.err, 10),
                              ymin, eta_PDG.val,
                              color='0.9')
    axes[0, idx].text(e.am_C.val, 0.3, r'$am_c$',
                      ha='right', va='center',
                      color='0.5', rotation=270, fontsize=18)

    axes[0, idx].set_xlim(xmin, xmax)
    axes[0, idx].set_ylim(axes_info_combined[0, 1, :])
    axes[0, idx].set_title(
        ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')

    x, y = e.load_mSMOM(key='m')
    y = y*e.m_C
    axes[1, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)
    xmin, xmax = axes[0, idx].get_xlim()

    am_grain = stat(
        val=np.linspace(0.005, xmax+0.5, 100),
        err=np.zeros(100),
        btsp='fill')
    m_c_ren_grain = e.Z_m_mSMOM_fit.mapping(am_grain)*e.m_C
    axes[1, idx].fill_between(am_grain.val,
                              m_c_ren_grain.val+m_c_ren_grain.err,
                              m_c_ren_grain.val-m_c_ren_grain.err,
                              color='k', alpha=0.3, label='fit', zorder=0)

    m_c_ren_star = e.Z_m_mSMOM_fit.mapping(am_star)*e.m_C
    axes[1, idx].errorbar([am_star.val], [m_c_ren_star.val],
                          xerr=[am_star.err],
                          yerr=[m_c_ren_star.err],
                          fmt='o', capsize=4,
                          color=color_list[3+eta_C_star_idx],
                          zorder=2,)

    axes[1, idx].set_xlim(xmin, xmax)

    x, y = e.load_mSMOM(key='m')
    y = y*x*e.ainv
    axes[2, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[2, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)
    xmin, xmax = axes[2, idx].get_xlim()

    am_grain = stat(
        val=np.linspace(0.0001, xmax+0.5, 100),
        err=np.zeros(100),
        btsp='fill')
    m_bar_grain = e.Z_m_mSMOM_fit.mapping(am_grain)*am_grain*e.ainv
    axes[2, idx].fill_between(am_grain.val,
                              m_bar_grain.val+m_bar_grain.err,
                              m_bar_grain.val-m_bar_grain.err,
                              color='k', alpha=0.3, label='fit', zorder=0)
    m_bar_star = e.Z_m_mSMOM_fit.mapping(am_star)*am_star*e.ainv
    axes[2, idx].errorbar([am_star.val], [m_bar_star.val],
                          xerr=[am_star.err],
                          yerr=[m_bar_star.err],
                          fmt='o', capsize=4,
                          color=color_list[3+eta_C_star_idx],
                          zorder=2,)

    axes[2, idx].set_xlim(xmin, xmax)

for ens_idx, ens in enumerate(ens_list):
    e = mNPR_dict[ens]
    am_star = e.interpolate_eta_c(eta_C_star)
    for ax_idx in [1, 2]:
        ymin, ymax = axes[ax_idx, ens_idx].get_ylim()
        if ax_idx == 2:
            ymax *= 0.75
        else:
            ymin *= 1.1
        axes[ax_idx, ens_idx].vlines(
            am_star.val, ymin, ymax,
            linestyle='dashed',
            color=color_list[3+eta_C_star_idx], zorder=0)
        axes[ax_idx, ens_idx].axvspan(am_star.val-am_star.err,
                                      am_star.val+am_star.err,
                                      color='k', alpha=0.1)
        axes[ax_idx, ens_idx].set_ylim(ymin, ymax)


axes[0, 0].set_ylabel(r'$M_{\eta_h}$ (GeV)', fontsize=16)
axes[0, 0].text(axes_info_combined[0, 0, 0]+0.02, eta_C_star *
                1.03, r'$M^\star$', fontsize=16)
axes[0, 0].text(axes_info_combined[0, 0, 0]+0.02, eta_PDG.val *
                1.04, r'$M_{\eta_c}^{PDG}$', fontsize=16)
axes[2, 0].set_ylabel(
    r'$Z_m(am,a\mu)\cdot a^{-1}\cdot (am)$ (GeV)', fontsize=16)
axes[1, 0].set_ylabel(
    r'$Z_m(am,a\mu)\cdot a^{-1}\cdot (am_c)$ (GeV)', fontsize=16)
plt.savefig(
    '/Users/rajnandinimukherjee/Desktop/Proceedings_Plots/interpolation.pdf')

# ===STEP4 plot 2: m_C quadratic continuum extrap for eta_star=============
fig = plt.figure(figsize=(h, w))
ax = fig.add_subplot(111)

asq_grain = np.linspace(0, 0.45, 50)
ren_xaxes = [-0.01, 0.35]
extrap = cont_extrap(ens_list, mu=mu_chosen)
extrap.load_SMOM()
m_C_SMOM_mapping = extrap.extrap_mapping(extrap.m_C_SMOM, fit='quad')
m_C_SMOM_grain = m_C_SMOM_mapping(asq_grain)
phys_SMOM = (m_C_SMOM_mapping(0.0), 0.0)
ax.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)

ax.plot(asq_grain, m_C_SMOM_grain.val,
        linestyle='dashed',
        color='0.5', label='SMOM')
ax.fill_between(asq_grain,
                m_C_SMOM_grain.val - m_C_SMOM_grain.err,
                m_C_SMOM_grain.val + m_C_SMOM_grain.err,
                color='0.8')
for ens_idx, ens in enumerate(ens_list):
    ax.errorbar(extrap.a_sq.val[ens_idx],
                extrap.m_C_SMOM.val[ens_idx],
                xerr=extrap.a_sq.err[ens_idx],
                yerr=extrap.m_C_SMOM.err[ens_idx],
                fmt='o', capsize=4, color='0.5')
ax.errorbar([0], phys_SMOM[0].val,
            yerr=phys_SMOM[0].err,
            fmt='o', capsize=4,
            color='0.5')

phys_mSMOM = []
for eta_idx, eta_star in enumerate(eta_stars):
    m_C_mSMOM = []
    for ens in ens_list:
        am_star = mNPR_dict[ens].interpolate_eta_c(eta_star)
        Z_m_star = mNPR_dict[ens].Z_m_mSMOM_fit.mapping(am_star)
        m_C_ren = Z_m_star*mNPR_dict[ens].m_C
        m_C_mSMOM.append(m_C_ren)
    m_C_mSMOM = join_stats(m_C_mSMOM)

    join_stats([mNPR_dict[ens].Z_m_mSMOM_fit.mapping(
        eta_star)*mNPR_dict[ens].m_C for ens in ens_list])
    m_C_mSMOM_mapping = extrap.extrap_mapping(m_C_mSMOM, fit='quad')
    phys_mSMOM.append([m_C_mSMOM_mapping(0.0)])
    m_C_mSMOM_grain = m_C_mSMOM_mapping(asq_grain)

    # label = r'$\overline{m}\approx'+str(np.around(phys_mbar.val, 2))+r'$ GeV'
    label = r'$M^\star='+str(eta_star)+'$ GeV'\
        if eta_star != eta_PDG.val else r'$M^\star=M_{\eta_C}^{PDG}$'

    ax.plot(asq_grain, m_C_mSMOM_grain.val,
            linestyle='dashed',
            color=color_list[3+eta_idx], label=label)
    ax.fill_between(asq_grain,
                    m_C_mSMOM_grain.val - m_C_mSMOM_grain.err,
                    m_C_mSMOM_grain.val + m_C_mSMOM_grain.err,
                    color=color_list[3+eta_idx], alpha=0.5)

    for ens_idx, ens in enumerate(ens_list):
        ax.errorbar(extrap.a_sq.val[ens_idx],
                    m_C_mSMOM.val[ens_idx],
                    xerr=extrap.a_sq.err[ens_idx],
                    yerr=m_C_mSMOM.err[ens_idx],
                    fmt='o', capsize=4,
                    color=color_list[3+eta_idx])

    m_C_mSMOM_phys = m_C_mSMOM_mapping(0.0)
    ax.errorbar([0], m_C_mSMOM_phys.val,
                yerr=m_C_mSMOM_phys.err,
                fmt='o', capsize=4,
                color=color_list[3+eta_idx])


ax.legend(loc='lower right')
ax.set_xlabel(r'$a^2$ (GeV${}^{-2}$)', fontsize=18)
# plt.ylabel(r'$m_{C}(am^\star,a\mu)$', fontsize=18)
ax.set_ylabel(r'$m_{c}^{ren}$ (GeV)', fontsize=18)
ax.set_xlim(ren_xaxes)
ax_twin = ax.twiny()
ax_twin.set_xlim(ax.get_xlim())
ax_twin.set_xticks([0]+list(extrap.a_sq.val))
ax_twin.set_xticklabels([r'cont', 'C', 'M', 'F'])
fig.savefig(
    '/Users/rajnandinimukherjee/Desktop/Proceedings_Plots/m_c.pdf')

# == = STEP4 plot 2: m_C linear continuum extrap for eta_star == == == == == == =
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
for eta_idx, eta_star in enumerate(eta_stars):
    m_C_mSMOM = []
    for ens in ens_list:
        am_star = mNPR_dict[ens].interpolate_eta_c(eta_star)
        Z_m_star = mNPR_dict[ens].Z_m_mSMOM_fit.mapping(am_star)
        m_C_ren = Z_m_star*mNPR_dict[ens].m_C
        m_C_mSMOM.append(m_C_ren)
    m_C_mSMOM = join_stats(m_C_mSMOM)
    m_C_mSMOM_mapping = extrap.extrap_mapping(m_C_mSMOM, fit='linear')
    m_C_mSMOM_grain = m_C_mSMOM_mapping(asq_grain)

    label = r'$M_{\eta_C}^\star='+str(eta_star)+'$ GeV'\
            if eta_star != eta_PDG.val else r'$M_{\eta_C}^{\star,PDG}$'

    plt.plot(asq_grain, m_C_mSMOM_grain.val,
             linestyle='dashed',
             color=color_list[3+eta_idx], label=label)
    plt.fill_between(asq_grain,
                     m_C_mSMOM_grain.val - m_C_mSMOM_grain.err,
                     m_C_mSMOM_grain.val + m_C_mSMOM_grain.err,
                     color=color_list[3+eta_idx], alpha=0.5)

    for ens_idx, ens in enumerate(ens_list):
        plt.errorbar(extrap.a_sq.val[ens_idx],
                     m_C_mSMOM.val[ens_idx],
                     m_C_mSMOM.err[ens_idx],
                     fmt='o', capsize=4,
                     color=color_list[3+eta_idx])

    m_C_mSMOM_phys = m_C_mSMOM_mapping(0.0)
    plt.errorbar([0], m_C_mSMOM_phys.val,
                 yerr=m_C_mSMOM_phys.err,
                 fmt='o', capsize=4,
                 color=color_list[3+eta_idx])

m_C_SMOM_lin_mapping = extrap.extrap_mapping(extrap.m_C_SMOM, fit='linear')
m_C_SMOM_lin_grain = m_C_SMOM_lin_mapping(asq_grain)
plt.plot(asq_grain, m_C_SMOM_lin_grain.val,
         linestyle='dashed',
         color='0.5', label='SMOM')
plt.fill_between(asq_grain,
                 m_C_SMOM_lin_grain.val - m_C_SMOM_lin_grain.err,
                 m_C_SMOM_lin_grain.val + m_C_SMOM_lin_grain.err,
                 color='0.8')
for ens_idx, ens in enumerate(ens_list):
    plt.errorbar(extrap.a_sq.val[ens_idx],
                 extrap.m_C_SMOM.val[ens_idx],
                 extrap.m_C_SMOM.err[ens_idx],
                 fmt='o', capsize=4,
                 color='0.5')


plt.legend(loc='upper left')
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)


# ===STEP4 plot 4: mbar continuum extrap for eta_stars=============
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)

for eta_idx, eta_star in enumerate(eta_stars):
    mbar = []
    for ens in ens_list:
        am_star = mNPR_dict[ens].interpolate_eta_c(eta_star)
        Z_m_star = mNPR_dict[ens].Z_m_mSMOM_fit.mapping(am_star)
        m_star_ren = Z_m_star*am_star*mNPR_dict[ens].ainv
        mbar.append(m_star_ren)

    mbar = join_stats(mbar)
    mbar = join_stats([mNPR_dict[ens].Z_m_mSMOM_fit.mapping(
        eta_star)*mNPR_dict[ens].interpolate_eta_c(
            eta_star)*mNPR_dict[ens].ainv for ens in ens_list])
    mbar_mSMOM_mapping = extrap.extrap_mapping(mbar)
    mbar_mSMOM_grain = mbar_mSMOM_mapping(asq_grain)

    label = r'$M_{\eta_C}^\star='+str(eta_star)+'$ GeV'\
            if eta_star != eta_PDG.val else r'$M_{\eta_C}^{\star,PDG}$'

    plt.plot(asq_grain, mbar_mSMOM_grain.val,
             linestyle='dashed',
             color=color_list[3+eta_idx], label=label)
    plt.fill_between(asq_grain,
                     mbar_mSMOM_grain.val - mbar_mSMOM_grain.err,
                     mbar_mSMOM_grain.val + mbar_mSMOM_grain.err,
                     color=color_list[3+eta_idx], alpha=0.5)

    for ens_idx, ens in enumerate(ens_list):
        plt.errorbar(extrap.a_sq.val[ens_idx],
                     mbar.val[ens_idx],
                     mbar.err[ens_idx],
                     fmt='o', capsize=4,
                     color=color_list[3+eta_idx])

    mbar_mSMOM_phys = mbar_mSMOM_mapping(0.0)
    phys_mSMOM[eta_idx].append(mbar_mSMOM_mapping(0.0))
    plt.errorbar([0], mbar_mSMOM_phys.val,
                 yerr=mbar_mSMOM_phys.err,
                 fmt='o', capsize=4,
                 color=color_list[3+eta_idx])

plt.xlabel(r'$a^2$ (GeV${}^{-2}$)', fontsize=18)
plt.ylabel(r'$\overline{m}(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)
plt.savefig(
    '/Users/rajnandinimukherjee/Desktop/Proceedings_Plots/mbar.pdf')


# ===STEP5 plot 1: convert to MS-bar=============
plt.figure(figsize=(h, w))
phys_SMOM_MS = phys_SMOM*R_mSMOM_to_MSbar(mu_chosen, 0.0)
plt.errorbar([phys_SMOM[1]], phys_SMOM[0].val, yerr=phys_SMOM[0].err,
             fmt='o', capsize=4, color='k', label='SMOM')

phys_mSMOM_MS = []
for eta_idx, eta in enumerate(eta_stars):
    R = stat(
        val=R_mSMOM_to_MSbar(mu_chosen, phys_mSMOM[eta_idx][1].val),
        err='fill',
        btsp=np.array([R_mSMOM_to_MSbar(
            mu_chosen, phys_mSMOM[eta_idx][1].btsp[k])
            for k in range(N_boot)])
    )
    phys = phys_mSMOM[eta_idx][0]*R
    phys_mSMOM_MS.append(phys)

    label = r'$M_{\eta_C}^\star='+str(eta)+'$ GeV'\
            if eta != eta_PDG.val else r'$M_{\eta_C}^{\star,PDG}$'
    plt.errorbar([eta], phys.val, yerr=phys.err,
                 fmt='o', capsize=4, color=color_list[3+eta_idx])

plt.xlabel(r'$M^\star$ (GeV)', fontsize=18)
plt.ylabel(r'$m_C^{\overline{MS}}(\mu='+str(mu_chosen)+'$ GeV$)$', fontsize=18)

filename = '/Users/rajnandinimukherjee/Desktop/LatPlots.pdf'
call_PDF(filename)
