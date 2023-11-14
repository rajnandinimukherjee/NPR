from massive import *

mu_chosen = 2.0
ens_list = list(eta_c_data.keys())
mNPR_dict = {ens: mNPR(ens, mu=mu_chosen)
             for ens in ens_list}

filename = '/Users/rajnandinimukherjee/Desktop/LatPlots.pdf'
pdf = PdfPages(filename)
h, w = 4, 2.75
num_ens = len(ens_list)

# ===plot 1: M1 M_eta_C measurements==========================
plt.figure(figsize=(w, h))
M1 = mNPR_dict['M1']
plt.errorbar(M1.eta_ax.val, M1.eta_y.val,
             yerr=M1.eta_y.err,
             fmt='o', capsize=4,
             color=color_list[ens_list.index('M1')])
xmin, xmax = plt.xlim()
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')


# ===plot 2: M1 M_eta_C meas+ PDG interpolation=========
plt.figure(figsize=(w, h))
plt.errorbar(M1.eta_ax.val, M1.eta_y.val,
             yerr=M1.eta_y.err,
             fmt='o', capsize=4,
             color=color_list[ens_list.index('M1')])
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.axhspan(eta_PDG.val-eta_PDG.err,
            eta_PDG.val+eta_PDG.err,
            -0.05, M1.am_C.val+M1.am_C.err, alpha=0.1,
            color=color_list[3+eta_stars.index(eta_PDG.val)])
plt.hlines(eta_PDG.val, -0.05, M1.am_C.val+M1.am_C.err,
           color=color_list[3+eta_stars.index(eta_PDG.val)])

plt.axvspan(M1.am_C.val-M1.am_C.err,
            M1.am_C.val+M1.am_C.err,
            color='k', alpha=0.1)
plt.vlines(M1.am_C.val, 0, eta_PDG.val+eta_PDG.err,
           linestyle='dashed',
           color=color_list[3+eta_stars.index(eta_PDG.val)])
plt.text(-0.03, eta_PDG.val*1.01, 'PDG', fontsize=18)
plt.text(M1.am_C.val*1.01, 0.3, r'$am_C$',
         rotation=270, fontsize=18)
plt.ylim([ymin, ymax])
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')

# ===plot 1: Zms vs am for M1========================
key = 'm'
plt.figure(figsize=(w, h))
x, y = M1.load_mSMOM(key='m')
plt.errorbar(x.val, y.val, yerr=y.err,
             fmt='o', capsize=4, label='mSMOM',
             color=color_list[ens_list.index('M1')])
plt.legend(loc='center right')
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')

# ===plot 2: Zms vs am for M1 with SMOM===============
plt.figure(figsize=(w, h))
plt.errorbar(x.val, y.val, yerr=y.err,
             fmt='o', capsize=4, label='mSMOM',
             color=color_list[ens_list.index('M1')])
plt.axhspan(M1.Z_SMOM.val-M1.Z_SMOM.err,
            M1.Z_SMOM.val+M1.Z_SMOM.err,
            color='k', alpha=0.1)
plt.axhline(M1.Z_SMOM.val, color='k', label='SMOM')
plt.legend(loc='center right')
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')


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
    ax.axhline(e.Z_SMOM.val, color='k', label='SMOM')
    ax.legend(loc='center right')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([-0.05, xmax])
    ax.set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')
    ax.set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)

# ===backup 1: Zms, Zm*am_q vs am for all with SMOM======
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))
plt.subplots_adjust(hspace=0, wspace=0)
for key in ['mam_q', 'm']:
    key_idx = ['m', 'mam_q'].index(key)
    for ens in ['C1', 'M1', 'F1S']:
        idx = ens_list.index(ens)
        e = mNPR_dict[ens]
        x, y = e.load_mSMOM(key=key)
        if key == 'mam_q':
            y = y*e.ainv

        ax = axes[key_idx, idx]
        ax.errorbar(x.val, y.val, yerr=y.err,
                    fmt='o', capsize=4, label='mSMOM',
                    color=color_list[idx])

        SMOM = e.Z_SMOM if key == 'm' else e.m_bar_SMOM
        ax.axhspan(SMOM.val-SMOM.err,
                   SMOM.val+SMOM.err,
                   color='k', alpha=0.1)
        ax.axhline(SMOM.val, color='k', label='SMOM')
        xmin, xmax = ax.get_xlim()
        ax.set_xlim([-0.05, xmax])

        axes[1, idx].legend(loc='upper left')
        axes[0, idx].set_title(
            ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')
        axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m\cdot m$', fontsize=18)

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
    axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
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
    axes_info[1, idx, 0, :] = np.min([xmin0, xmin1]), np.max([xmax0, xmax1])
    axes_info[0, idx, 0, :] = axes_info[1, idx, 0, :]
    axes[0, idx].set_xlim([-0.05, np.max([xmax0, xmax1])])
    axes_info[1, idx, 1, :] = axes[1, idx].get_ylim()

axes_info_combined = np.array([[[np.min(axes_info[row_num, :, axis, 0]),
                                 np.max(axes_info[row_num, :, axis, 1])]
                                for axis in range(2)]
                               for row_num in range(2)])
xmax = np.max(axes_info_combined[:, 0, 1])
axes_info_combined[:, 0, 1] = xmax, xmax

# ===plot 2: Zms and M_eta_C vs am for some choice M_eta_C_star===
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))
plt.subplots_adjust(hspace=0, wspace=0)
eta_C_star_idx = 1
eta_C_star = eta_stars[eta_C_star_idx]

for ens in ['C1', 'M1', 'F1S']:
    idx = ens_list.index(ens)
    e = mNPR_dict[ens]
    am_star = e.interpolate_eta_c(eta_C_star)

    axes[0, idx].errorbar(e.eta_ax.val, e.eta_y.val,
                          yerr=e.eta_y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(
        e.ainv.val)+r'$ GeV)')

    xmin, xmax = axes_info_combined[0, 0, :]
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

    axes[0, idx].set_xlim(axes_info_combined[0, 0, :])
    axes[0, idx].set_ylim(axes_info_combined[0, 1, :])
    axes[0, idx].set_title(
        ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')

    x, y = e.load_mSMOM(key='m')
    axes[1, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[1, idx].set_xlim(axes_info_combined[1, 0, :])
    axes[1, idx].set_ylim(axes_info_combined[1, 1, :])
    axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)


axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[0, 0].text(axes_info_combined[0, 0, 0]+0.1, eta_C_star *
                1.05, r'$M_{\eta_C}^\star$', fontsize=16)


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
    axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(
        e.ainv.val)+r'$ GeV)')

    xmin, xmax = axes_info_combined[0, 0, :]
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

    axes[0, idx].set_xlim(axes_info_combined[0, 0, :])
    axes[0, idx].set_ylim(axes_info_combined[0, 1, :])
    axes[0, idx].set_title(
        ens+r' ($a^{-1}='+'{:.2f}'.format(e.ainv.val)+r'$ GeV)')

    x, y = e.load_mSMOM(key='m')
    axes[1, idx].errorbar(x.val, y.val, yerr=y.err,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

    xmin, xmax = axes_info_combined[1, 0, :]
    ymin, ymax = axes_info_combined[1, 1, :]
    am_grain = stat(
        val=np.linspace(0.0001, xmax+0.5, 100),
        err=np.zeros(100),
        btsp='fill')
    Zm_grain = e.Z_m_mSMOM_map(am_grain)
    Zm_grain_up = Zm_grain.val+Zm_grain.err
    Zm_grain_dn = Zm_grain.val-Zm_grain.err
    axes[1, idx].fill_between(am_grain.val, Zm_grain_up, Zm_grain_dn,
                              color='k', alpha=0.3, label='fit', zorder=0)

    axes[1, idx].vlines(am_star.val, ymin, ymax, linestyle='dashed',
                        color=color_list[3+eta_C_star_idx], zorder=0)
    Zm_star = e.Z_m_mSMOM_map(am_star)
    axes[1, idx].errorbar([am_star.val], [Zm_star.val],
                          yerr=[Zm_star.err],
                          fmt='o', capsize=4,
                          color=color_list[3+eta_C_star_idx],
                          zorder=2,)
    axes[1, idx].set_xlim(axes_info_combined[1, 0, :])
    axes[1, idx].set_ylim(axes_info_combined[1, 1, :])


axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[0, 0].text(axes_info_combined[0, 0, 0]+0.1, eta_C_star *
                1.05, r'$M_{\eta_C}^\star$', fontsize=16)


# ===plot 1: m_C continuum extrap for 1 eta_star=============
plt.figure(figsize=(h, w))

label = r'$M_{\eta_C}^\star='+str(eta_C_star)+'$ GeV'
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)

asq_grain = np.linspace(0, 0.45, 50)
extrap = cont_extrap(ens_list, mu=mu_chosen)

m_C_mSMOM, mbar_mSMOM = extrap.load_mSMOM(eta_C_star)
m_C_mSMOM_mapping = extrap.extrap_mapping(m_C_mSMOM)
m_C_mSMOM_grain = m_C_mSMOM_mapping(asq_grain)

plt.plot(asq_grain, m_C_mSMOM_grain.val,
         linestyle='dashed',
         color=color_list[3+eta_C_star_idx],
         label=label)
plt.fill_between(asq_grain,
                 m_C_mSMOM_grain.val - m_C_mSMOM_grain.err,
                 m_C_mSMOM_grain.val + m_C_mSMOM_grain.err,
                 color=color_list[3+eta_C_star_idx], alpha=0.5)

for idx, ens in enumerate(ens_list):

    plt.errorbar(extrap.a_sq.val[idx], m_C_mSMOM.val[idx],
                 xerr=extrap.a_sq.err[idx],
                 yerr=m_C_mSMOM.err[idx],
                 fmt='o', capsize=4,
                 color=color_list[idx])

m_C_mSMOM_phys = m_C_mSMOM_mapping(0.0)
plt.errorbar([0], m_C_mSMOM_phys.val,
             yerr=m_C_mSMOM_phys.err,
             fmt='o', capsize=4,
             color=color_list[3+eta_C_star_idx])

plt.legend(loc='lower right')
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
ren_xaxes = [-0.01, 0.35]
ren_yaxes = plt.ylim()
plt.xlim(ren_xaxes)


# ===STEP4 plot 2: m_C continuum extrap for 3 eta_star=============
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
for eta_idx, eta in enumerate(eta_stars):
    m_C_mSMOM, mbar_mSMOM = extrap.load_mSMOM(eta)
    m_C_mSMOM_mapping = extrap.extrap_mapping(m_C_mSMOM)
    m_C_mSMOM_grain = m_C_mSMOM_mapping(asq_grain)

    label = r'$M_{\eta_C}^\star='+str(eta)+'$ GeV'\
            if eta != eta_PDG.val else r'$M_{\eta_C}^{\star,PDG}$'

    plt.plot(asq_grain, m_C_mSMOM_grain.val,
             linestyle='dashed',
             color=color_list[3+eta_C_star_idx], label=label)
    plt.fill_between(asq_grain,
                     m_C_mSMOM_grain.val - m_C_mSMOM_grain.err,
                     m_C_mSMOM_grain.val + m_C_mSMOM_grain.err,
                     color=color_list[3+eta_idx], alpha=0.5)

    for ens_idx, ens in enumerate(ens_list):
        plt.errorbar(extrap.a_sq.val[ens_idx],
                     m_C_mSMOM.val[ens_idx],
                     m_C_mSMOM.err[ens_idx],
                     fmt='o', capsize=4,
                     color=color_list[ens_idx])

    m_C_mSMOM_phys = m_C_mSMOM_mapping(0.0)
    plt.errorbar([0], m_C_mSMOM_phys.val,
                 yerr=m_C_mSMOM_phys.err,
                 fmt='o', capsize=4,
                 color=color_list[3+eta_idx])

plt.legend(loc='center')
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)


# ===STEP4 plot 4: mbar continuum extrap for eta_stars=============
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)

for eta_idx, eta in enumerate(eta_stars):
    m_C_mSMOM, mbar_mSMOM = extrap.load_mSMOM(eta)
    mbar_mSMOM_mapping = extrap.extrap_mapping(mbar_mSMOM)
    mbar_mSMOM_grain = mbar_mSMOM_mapping(asq_grain)

    label = r'$M_{\eta_C}^\star='+str(eta)+'$ GeV'\
            if eta != eta_PDG.val else r'$M_{\eta_C}^{\star,PDG}$'

    plt.plot(asq_grain, mbar_mSMOM_grain.val,
             linestyle='dashed',
             color=color_list[3+eta_C_star_idx], label=label)
    plt.fill_between(asq_grain,
                     mbar_mSMOM_grain.val - mbar_mSMOM_grain.err,
                     mbar_mSMOM_grain.val + mbar_mSMOM_grain.err,
                     color=color_list[3+eta_idx], alpha=0.5)

    for ens_idx, ens in enumerate(ens_list):
        plt.errorbar(extrap.a_sq.val[ens_idx],
                     mbar_mSMOM.val[ens_idx],
                     mbar_mSMOM.err[ens_idx],
                     fmt='o', capsize=4,
                     color=color_list[ens_idx])

    mbar_mSMOM_phys = mbar_mSMOM_mapping(0.0)
    plt.errorbar([0], mbar_mSMOM_phys.val,
                 yerr=mbar_mSMOM_phys.err,
                 fmt='o', capsize=4,
                 color=color_list[3+eta_idx])

plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$\overline{m}(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)


fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pdf, format='pdf')
pdf.close()
plt.close('all')
os.system('open '+filename)
