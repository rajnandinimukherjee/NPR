from matching import *
from NPR_classes import *
from basics import *
from eta_c import *

mres = True
folder = 'mres' if mres else 'no_mres'

ens_list = list(eta_c_data.keys())
if mres:
    for ens in ['C1', 'M1', 'F1S']:
        if ens in valence_ens:
            e = etaCvalence(ens)
            e.toDict(keys=list(e.mass_comb.keys()), mres=mres)

mu_chosen = 3.0
filename = '/Users/rajnandinimukherjee/Desktop/LatPlots.pdf'
pdf = PdfPages(filename)

h, w = 4, 2.75
num_ens = len(ens_list)
# ===STEP 1====================================================================
plot_dict = {}
for ens in ens_list:
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']
    am_C, am_C_err = interpolate_eta_c(ens, eta_PDG)
    plot_dict[ens] = {'ainv': ainv,
                      'x': x,
                      'M_eta_C': y*ainv,
                      'M_eta_C_err': yerr*ainv,
                      'am_C': am_C,
                      'am_C_err': am_C_err}

# ===plot 1: M1 M_eta_C measurements===================
plt.figure(figsize=(w, h))
x = plot_dict['M1']['x']
y = plot_dict['M1']['M_eta_C']
yerr = plot_dict['M1']['M_eta_C_err']
plt.errorbar(x, y, yerr=yerr,
             fmt='o', capsize=4,
             color=color_list[ens_list.index('M1')])
xmin, xmax = plt.xlim()
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')

# ===plot 2: M1 M_eta_C meas+ PDG interpolation=========
plt.figure(figsize=(w, h))
plt.errorbar(x, y, yerr=yerr,
             fmt='o', capsize=4,
             color=color_list[ens_list.index('M1')])
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
am_C = plot_dict['M1']['am_C']
am_C_err = plot_dict['M1']['am_C_err']
plt.hlines(eta_PDG, -0.05, am_C, color=color_list[
    3+eta_stars.index(eta_PDG)])
plt.axvspan(am_C-am_C_err, am_C+am_C_err,
            color='k', alpha=0.1)
plt.vlines(am_C, 0, eta_PDG, linestyle='dashed',
           color=color_list[3+eta_stars.index(eta_PDG)])
plt.text(-0.03, eta_PDG*1.01, 'PDG', fontsize=18)
plt.text(am_C*1.01, 0.3, r'$am_C$',
         rotation=270, fontsize=18)
plt.ylim([ymin, ymax])
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')


# ===STEP 2 ================================================================

for ens in ens_list:
    bl_SMOM = bilinear_analysis(ens, mres=mres,
                                loadpath=f'{folder}/{ens}_bl_massive_SMOM.p')
    bl_mSMOM = bilinear_analysis(ens, mres=mres,
                                 loadpath=f'{folder}/{ens}_bl_massive_mSMOM.p')
    for key in ['m', 'mam_q']:
        x, Zm_SMOM, Zm_SMOM_err = bl_SMOM.massive_Z_plots(key=key,
                                                          mu=mu_chosen, passinfo=True)
        Zm_SMOM, Zm_SMOM_err = Zm_SMOM[0], Zm_SMOM_err[0]
        x, Zm, Zm_err = bl_mSMOM.massive_Z_plots(key=key,
                                                 mu=mu_chosen, passinfo=True)
        plot_dict[ens][key] = {'x_Z': x,
                               'Zm': Zm,
                               'Zm_err': Zm_err,
                               'Zm_SMOM': Zm_SMOM,
                               'Zm_SMOM_err': Zm_SMOM_err}

# ===plot 1: Zms vs am for M1========================
key = 'm'
plt.figure(figsize=(w, h))
x = plot_dict['M1'][key]['x_Z']
y = plot_dict['M1'][key]['Zm']
yerr = plot_dict['M1'][key]['Zm_err']
plt.errorbar(x, y, yerr=yerr,
             fmt='o', capsize=4, label='mSMOM',
             color=color_list[ens_list.index('M1')])
plt.text(0.0, 1.5, r'$\mu=2.0$ GeV')
plt.legend(loc='center right')
plt.xlim([-0.05, xmax])
plt.xlabel(r'$am$', fontsize=18)
plt.ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
plt.title(r'M1 ($a^{-1}=2.38$ GeV)')

# ===plot 2: Zms vs am for M1 with SMOM===============
plt.figure(figsize=(w, h))
plt.errorbar(x, y, yerr=yerr,
             fmt='o', capsize=4, label='mSMOM',
             color=color_list[ens_list.index('M1')])
Z_SMOM = plot_dict['M1'][key]['Zm_SMOM']
Z_SMOM_err = plot_dict['M1'][key]['Zm_SMOM_err']
plt.axhspan(Z_SMOM-Z_SMOM_err, Z_SMOM+Z_SMOM_err,
            color='k', alpha=0.1)
plt.axhline(Z_SMOM, color='k', label='SMOM')
plt.text(0.0, 1.5, r'$\mu=2.0$ GeV')
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
    ainv = params[ens]['ainv']

    x = plot_dict[ens][key]['x_Z']
    y = plot_dict[ens][key]['Zm']
    yerr = plot_dict[ens][key]['Zm_err']
    ax.errorbar(x, y, yerr=yerr,
                fmt='o', capsize=4, label='mSMOM',
                color=color_list[idx])
    Z_SMOM = plot_dict[ens][key]['Zm_SMOM']
    Z_SMOM_err = plot_dict[ens][key]['Zm_SMOM_err']
    ax.axhspan(Z_SMOM-Z_SMOM_err, Z_SMOM+Z_SMOM_err,
               color='k', alpha=0.1)
    ax.axhline(Z_SMOM, color='k', label='SMOM')
    ax.legend(loc='center right')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([-0.05, xmax])
    ax.set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(ainv)+r'$ GeV)')
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
        ainv = params[ens]['ainv']

        ax = axes[key_idx, idx]
        x = plot_dict[ens][key]['x_Z']
        y = plot_dict[ens][key]['Zm']
        yerr = plot_dict[ens][key]['Zm_err']
        ax.errorbar(x, y, yerr=yerr,
                    fmt='o', capsize=4, label='mSMOM',
                    color=color_list[idx])
        xmin, xmax = ax.get_xlim()
        Z_SMOM = plot_dict[ens][key]['Zm_SMOM']
        Z_SMOM_err = plot_dict[ens][key]['Zm_SMOM_err']
        ax.axhspan(Z_SMOM-Z_SMOM_err, Z_SMOM+Z_SMOM_err,
                   color='k', alpha=0.1)
        ax.axhline(Z_SMOM, color='k', label='SMOM')
        ax.set_xlim([-0.05, xmax])

        axes[1, idx].legend(loc='upper left')
        axes[0, idx].set_title(
            ens+r' ($a^{-1}='+'{:.2f}'.format(ainv)+r'$ GeV)')
        axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m\cdot m$', fontsize=18)

# ===STEP 3 ================================================================

for ens in ens_list:
    eta_star_dict = {eta: interpolate_eta_c(ens, eta)
                     for eta in eta_stars}
    plot_dict[ens]['eta_stars'] = eta_star_dict


# ===plot 1: Zms and M_eta_C vs am============================
axes_info = np.zeros(shape=(2, num_ens, 2, 2))
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))
plt.subplots_adjust(hspace=0, wspace=0)
for ens in ['C1', 'M1', 'F1S']:
    idx = ens_list.index(ens)
    ainv = params[ens]['ainv']

    x = plot_dict[ens]['x']
    y = plot_dict[ens]['M_eta_C']
    yerr = plot_dict[ens]['M_eta_C_err']
    axes[0, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    xmin, xmax = axes[0, idx].get_xlim()
    axes[0, idx].set_xlim([-0.05, xmax])
    axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(ainv)+r'$ GeV)')
    axes_info[0, idx, 0, :] = axes[0, idx].get_xlim()
    axes_info[0, idx, 1, :] = axes[0, idx].get_ylim()

    key = 'm'
    x = plot_dict[ens][key]['x_Z']
    y = plot_dict[ens][key]['Zm']
    yerr = plot_dict[ens][key]['Zm_err']
    axes[1, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx])
    xmin, xmax = axes[1, idx].get_xlim()
    axes[1, idx].set_xlim([-0.05, xmax])
    axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)
    axes_info[1, idx, 0, :] = axes[1, idx].get_xlim()
    axes_info[1, idx, 1, :] = axes[1, idx].get_ylim()

# ===plot 2: Zms and M_eta_C vs am for some choice M_eta_C_star===
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))
plt.subplots_adjust(hspace=0, wspace=0)
eta_C_star_idx = 2

for ens in ['C1', 'M1', 'F1S']:
    idx = ens_list.index(ens)
    ainv = params[ens]['ainv']

    x = plot_dict[ens]['x']
    y = plot_dict[ens]['M_eta_C']
    yerr = plot_dict[ens]['M_eta_C_err']
    axes[0, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx])

    xmin, xmax = axes_info[0, idx, 0, :]
    ymin, ymax = axes_info[0, idx, 1, :]

    eta_C_star = eta_stars[eta_C_star_idx]
    am_star, am_star_err = plot_dict[ens]['eta_stars'][eta_C_star]
    axes[0, idx].hlines(eta_C_star, xmin, xmax,
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].axvspan(am_star-am_star_err, am_star+am_star_err,
                         color='k', alpha=0.1)
    axes[0, idx].vlines(am_star, ymin, eta_C_star, linestyle='dashed',
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].text(am_star*1.01, 0.3,
                      r'$a_{'+ens+r'}m^\star$', rotation=270, fontsize=18)

    axes[0, idx].set_xlim([xmin, xmax])
    axes[0, idx].set_ylim([ymin, ymax])
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(ainv)+r'$ GeV)')

    key = 'm'
    x = plot_dict[ens][key]['x_Z']
    y = plot_dict[ens][key]['Zm']
    yerr = plot_dict[ens][key]['Zm_err']
    axes[1, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx])

    xmin, xmax = axes_info[1, idx, 0, :]
    ymin, ymax = axes_info[1, idx, 1, :]
    axes[1, idx].set_xlim([xmin, xmax])
    axes[1, idx].set_ylim([ymin, ymax])
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[0, 0].text(-0.03, eta_C_star*1.05, r'$M_{\eta_C}^\star$', fontsize=16)

# ===fitting===================================================================

plot_dict['C1']['fit_idx'] = [1, 2, 3]
plot_dict['M1']['fit_idx'] = [1, 2, 3, 4]
plot_dict['F1S']['fit_idx'] = [1, 2, 3, 4]


def Zm_ansatz(params, am, key='m', **kwargs):
    if key == 'm':
        return params[0] + params[1]*am + params[2]/am
        # return params[0] + params[1]*(am**2) + params[2]*np.log(am)*am
    else:
        return params[0]*am + (am**2)*params[1] + params[2]


for ens in ['M1', 'C1', 'F1S']:
    key = 'm'
    x = plot_dict[ens][key]['x_Z']
    y = plot_dict[ens][key]['Zm']
    yerr = plot_dict[ens][key]['Zm_err']

    fit_idx = plot_dict[ens]['fit_idx']

    def diff(params):
        return y[fit_idx] - Zm_ansatz(params, x[fit_idx], key=key)

    cov = np.diag(yerr[fit_idx]**2)
    L_inv = np.linalg.cholesky(cov)
    L = np.linalg.inv(L_inv)

    def LD(params, **akwargs):
        return L.dot(diff(params))

    guess = [1, 1, 1]
    res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
    chi_sq = LD(res.x).dot(LD(res.x))
    dof = len(fit_idx)-len(guess)
    pvalue = gammaincc(dof/2, chi_sq/2)

    am_stars = np.array([v[0] for k, v in plot_dict[ens]['eta_stars'].items()])
    am_stars_err = np.array([v[1]
                            for k, v in plot_dict[ens]['eta_stars'].items()])
    xmin, xmax = axes_info[1, idx, 0, :]
    am_grain = np.linspace(0.0001, xmax+0.5, 100)

    Zm_stars = Zm_ansatz(res.x, am_stars, key=key)
    Zm_grain = Zm_ansatz(res.x, am_grain, key=key)

    y_btsp = np.array([np.random.normal(y[i], yerr[i], N_boot)
                       for i in range(len(y))])
    am_stars_btsp = np.array([np.random.normal(v[0], v[1], N_boot)
                              for k, v in plot_dict[ens]['eta_stars'].items()])

    Zm_stars_btsp = np.zeros(shape=(len(Zm_stars), N_boot))
    Zm_grain_btsp = np.zeros(shape=(len(Zm_grain), N_boot))
    for k in range(N_boot):
        def diff_k(params):
            return y_btsp[fit_idx, k] - Zm_ansatz(params, x[fit_idx], key=key)

        def LD_k(params, **akwargs):
            return L.dot(diff_k(params))
        res_k = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
        Zm_stars_btsp[:, k] = Zm_ansatz(res_k.x, am_stars_btsp[:, k], key=key)
        Zm_grain_btsp[:, k] = Zm_ansatz(res_k.x, am_grain, key=key)

    Zm_stars_err = np.array([st_dev(Zm_stars_btsp[i, :], Zm_stars[i])
                             for i in range(len(Zm_stars))])
    Zm_grain_err = np.array([st_dev(Zm_grain_btsp[i, :], Zm_grain[i])
                             for i in range(len(Zm_grain))])

    plot_dict[ens]['fits'] = {'am_stars': am_stars,
                              'am_stars_err': am_stars_err,
                              'Zm_stars': Zm_stars,
                              'Zm_stars_err': Zm_stars_err,
                              'Zm_stars_btsp': Zm_stars_btsp,
                              'am_grain': am_grain,
                              'Zm_grain': Zm_grain,
                              'Zm_grain_up': Zm_grain+Zm_grain_err,
                              'Zm_grain_dn': Zm_grain-Zm_grain_err,
                              'pvalue': pvalue}

# ===plot 3: Zms interpolate for some choice M_eta_C_star===
fig, axes = plt.subplots(nrows=2, ncols=num_ens,
                         sharex='col', sharey='row',
                         figsize=(w*num_ens, h*2))

plt.subplots_adjust(hspace=0, wspace=0)
eta_C_star_idx = 2

for ens in ['M1', 'C1', 'F1S']:
    idx = ens_list.index(ens)
    ainv = params[ens]['ainv']

    x = plot_dict[ens]['x']
    y = plot_dict[ens]['M_eta_C']
    yerr = plot_dict[ens]['M_eta_C_err']
    axes[0, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx])

    xmin, xmax = axes_info[0, idx, 0, :]
    ymin, ymax = axes_info[0, idx, 1, :]

    eta_C_star = eta_stars[eta_C_star_idx]
    am_star, am_star_err = plot_dict[ens]['eta_stars'][eta_C_star]
    axes[0, idx].hlines(eta_C_star, xmin, xmax,
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].axvspan(am_star-am_star_err, am_star+am_star_err,
                         color='k', alpha=0.1)
    axes[0, idx].vlines(am_star, ymin, eta_C_star, linestyle='dashed',
                        color=color_list[3+eta_C_star_idx])
    axes[0, idx].text(am_star*1.01, 0.3,
                      r'$a_{'+ens+r'}m^\star$', rotation=270, fontsize=18)

    axes[0, idx].set_xlim([xmin, xmax])
    axes[0, idx].set_ylim([ymin, ymax])
    axes[0, idx].set_title(ens+r' ($a^{-1}='+'{:.2f}'.format(ainv)+r'$ GeV)')

    key = 'm'
    x = plot_dict[ens][key]['x_Z']
    y = plot_dict[ens][key]['Zm']
    yerr = plot_dict[ens][key]['Zm_err']
    axes[1, idx].errorbar(x, y, yerr=yerr,
                          fmt='o', capsize=4,
                          color=color_list[idx], zorder=1)

    xmin, xmax = axes_info[1, idx, 0, :]
    ymin, ymax = axes_info[1, idx, 1, :]

    am_grain = plot_dict[ens]['fits']['am_grain']
    Zm_grain_up = plot_dict[ens]['fits']['Zm_grain_up']
    Zm_grain_dn = plot_dict[ens]['fits']['Zm_grain_dn']
    axes[1, idx].fill_between(am_grain, Zm_grain_up, Zm_grain_dn,
                              color='k', alpha=0.3, label='fit', zorder=0)

    axes[1, idx].vlines(am_star, ymin, ymax, linestyle='dashed',
                        color=color_list[3+eta_C_star_idx], zorder=0)
    Zm_star = plot_dict[ens]['fits']['Zm_stars'][eta_C_star_idx]
    Zm_star_err = plot_dict[ens]['fits']['Zm_stars_err'][eta_C_star_idx]
    axes[1, idx].errorbar([am_star], [Zm_star], yerr=[Zm_star_err],
                          fmt='o', capsize=4,
                          color=color_list[3+eta_C_star_idx],
                          zorder=2,)

    axes[1, idx].set_xlim([xmin, xmax])
    axes[1, idx].set_ylim([ymin, ymax])
    axes[1, idx].set_xlabel('$a_{'+ens+'}m$', fontsize=18)

axes[0, 0].set_ylabel(r'$M_{\eta_C}$ (GeV)', fontsize=18)
axes[1, 0].set_ylabel(r'$Z_m(am,a\mu)$', fontsize=18)
axes[0, 0].text(-0.03, eta_C_star*1.1, r'$M_{\eta_C}^\star$', fontsize=16)

# ===STEP 3: ren  ================================================================
for ens in ens_list:
    ainv = params[ens]['ainv']
    am_C, am_C_err = plot_dict[ens]['am_C'], plot_dict[ens]['am_C_err']
    am_C_btsp = np.random.normal(am_C, am_C_err, N_boot)

    Zm_stars = plot_dict[ens]['fits']['Zm_stars']
    Zm_stars_err = plot_dict[ens]['fits']['Zm_stars_err']
    Zm_stars_btsp = plot_dict[ens]['fits']['Zm_stars_btsp']

    m_C_ren = ainv*am_C*Zm_stars
    m_C_btsp = np.array([ainv*am_C_btsp[:]*Zm_stars_btsp[i, :]
                         for i in range(len(Zm_stars))])
    m_C_err = np.array([st_dev(m_C_btsp[i, :], m_C_ren[i])
                        for i in range(len(m_C_ren))])

    am_stars = plot_dict[ens]['fits']['am_stars']
    am_stars_err = plot_dict[ens]['fits']['am_stars_err']
    am_stars_btsp = np.array([np.random.normal(am_stars[i],
                              am_stars_err[i], N_boot)
                              for i in range(len(am_stars))])

    m_star_ren = ainv*am_stars*Zm_stars
    m_star_btsp = np.array([ainv*am_stars_btsp[i, :]*Zm_stars_btsp[i, :]
                           for i in range(len(Zm_stars))])
    m_star_err = np.array([st_dev(m_star_btsp[i, :], m_star_ren[i])
                           for i in range(len(m_star_ren))])

    plot_dict[ens]['ren'] = {'m_C_ren': m_C_ren,
                             'm_C_ren_err': m_C_err,
                             'm_C_ren_btsp': m_C_btsp,
                             'm_q_ren': m_star_ren,
                             'm_q_ren_err': m_star_err,
                             'm_q_ren_btsp': m_star_btsp}


# ===STEP 4====================================================================
def continuum_ansatz(params, a_sq, **kwargs):
    return params[0]*(1 + params[1]*a_sq + params[2]*(a_sq**2))


plot_dict['fits'] = {}
x = np.array([plot_dict[ens]['ainv']**(-2) for ens in ens_list])
asq_grain = np.linspace(0, 0.3*1.5, 100)

for star in eta_stars:
    eta_idx = eta_stars.index(star)
    plot_dict['fits'][star] = {}

    for key in ['m_C', 'm_q']:
        y = np.array([plot_dict[ens]['ren'][key+'_ren'][eta_idx]
                      for ens in ens_list])
        yerr = np.array([plot_dict[ens]['ren'][key+'_ren_err'][eta_idx]
                         for ens in ens_list])

        def diff(params):
            return y - continuum_ansatz(params, x)
        cov = np.diag(yerr**2)
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)

        def LD(params, **akwargs):
            return L.dot(diff(params))
        guess = [1, 1, 1]
        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        dof = len(fit_idx)-len(guess)
        pvalue = gammaincc(dof/2, chi_sq/2)
        m_cont = continuum_ansatz(res.x, 0)
        m_grain = continuum_ansatz(res.x, asq_grain)
        y_btsp = np.array([plot_dict[ens]['ren'][key+'_ren_btsp'][eta_idx]
                           for ens in ens_list])
        m_cont_btsp = np.zeros(N_boot)
        m_grain_btsp = np.zeros(shape=(len(m_grain), N_boot))
        for k in range(N_boot):
            def diff_k(params):
                return y_btsp[:, k] - continuum_ansatz(params, x)

            def LD_k(params, **akwargs):
                return L.dot(diff_k(params))
            res_k = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
            m_cont_btsp[k] = continuum_ansatz(res_k.x, 0)
            m_grain_btsp[:, k] = continuum_ansatz(res_k.x, asq_grain)
        m_cont_err = st_dev(m_cont_btsp, m_cont)
        m_grain_err = np.array([st_dev(m_grain_btsp[i, :], m_grain[i])
                                for i in range(len(m_grain))])
        m_grain_up = m_grain+m_grain_err
        m_grain_dn = m_grain-m_grain_err

        plot_dict['fits'][star].update({key+'_cont': m_cont,
                                        key+'_cont_err': m_cont_err,
                                        key+'_cont_btsp': m_cont_btsp,
                                        key+'_grain': m_grain,
                                        key+'_grain_up': m_grain_up,
                                        key+'_grain_dn': m_grain_dn})

# ===plot 1: m_C continuum extrap for 1 eta_star=============
plt.figure(figsize=(h, w))
eta_C_star_idx = 2
eta_star_chosen = eta_stars[eta_C_star_idx]

label = r'$M_{\eta_C}^\star='+str(eta_star_chosen)+'$ GeV'
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
plt.plot(asq_grain, plot_dict['fits'][eta_star_chosen]['m_C_grain'],
         linestyle='dashed', color=color_list[3+eta_C_star_idx], label=label)
plt.fill_between(asq_grain, plot_dict['fits'][eta_star_chosen]['m_C_grain_up'],
                 plot_dict['fits'][eta_star_chosen]['m_C_grain_dn'],
                 color=color_list[3+eta_C_star_idx], alpha=0.5)
for ens in ens_list:
    idx = ens_list.index(ens)
    x = np.array([params[ens]['ainv']**(-2)])
    y = np.array([plot_dict[ens]['ren']['m_C_ren'][eta_C_star_idx]])
    yerr = np.array([plot_dict[ens]['ren']['m_C_ren_err'][eta_C_star_idx]])

    plt.errorbar(x, y, yerr, fmt='o', capsize=4,
                 color=color_list[idx])

plt.errorbar([0], [plot_dict['fits'][eta_star_chosen]['m_C_cont']],
             yerr=[plot_dict['fits'][eta_star_chosen]['m_C_cont_err']],
             fmt='o', capsize=4, color=color_list[3+eta_C_star_idx])

plt.legend(loc='lower right')
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
ren_xaxes = [-0.01, 0.35]
ren_yaxes = plt.ylim()
plt.xlim(ren_xaxes)


# ===STEP3 plot 1: m_c^ren from 3 lattice spacings============
plt.figure(figsize=(h, w))
eta_C_star_idx = 2

plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
for ens in ens_list:
    idx = ens_list.index(ens)
    x = np.array([params[ens]['ainv']**(-2)])
    y = np.array([plot_dict[ens]['ren']['m_C_ren'][eta_C_star_idx]])
    yerr = np.array([plot_dict[ens]['ren']['m_C_ren_err'][eta_C_star_idx]])

    plt.errorbar(x, y, yerr, fmt='o', capsize=4,
                 color=color_list[idx])
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)
plt.ylim(ren_yaxes)


# ===STEP4 plot 2: m_C continuum extrap for 3 eta_star=============
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
for eta_C_star_idx in range(len(eta_stars)):
    eta_star_chosen = eta_stars[eta_C_star_idx]

    label = r'$M_{\eta_C}^\star='+str(eta_star_chosen)+'$ GeV'\
            if eta_star_chosen != eta_PDG else r'$M_{\eta_C}^{\star,PDG}$'
    plt.plot(asq_grain, plot_dict['fits'][eta_star_chosen]['m_C_grain'],
             linestyle='dashed', color=color_list[3+eta_C_star_idx], label=label)
    plt.fill_between(asq_grain, plot_dict['fits'][eta_star_chosen]['m_C_grain_up'],
                     plot_dict['fits'][eta_star_chosen]['m_C_grain_dn'],
                     color=color_list[3+eta_C_star_idx], alpha=0.5)
    for ens in ens_list:
        idx = ens_list.index(ens)
        x = np.array([params[ens]['ainv']**(-2)])
        y = np.array([plot_dict[ens]['ren']['m_C_ren'][eta_C_star_idx]])
        yerr = np.array([plot_dict[ens]['ren']['m_C_ren_err'][eta_C_star_idx]])

        plt.errorbar(x, y, yerr, fmt='o', capsize=4,
                     color=color_list[idx])

    plt.errorbar([0], [plot_dict['fits'][eta_star_chosen]['m_C_cont']],
                 yerr=[plot_dict['fits'][eta_star_chosen]['m_C_cont_err']],
                 fmt='o', capsize=4, color=color_list[3+eta_C_star_idx])

plt.legend(loc='center')
plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$m_{C}^S(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)


# ===STEP4 plot 4: mbar continuum extrap for eta_stars=============
plt.figure(figsize=(h, w))
plt.axvline(x=0, linestyle='dashed', color='k', alpha=0.4)
for eta_C_star_idx in range(len(eta_stars)):
    eta_star_chosen = eta_stars[eta_C_star_idx]

    label = r'$M_{\eta_C}^\star='+str(eta_star_chosen)+'$ GeV'\
            if eta_star_chosen != eta_PDG else r'$M_{\eta_C}^{\star,PDG}$'
    plt.plot(asq_grain, plot_dict['fits'][eta_star_chosen]['m_q_grain'],
             linestyle='dashed', color=color_list[3+eta_C_star_idx], label=label)
    plt.fill_between(asq_grain, plot_dict['fits'][eta_star_chosen]['m_q_grain_up'],
                     plot_dict['fits'][eta_star_chosen]['m_q_grain_dn'],
                     color=color_list[3+eta_C_star_idx], alpha=0.5)
    for ens in ens_list:
        idx = ens_list.index(ens)
        x = np.array([params[ens]['ainv']**(-2)])
        y = np.array([plot_dict[ens]['ren']['m_q_ren'][eta_C_star_idx]])
        yerr = np.array([plot_dict[ens]['ren']['m_q_ren_err'][eta_C_star_idx]])

        plt.errorbar(x, y, yerr, fmt='o', capsize=4,
                     color=color_list[idx])

    plt.errorbar([0], [plot_dict['fits'][eta_star_chosen]['m_q_cont']],
                 yerr=[plot_dict['fits'][eta_star_chosen]['m_q_cont_err']],
                 fmt='o', capsize=4, color=color_list[3+eta_C_star_idx])

plt.xlabel(r'$a^2$ (GeV${}^2$)', fontsize=18)
plt.ylabel(r'$\overline{m}(am^\star,a\mu)$', fontsize=18)
plt.xlim(ren_xaxes)

# ===STEP5================================================================================================
for eta_C_star_idx in range(1, len(eta_stars)):
    eta_star_chosen = eta_stars[eta_C_star_idx]
    mbar = plot_dict['fits'][eta_star_chosen]['m_q_cont']
    mbar_btsp = plot_dict['fits'][eta_star_chosen]['m_q_cont_btsp']

    m_C_ren = plot_dict['fits'][eta_star_chosen]['m_C_cont']
    m_C_ren_btsp = plot_dict['fits'][eta_star_chosen]['m_C_cont_btsp']

    m_C_MS = R_mSMOM_to_MSbar(mu_chosen, mbar)*m_C_ren
    m_C_MS_btsp = np.array([R_mSMOM_to_MSbar(mu_chosen,
                           mbar_btsp[k])*m_C_ren_btsp[k]
        for k in range(N_boot)]).reshape((200,))
    m_C_MS_err = st_dev(m_C_MS_btsp, m_C_MS)
    plot_dict['fits'][eta_star_chosen].update({'m_C_MS': m_C_MS,
                                               'm_C_MS_err': m_C_MS_err,
                                               'm_C_MS_btsp': m_C_MS_btsp})
# ===STEP5 plot 1: mbar vs m_C_ren==================================
plt.figure(figsize=(h, w))
for eta_C_star_idx in range(1, len(eta_stars)):
    eta_star_chosen = eta_stars[eta_C_star_idx]

    mbar = plot_dict['fits'][eta_star_chosen]['m_q_cont']
    mbar_err = plot_dict['fits'][eta_star_chosen]['m_q_cont_err']

    m_C_ren = plot_dict['fits'][eta_star_chosen]['m_C_cont']
    m_C_ren_err = plot_dict['fits'][eta_star_chosen]['m_C_cont_err']
    plt.errorbar([mbar], [m_C_ren], yerr=[m_C_ren_err], xerr=[mbar_err],
                 capsize=2, markerfacecolor='None', fmt='o',
                 color=color_list[3+eta_C_star_idx])

plt.xlabel(r'$\overline{m}$ (GeV)')
plt.ylabel(r'$m_{C,cont}^S(\overline{m},\mu=2.0$ GeV$)$')

# ===STEP5 plot 2: matching to MSbar=================================
plt.figure(figsize=(h, w))
for eta_C_star_idx in range(1, len(eta_stars)):
    eta_star_chosen = eta_stars[eta_C_star_idx]

    mbar = plot_dict['fits'][eta_star_chosen]['m_q_cont']
    mbar_err = plot_dict['fits'][eta_star_chosen]['m_q_cont_err']

    m_C_MS = plot_dict['fits'][eta_star_chosen]['m_C_MS']
    m_C_MS_err = plot_dict['fits'][eta_star_chosen]['m_C_MS_err']

    plt.errorbar([eta_star_chosen], [m_C_MS], yerr=[m_C_MS_err], xerr=[mbar_err],
                 capsize=2, markerfacecolor='None', fmt='o',
                 color=color_list[3+eta_C_star_idx])

plt.xlabel(r'$\overline{m}$ (GeV)')
plt.ylabel(r'$m_{C,cont}^{\overline{MS}}(\mu=2.0$ GeV$)$')


fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pdf, format='pdf')
pdf.close()
plt.close('all')
os.system('open '+filename)
