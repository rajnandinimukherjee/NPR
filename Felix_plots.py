from massive import *

M1M = Z_bl_analysis("M1M", renorm="mSMOM")

# ====== Z_m vs \mu =============================
masses = ("0.1500", "0.1500")
fig, ax = plt.subplots(figsize=(4.5, 3))
x = M1M.momenta[masses] * M1M.ainv
y = M1M.Z[masses]["m"]
ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err, capsize=4, fmt="o")
ax.set_xlabel(r"$\mu\,[\mathrm{GeV}]$")
ax.set_ylabel(r"$Z_m^{\mathrm{mSMOM}}(am_q=" + str(float(masses[0])) + r")$")
ax.text(0.9, 0.9, "M1M", ha="center", va="center", transform=ax.transAxes)

mu = 2.0
linear_Z = M1M.fit_momentum_dependence(mu, masses, "m", fittype="linear")
quadratic_Z = M1M.fit_momentum_dependence(mu, masses, "m", fittype="quadratic")

stat_err = max(linear_Z.err, quadratic_Z.err)
sys_err = np.abs(quadratic_Z.val - linear_Z.val) / 2
Z_mu = stat(
    val=(linear_Z.val + quadratic_Z.val) / 2,
    err=(stat_err**2 + sys_err**2) ** 0.5,
    btsp="fill",
)
ax.errorbar([mu], Z_mu.val, yerr=Z_mu.err, capsize=4, fmt="o", color="k")
ax_twin = ax.twiny()
ax_twin.set_xlim(ax.get_xlim())
ax_twin.set_xticks([mu])
ax_twin.set_xticklabels([str(mu) + r" GeV"])
ax.axvline(mu, linestyle="dashed", c="k", alpha=0.2)

# ====== M_eta and Z_m ==========================
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3, 7.5))
plt.subplots_adjust(hspace=0)
x = join_stats([M1M.valence.amres[m] + eval(m) for m in M1M.all_masses[:-2]])
ax[1].set_xlabel(r"$am_q + am_\mathrm{res}$")

M_eta = join_stats(
    [M1M.valence.eta_h_masses[m] * M1M.ainv for m in M1M.all_masses[:-2]]
)
ax[0].errorbar(x.val, M_eta.val, xerr=x.err, yerr=M_eta.err, capsize=4, fmt="o")
ax[0].set_ylabel(r"$M_{\eta_h}\,[\mathrm{GeV}]$")


def eta_mass_ansatz(am, param, **kwargs):
    return param[0] + param[1] * am**0.5 + param[2] * am


ax[0].text(
    1.05,
    0.5,
    r"$\alpha + \beta\sqrt{am} + \gamma\,am$",
    va="center",
    ha="center",
    rotation=90,
    color="k",
    alpha=0.3,
    transform=ax[0].transAxes,
)

res = fit_func(x, M_eta, eta_mass_ansatz, [0.1, 1, 1])
xmin, xmax = ax[0].get_xlim()
xrange = np.linspace(x.val[0], xmax, 100)
yrange = res.mapping(xrange)
ax[0].text(
    0.5,
    0.05,
    r"$\chi^2/\mathrm{DOF}:" + str(np.around(res.chi_sq / res.DOF, 3)) + r"$",
    ha="center",
    va="center",
    transform=ax[0].transAxes,
)
ax[0].fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)
ax[0].set_xlim(0, xmax)
ax[0].set_title("M1M")

Z_m = join_stats([M1M.interpolate(mu, (m, m), "m") for m in M1M.all_masses[:-2]])
ax[1].errorbar(x.val, Z_m.val, xerr=x.err, yerr=Z_m.err, capsize=4, fmt="o")
ax[1].set_ylabel(r"$Z_m(\mu=" + str(mu) + r"\,\mathrm{GeV})$")


def Z_m_ansatz(am, param, **kwargs):
    return param[0] / am + param[1] + param[2] * am + param[3] * am**2


ax[1].text(
    1.05,
    0.5,
    r"$\alpha/am + \beta + \gamma\,am + \delta\,(am)^2$",
    va="center",
    ha="center",
    rotation=90,
    color="k",
    alpha=0.3,
    transform=ax[1].transAxes,
)

res = fit_func(x, Z_m, Z_m_ansatz, [0.1, 0.1, 0.1, 0.1])
xmin, xmax = ax[1].get_xlim()
xrange = np.linspace(x.val[0], xmax, 100)
yrange = res.mapping(xrange)
ax[1].text(
    0.5,
    0.05,
    r"$\chi^2/\mathrm{DOF}:" + str(np.around(res.chi_sq / res.DOF, 3)) + r"$",
    ha="center",
    va="center",
    transform=ax[1].transAxes,
)
ax[1].fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)

# ==== all ensembles M ================================
eta_pdg = eta_PDG * 0.75
eta_star = eta_PDG * 0.6
ens_list = ["C1M", "M1M", "F1M"]
fig, ax = plt.subplots(ncols=len(ens_list), sharey=True, figsize=(3 * len(ens_list), 4))
plt.subplots_adjust(wspace=0.08)
for ens in reversed(ens_list):
    i = ens_list.index(ens)
    e = Z_bl_analysis(ens, renorm="mSMOM")

    stop = 3 if ens == "F1M" else 2
    x = join_stats([e.valence.amres[m] + eval(m) for m in e.all_masses[:-stop]])
    ax[i].set_xlabel(r"$am_q + am_\mathrm{res}$")

    M_eta = join_stats(
        [e.valence.eta_h_masses[m] * e.ainv for m in e.all_masses[:-stop]]
    )
    ax[i].errorbar(x.val, M_eta.val, xerr=x.err, yerr=M_eta.err, capsize=4, fmt="o")

    def eta_mass_ansatz(am, param, **kwargs):
        return param[0] + param[1] * am**0.5 + param[2] * am

    res = fit_func(x, M_eta, eta_mass_ansatz, [0.1, 1, 1])

    def pred_amq(eta_mass, param, **kwargs):
        a, b, c = param
        root = (-b + (b**2 + 4 * c * (eta_mass - a)) ** 0.5) / (2 * c)
        return root**2

    am_c = stat(
        val=pred_amq(eta_pdg.val, res.val),
        err="fill",
        btsp=[pred_amq(eta_pdg.btsp[k], res.btsp[k]) for k in range(N_boot)],
    )
    am_star = stat(
        val=pred_amq(eta_star.val, res.val),
        err="fill",
        btsp=[pred_amq(eta_star.btsp[k], res.btsp[k]) for k in range(N_boot)],
    )
    xmin, xmax = ax[i].get_xlim()
    xrange = np.linspace(x.val[0], xmax, 100)
    yrange = res.mapping(xrange)
    ax[i].fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
    )
    ax[i].set_xlim(0, xmax)
    ax[i].set_title(ens)
    ymin, ymax = ax[i].get_ylim()

    ax[i].hlines(y=eta_pdg.val, xmin=0, xmax=xmax if i != 2 else am_c.val, color="k")
    ax[i].fill_between(
        xrange if i != 2 else np.linspace(0, am_c.val, 100),
        eta_pdg.val + eta_pdg.err,
        eta_pdg.val - eta_pdg.err,
        color="k",
        alpha=0.2,
    )
    ax[i].vlines(x=am_c.val, ymin=ymin, ymax=eta_pdg.val, color="k", linestyle="dashed")
    ax[i].text(
        am_c.val, 0.5, r"$am_c$", rotation=90, va="center", ha="right", color="k"
    )
    ax[i].fill_between(
        np.linspace(am_c.val - am_c.err, am_c.val + am_c.err, 100),
        ymin,
        eta_pdg.val + eta_pdg.err,
        color="k",
        alpha=0.2,
    )

    ax[i].hlines(
        y=eta_star.val, xmin=0, xmax=xmax if i != 2 else am_star.val, color="r"
    )
    ax[i].fill_between(
        xrange if i != 2 else np.linspace(0, am_star.val, 100),
        eta_star.val + eta_star.err,
        eta_star.val - eta_star.err,
        color="r",
        alpha=0.2,
    )
    ax[i].vlines(
        x=am_star.val, ymin=ymin, ymax=eta_star.val, color="r", linestyle="dashed"
    )
    ax[i].text(
        am_star.val, 0.5, r"$am^\star$", rotation=90, va="center", ha="right", color="r"
    )
    ax[i].fill_between(
        np.linspace(am_star.val - am_star.err, am_star.val + am_star.err, 100),
        ymin,
        eta_star.val + eta_star.err,
        color="r",
        alpha=0.2,
    )

    if i == 0:
        ax[i].set_ylabel(r"$M_{\eta_h}\,[\mathrm{GeV}]$")
        ax[i].text(
            0.05,
            eta_pdg.val - 0.05,
            r"$M_i=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$",
            va="top",
            ha="center",
            color="k",
        )
        ax[i].text(
            0.05,
            eta_star.val - 0.05,
            r"$\overline{M}=\frac{3}{5}M_{\eta_c}^\mathrm{PDG}$",
            va="top",
            ha="center",
            color="r",
        )
    if i == len(ens_list) - 1:
        ax[i].text(
            1.05,
            0.5,
            r"$\alpha + \beta\sqrt{am} + \gamma\,am$",
            va="center",
            ha="center",
            rotation=90,
            color="k",
            alpha=0.3,
            transform=ax[i].transAxes,
        )

    discard, new_ymax = ax[i].get_ylim()
    ax[i].set_ylim([ymin, new_ymax])


# ==== all ensembles Z_m ================================
fig, ax = plt.subplots(ncols=len(ens_list), sharey=True, figsize=(3 * len(ens_list), 4))
plt.subplots_adjust(wspace=0.08)
for i, ens in enumerate(ens_list):
    e = Z_bl_analysis(ens, renorm="mSMOM")
    stop = 3 if ens == "F1M" else 2
    x = join_stats([e.valence.amres[m] + eval(m) for m in e.all_masses[:-stop]])
    M_eta = join_stats(
        [e.valence.eta_h_masses[m] * e.ainv for m in e.all_masses[:-stop]]
    )

    def eta_mass_ansatz(am, param, **kwargs):
        return param[0] + param[1] * am**0.5 + param[2] * am

    res = fit_func(x, M_eta, eta_mass_ansatz, [0.1, 1, 1])

    def pred_amq(eta_mass, param, **kwargs):
        a, b, c = param
        root = (-b + (b**2 + 4 * c * (eta_mass - a)) ** 0.5) / (2 * c)
        return root**2

    am_star = stat(
        val=pred_amq(eta_star.val, res.val),
        err="fill",
        btsp=[pred_amq(eta_star.btsp[k], res.btsp[k]) for k in range(N_boot)],
    )

    ax[i].set_xlabel(r"$am_q + am_\mathrm{res}$")

    Z_m = join_stats([e.interpolate(mu, (m, m), "m") for m in e.all_masses[:-stop]])
    ax[i].errorbar(x.val, Z_m.val, xerr=x.err, yerr=Z_m.err, capsize=4, fmt="o")

    def Z_m_ansatz(am, param, **kwargs):
        return param[0] / am + param[1] + param[2] * am + param[3] * am**2

    res = fit_func(x, Z_m, Z_m_ansatz, [0.1, 1, 1, 1])
    Z_m_amstar = res.mapping(am_star)

    xmin, xmax = ax[i].get_xlim()
    xrange = np.linspace(x.val[0], xmax, 100)
    yrange = res.mapping(xrange)
    ax[i].fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
    )
    ax[i].set_xlim(0, xmax)
    ax[i].set_title(ens)
    ymin, ymax = ax[i].get_ylim()
    ax[i].errorbar(
        [am_star.val],
        [Z_m_amstar.val],
        xerr=[am_star.err],
        yerr=[Z_m_amstar.err],
        fmt="o",
        capsize=4,
        color="r",
    )
    ax[i].vlines(x=am_star.val, ymin=ymin, ymax=ymax, color="r", linestyle="dashed")
    ax[i].fill_between(
        np.linspace(am_star.val - am_star.err, am_star.val + am_star.err, 100),
        ymin,
        eta_star.val + eta_star.err,
        color="r",
        alpha=0.2,
    )

    ax[i].set_ylim([ymin, ymax])

    if i == 0:
        ax[i].set_ylabel(r"$Z_m^\mathrm{mSMOM}(\mu=" + str(mu) + r"\,\mathrm{GeV})$")
    if i == len(ens_list) - 1:
        ax[i].text(
            1.05,
            0.5,
            r"$\alpha/am + \beta + \gamma\,am + \delta\,(am)^2$",
            va="center",
            ha="center",
            rotation=90,
            color="k",
            alpha=0.3,
            transform=ax[i].transAxes,
        )

# ======= continuum limit example: quadratic========================
eta_pdg = eta_PDG
eta_star = eta_PDG * 0.75

ens_list = ["F1M", "M1M", "C1M"]
c = cont_extrap(ens_list)
ansatz, guess = c.ansatz()

fig, ax = plt.subplots(figsize=(3.5, 2.5))
x = c.a_sq
ax.set_xlabel(r"$a^2\,[\mathrm{GeV}^{-2}]$")
y_mc_mSMOM, y_mc_SMOM, y_mbar = c.load_renorm_masses(mu, eta_pdg, eta_star)

ax.errorbar(
    x.val,
    y_mc_mSMOM.val,
    xerr=x.err,
    yerr=y_mc_mSMOM.err,
    fmt="o",
    capsize=4,
    color="b",
    mfc="None",
    label="mSMOM",
)
res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess)
y_mc_mSMOM_phys = res_mc_mSMOM[0]

xmin, xmax = ax.get_xlim()
xrange = np.linspace(0, xmax)
yrange = res_mc_mSMOM.mapping(xrange)

ax.errorbar(
    [0.0],
    [y_mc_mSMOM_phys.val],
    yerr=[y_mc_mSMOM_phys.err],
    capsize=4,
    fmt="o",
    color="b",
)
ax.fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="b", alpha=0.2
)


ax.errorbar(
    x.val,
    y_mc_SMOM.val,
    xerr=x.err,
    yerr=y_mc_SMOM.err,
    fmt="x",
    capsize=4,
    color="k",
    mfc="None",
    label="SMOM",
)

res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
y_mc_SMOM_phys = res_mc_SMOM[0]

yrange = res_mc_SMOM.mapping(xrange)
ax.errorbar(
    [0.0],
    [y_mc_SMOM_phys.val],
    yerr=[y_mc_SMOM_phys.err],
    capsize=4,
    fmt="o",
    color="k",
)
ax.fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)
ymin, ymax = ax.get_ylim()
ax.vlines(x=0.0, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
ax.set_ylim([ymin, ymax])
xmin, discard = ax.get_xlim()
ax.set_xlim(xmin, xmax)

ax.legend()
ax.set_title(
    r"$M_i=M_{\eta_c}^\mathrm{PDG},\,"
    + r" \overline{M}=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$"
)
ax.set_ylabel(r"$m_i^R\,[\mathrm{GeV}]$")


# ======= continuum limit example: linear========================
ansatz, guess = c.ansatz(choose="linear")

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_xlabel(r"$a^2\,[\mathrm{GeV}^{-2}]$")
y_mc_mSMOM, y_mc_SMOM, y_mbar = c.load_renorm_masses(mu, eta_pdg, eta_star)

ax.errorbar(
    x.val,
    y_mc_mSMOM.val,
    xerr=x.err,
    yerr=y_mc_mSMOM.err,
    fmt="o",
    capsize=4,
    color="b",
    mfc="None",
    label="mSMOM",
)
res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess)
y_mc_mSMOM_phys = res_mc_mSMOM[0]

xmin, xmax = ax.get_xlim()
xrange = np.linspace(0, xmax)
yrange = res_mc_mSMOM.mapping(xrange)

ax.errorbar(
    [0.0],
    [y_mc_mSMOM_phys.val],
    yerr=[y_mc_mSMOM_phys.err],
    capsize=4,
    fmt="o",
    color="b",
)
ax.fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="b", alpha=0.2
)


ax.errorbar(
    x.val,
    y_mc_SMOM.val,
    xerr=x.err,
    yerr=y_mc_SMOM.err,
    fmt="x",
    capsize=4,
    color="k",
    mfc="None",
    label="SMOM",
)

res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
y_mc_SMOM_phys = res_mc_SMOM[0]

yrange = res_mc_SMOM.mapping(xrange)
ax.errorbar(
    [0.0],
    [y_mc_SMOM_phys.val],
    yerr=[y_mc_SMOM_phys.err],
    capsize=4,
    fmt="o",
    color="k",
)
ax.fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)
ymin, ymax = ax.get_ylim()
ax.vlines(x=0.0, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
ax.set_ylim([ymin, ymax])
xmin, discard = ax.get_xlim()
ax.set_xlim(xmin, xmax)

ax.legend()
ax.set_title(
    r"$M_i=M_{\eta_c}^\mathrm{PDG},\,"
    + r" \overline{M}=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$"
)
ax.set_ylabel(r"$m_i^R\,[\mathrm{GeV}]$")

# ====== variations with eta_c mass======================
fracs = [0.4, 0.5, 0.6, 0.75, 0.9, 1]
frac_labels = [
    r"\frac{2}{5}",
    r"\frac{1}{2}",
    r"\frac{3}{5}",
    r"\frac{3}{4}",
    r"\frac{9}{10}",
    r"",
]
M_pdg_list = [eta_PDG * f for f in fracs]
eta_star = eta_PDG * 0.75
fig, ax = plt.subplots(figsize=(3, 4))
ax.set_xlabel(r"$a^2\,[\mathrm{GeV}^{-2}]$")

y_phys = []
for eta_idx, eta_pdg in enumerate(M_pdg_list):
    label = r"$M_i=" + frac_labels[eta_idx] + r"M_{\eta_c}^\mathrm{PDG}$"
    color = color_list[eta_idx]

    y_mc_mSMOM, y_mc_SMOM, y_mbar = c.load_renorm_masses(
        mu, eta_pdg, eta_star, fit_alt=True
    )
    ax.errorbar(
        x.val,
        y_mc_mSMOM.val,
        xerr=x.err,
        yerr=y_mc_mSMOM.err,
        fmt="o",
        capsize=4,
        color=color,
        mfc="None",
        label=label,
    )

    if eta_idx == 0:
        xmin, xmax = ax.get_xlim()

    if eta_pdg.val > eta_PDG.val * 0.75:
        if len(c.ens_list) == 3:
            ansatz, guess = c.ansatz(choose="linear")
            end = -1
            xrange = np.linspace(0, x.val[-2])
        else:
            ansatz, guess = c.ansatz(choose="linear")
            end = -2
            xrange = np.linspace(0, x.val[-3])

    else:
        ansatz, guess = c.ansatz()
        end = None
        xrange = np.linspace(0, xmax)

    res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess, end=end)
    y_mc_mSMOM_phys = res_mc_mSMOM[0]
    y_phys.append(y_mc_mSMOM_phys)

    yrange = res_mc_mSMOM.mapping(xrange)

    ax.errorbar(
        [0.0],
        [y_mc_mSMOM_phys.val],
        yerr=[y_mc_mSMOM_phys.err],
        capsize=4,
        fmt="o",
        color=color,
    )
    ax.fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.2
    )

    ax.errorbar(
        x.val,
        y_mc_SMOM.val,
        xerr=x.err,
        yerr=y_mc_SMOM.err,
        fmt="x",
        capsize=4,
        color=color,
        mfc="None",
    )

    res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
    y_mc_SMOM_phys = res_mc_SMOM[0]

    yrange = res_mc_SMOM.mapping(xrange)
    ax.errorbar(
        [0.0],
        [y_mc_SMOM_phys.val],
        yerr=[y_mc_SMOM_phys.err],
        capsize=4,
        fmt="x",
        color=color,
    )
    ax.fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.1
    )

ymin, ymax = ax.get_ylim()
ax.vlines(x=0.0, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
ax.set_ylim([ymin, ymax])
xmin, discard = ax.get_xlim()
ax.set_xlim(xmin, xmax)

ax.legend(bbox_to_anchor=(1, 0.5))
ax.set_title(r"$\overline{M}=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$")
ax.set_ylabel(r"$m_i^R\,[\mathrm{GeV}]$")


fig, ax = plt.subplots(figsize=(4, 3))
x = join_stats(M_pdg_list)
y = join_stats(y_phys)
y = y / x
ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err, fmt="o", capsize=4, c="k")
ymin, ymax = ax.get_ylim()
ax.vlines(x=eta_PDG.val, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
ax.set_ylim([ymin, ymax])
ax.set_ylabel(r"$m_i^R\,/\,M_i$")
ax.set_xlabel(r"$M_i$ [GeV]")

ax_twin = ax.twiny()
ax_twin.set_xlim(ax.get_xlim())
ax_twin.set_xticks([eta_PDG.val])
ax_twin.set_xticklabels([r"$M_{\eta_c}^\mathrm{PDG}$"])


def M_ansatz(m, param, **kwargs):
    return param[0] / m + param[1] + param[2] * m


guess = [1, 1e-1, 1e-2]
res = fit_func(x, y, M_ansatz, guess)
xmin, xmax = ax.get_xlim()
xrange = np.linspace(xmin, xmax)
yrange = res.mapping(xrange)
ax.fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)
ax.set_xlim([xmin, xmax])
ax.text(
    0.5,
    0.1,
    r"$\chi^2/\mathrm{DOF}:" + str(np.around(res.chi_sq / res.DOF, 3)) + r"$",
    ha="center",
    va="center",
    transform=ax.transAxes,
)
ax.text(
    0.1, 0.9, r"$\mu=2\,\mathrm{GeV}$", ha="left", va="center", transform=ax.transAxes
)
ax.text(
    0.1,
    0.8,
    r"$\overline{M}=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$",
    ha="left",
    va="center",
    transform=ax.transAxes,
)
ax.text(
    1.05,
    0.5,
    r"$\alpha/M_i + \beta + \gamma\,M_i$",
    va="center",
    ha="center",
    rotation=90,
    color="k",
    alpha=0.3,
    transform=ax.transAxes,
)


# ====== variations with Mbar mass======================
fracs = [0.4, 0.5, 0.6, 0.75, 0.9, 1]
frac_labels = [
    r"\frac{2}{5}",
    r"\frac{1}{2}",
    r"\frac{3}{5}",
    r"\frac{3}{4}",
    r"\frac{9}{10}",
    r"",
]
M_star_list = [eta_PDG * f for f in fracs]
eta_pdg = eta_PDG * 0.75
x = c.a_sq
fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", figsize=(3, 5))
plt.subplots_adjust(hspace=0)
ax[1].set_xlabel(r"$a^2\,[\mathrm{GeV}^{-2}]$")
ansatz, guess = c.ansatz()

for eta_idx, eta_star in enumerate(M_star_list):
    label = r"$\overline{M}=" + frac_labels[eta_idx] + r"M_{\eta_c}^\mathrm{PDG}$"
    color = color_list[eta_idx]

    y_mc_mSMOM, y_mc_SMOM, y_mbar = c.load_renorm_masses(mu, eta_pdg, eta_star)
    ax[0].errorbar(
        x.val,
        y_mc_mSMOM.val,
        xerr=x.err,
        yerr=y_mc_mSMOM.err,
        fmt="o",
        capsize=4,
        color=color,
        mfc="None",
        label=label,
    )

    xmin, xmax = ax[0].get_xlim()
    if eta_star.val > eta_PDG.val * 0.75:
        if len(c.ens_list) == 3:
            ansatz, guess = c.ansatz(choose="linear")
            end = -1
            xrange = np.linspace(0, x.val[-2])
        else:
            ansatz, guess = c.ansatz(choose="linear")
            end = -2
            xrange = np.linspace(0, x.val[-3])

    else:
        ansatz, guess = c.ansatz()
        end = None
        xrange = np.linspace(0, xmax)

    res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess, end=end)
    y_mc_mSMOM_phys = res_mc_mSMOM[0]

    yrange = res_mc_mSMOM.mapping(xrange)

    ax[0].errorbar(
        [0.0],
        [y_mc_mSMOM_phys.val],
        yerr=[y_mc_mSMOM_phys.err],
        capsize=4,
        fmt="o",
        color=color,
    )
    ax[0].fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.2
    )

    ax[1].errorbar(
        x.val,
        y_mbar.val,
        xerr=x.err,
        yerr=y_mbar.err,
        fmt="o",
        capsize=4,
        color=color,
        mfc="None",
        label=label,
    )

    res_mbar = fit_func(x, y_mbar, ansatz, guess, end=end)
    y_mbar_phys = res_mbar[0]

    yrange = res_mbar.mapping(xrange)
    ax[1].errorbar(
        [0.0],
        [y_mbar_phys.val],
        yerr=[y_mbar_phys.err],
        capsize=4,
        fmt="o",
        color=color,
    )
    ax[1].fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.2
    )
    ax[1].set_xlim([-0.02, xmax])

ax[0].errorbar(
    x.val,
    y_mc_SMOM.val,
    xerr=x.err,
    yerr=y_mc_SMOM.err,
    fmt="x",
    capsize=4,
    color="k",
    mfc="None",
)

res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
y_mc_SMOM_phys = res_mc_SMOM[0]

yrange = res_mc_SMOM.mapping(xrange)
ax[0].errorbar(
    [0.0],
    [y_mc_SMOM_phys.val],
    yerr=[y_mc_SMOM_phys.err],
    capsize=4,
    fmt="x",
    color="k",
)
ax[0].fill_between(
    xrange, yrange.val + yrange.err, yrange.val - yrange.err, color="k", alpha=0.1
)

ax[0].set_title(r"$M_i=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$")
ax[0].set_ylabel(r"$m_i^R\,\mathrm{[GeV]}$")
ax[1].set_ylabel(r"$\overline{m}\,[\mathrm{GeV}]$")

ax[0].legend(bbox_to_anchor=(1, 0.4))

for idx in range(2):
    ymin, ymax = ax[idx].get_ylim()
    ax[idx].set_ylim([ymin, ymax])
    ax[idx].vlines(x=0.0, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
    ax[idx].set_ylim([ymin, ymax])


# ====== S and M ensembles===========================

ens_list = ["F1S", "F1M", "M1", "M1M", "C1", "C1M"]
eta_star = eta_PDG * 0.75
c = cont_extrap(ens_list)
x = c.a_sq

fig, ax = plt.subplots(figsize=(3.5, 5))
ax.set_xlabel(r"$a^2\,[\mathrm{GeV}^{-2}]$")

y_phys = []
for eta_idx, eta_pdg in enumerate(M_pdg_list):
    label = r"$M_i=" + frac_labels[eta_idx] + r"M_{\eta_c}^\mathrm{PDG}$"
    color = color_list[eta_idx]

    y_mc_mSMOM, y_mc_SMOM, y_mbar = c.load_renorm_masses(
        mu, eta_pdg, eta_star, fit_alt=True
    )
    ax.errorbar(
        x.val,
        y_mc_mSMOM.val,
        xerr=x.err,
        yerr=y_mc_mSMOM.err,
        fmt="o",
        capsize=4,
        color=color,
        mfc="None",
        label=label,
    )

    if eta_idx == 0:
        xmin, xmax = ax.get_xlim()

    if eta_pdg.val > eta_PDG.val * 0.75:
        if len(c.ens_list) == 3:
            ansatz, guess = c.ansatz(choose="linear")
            end = -1
            xrange = np.linspace(0, x.val[-2])
        else:
            ansatz, guess = c.ansatz(choose="linear")
            end = -2
            xrange = np.linspace(0, x.val[-3])

    else:
        ansatz, guess = c.ansatz()
        end = None
        xrange = np.linspace(0, xmax)

    res_mc_mSMOM = fit_func(x, y_mc_mSMOM, ansatz, guess, end=end)
    y_mc_mSMOM_phys = res_mc_mSMOM[0]
    y_phys.append(y_mc_mSMOM_phys)

    yrange = res_mc_mSMOM.mapping(xrange)

    ax.errorbar(
        [0.0],
        [y_mc_mSMOM_phys.val],
        yerr=[y_mc_mSMOM_phys.err],
        capsize=4,
        fmt="o",
        color=color,
    )
    ax.fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.2
    )

    ax.errorbar(
        x.val,
        y_mc_SMOM.val,
        xerr=x.err,
        yerr=y_mc_SMOM.err,
        fmt="x",
        capsize=4,
        color=color,
        mfc="None",
    )

    res_mc_SMOM = fit_func(x, y_mc_SMOM, ansatz, guess)
    y_mc_SMOM_phys = res_mc_SMOM[0]

    yrange = res_mc_SMOM.mapping(xrange)
    ax.errorbar(
        [0.0],
        [y_mc_SMOM_phys.val],
        yerr=[y_mc_SMOM_phys.err],
        capsize=4,
        fmt="x",
        color=color,
    )
    ax.fill_between(
        xrange, yrange.val + yrange.err, yrange.val - yrange.err, color=color, alpha=0.1
    )

ymin, ymax = ax.get_ylim()
ax.vlines(x=0.0, ymin=ymin, ymax=ymax, color="k", linestyle="dashed")
ax.set_ylim([ymin, ymax])
xmin, discard = ax.get_xlim()
ax.set_xlim(xmin, xmax)

ax.legend(bbox_to_anchor=(1, 0.5))
ax.set_title(r"$\overline{M}=\frac{3}{4}M_{\eta_c}^\mathrm{PDG}$")
ax.set_ylabel(r"$m_i^R\,[\mathrm{GeV}]$")


filename = "/Users/rajnandinimukherjee/Desktop/Felix_plots.pdf"
call_PDF(filename, open=True)
