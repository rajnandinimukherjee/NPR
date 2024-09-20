from cont_chir_extrap import *

MS_bar_results = pickle.load(open("MS_bar_results.p", "rb"))

N_flav = 3

m_K_pm = stat(val=493.677, err=0.013, btsp="fill") / 1000
m_K_0 = stat(val=498.611, err=0.013, btsp="fill") / 1000
m_K = (m_K_0 + m_K_pm) / 2
f_K = stat(val=155.7, err=0.3 if N_flav == 4 else 0.7, btsp="fill") / 1000
m_s_MS_2GeV = (
    stat(
        val=93.44 if N_flav == 4 else 92.03,
        err=0.68 if N_flav == 4 else 0.88,
        btsp="fill",
    )
    / 1000
)

# isospin asymmetric down quark mass
# m_d_MS_2GeV = stat(
#        val=4.70 if N_flav==4 else 4.67,
#        err=0.05 if N_flav==4 else 0.09,
#        btsp='fill')/1000

# isospin symmetric down quark mass
m_d_MS_2GeV = (
    stat(
        val=3.410 if N_flav == 4 else 3.364,
        err=0.043 if N_flav == 4 else 0.041,
        btsp="fill",
    )
    / 1000
)

conv_factor_2GeV_to_3GeV = 0.910526 if N_flav == 4 else 0.90511
m_s_MS_3GeV = m_s_MS_2GeV * conv_factor_2GeV_to_3GeV
m_d_MS_3GeV = m_d_MS_2GeV * conv_factor_2GeV_to_3GeV

# See eq 144 (pg 119) in [arXiv:1902.08191]
NLO_3_to_inf = expm(
    -gamma_0 * np.log(alpha_s(3.0, f=N_flav)) / (2 * Bcoeffs(N_flav)[0])
)
NLO_term = Bcoeffs(N_flav)[1] * gamma_0 - Bcoeffs(N_flav)[0] * gamma_1_MS(N_flav)
NLO_term = NLO_term / (2 * Bcoeffs(N_flav)[0] ** 2)
NLO_3_to_inf = NLO_3_to_inf @ (
    np.identity(5) + (NLO_term) * (alpha_s(3.0, f=N_flav) / (4 * np.pi))
)

B_K_RGI = stat(
    val=0.717 if N_flav == 4 else 0.7625,
    err=0.024 if N_flav == 4 else 0.0097,
    btsp="fill",
)
B_K_3GeV = B_K_RGI / NLO_3_to_inf[0, 0]


UKQCD16 = {
    "ratios": {
        "val": [-19.48, 6.08, 43.11, 10.99],
        "err": [0.44, 0.15, 0.89, 0.20],
        "sys": [0.52, 0.23, 2.30, 0.88],
    },
    "bags": {
        "val": [0.488, 0.743, 0.920, 0.707],
        "err": [0.007, 0.014, 0.012, 0.008],
        "sys": [0.017, 0.065, 0.016, 0.044],
    },
    "kwargs": {"fmt": "x", "label": "RBC-UKQCD16", "color": "tab:cyan"},
}

SWME15 = {
    "bags": {
        "val": [0.525, 0.772, 0.981, 0.751],
        "err": [0.001, 0.005, 0.003, 0.007],
        "sys": [0.023, 0.035, 0.062, 0.068],
    },
    "kwargs": {"fmt": "d", "label": "SWME15", "color": "tab:blue"},
}

FLAG21 = {
    "bags": {"val": [0.502, 0.766, 0.926, 0.720], "err": [0.014, 0.032, 0.019, 0.038]}
}

ETM15 = {
    "bags": {
        "val": [0.46, 0.79, 0.78, 0.49],
        "err": [0.01, 0.02, 0.02, 0.03],
        "sys": [0.03, 0.05, 0.04, 0.03],
    },
    "kwargs": {"fmt": "^", "label": "ETM15", "color": "tab:orange"},
}

ETM12 = {
    "bags": {
        "val": [0.47, 0.78, 0.76, 0.58],
        "err": [0.02, 0.04, 0.02, 0.02],
        "sys": [0.01, 0.02, 0.02, 0.02],
    },
    "kwargs": {"fmt": "v", "label": "ETM12", "color": "tab:red"},
}

for idx in range(4):
    for fitter in [UKQCD16, SWME15, ETM15, ETM12]:
        fitter[f"B{idx+2}"] = stat(
            val=fitter["bags"]["val"][idx],
            err=(fitter["bags"]["err"][idx] ** 2 + fitter["bags"]["sys"][idx] ** 2)
            ** 0.5,
        )
        fitter[f"B{idx+2}"].disp = err_disp(
            fitter["bags"]["val"][idx],
            fitter["bags"]["err"][idx],
            sys_err=fitter["bags"]["sys"][idx],
        )

    FLAG21[f"B{idx+2}"] = stat(
        val=FLAG21["bags"]["val"][idx], err=FLAG21["bags"]["err"][idx]
    )
    FLAG21[f"B{idx+2}"].disp = err_disp(
        FLAG21["bags"]["val"][idx], FLAG21["bags"]["err"][idx]
    )
    UKQCD16[f"R{idx+2}"] = stat(
        val=UKQCD16["ratios"]["val"][idx],
        err=(UKQCD16["ratios"]["err"][idx] ** 2 + UKQCD16["ratios"]["sys"][idx] ** 2)
        ** 0.5,
    )
    UKQCD16[f"R{idx+2}"].disp = err_disp(
        UKQCD16["ratios"]["val"][idx],
        UKQCD16["ratios"]["err"][idx],
        sys_err=UKQCD16["ratios"]["sys"][idx],
    )

    SWME15[f"R{idx+2}"] = stat(val=0, err=0)
    SWME15[f"R{idx+2}"].disp = "-"


UKQCD24 = {
    key: val["store"]
    for key, val in MS_bar_results["combined"].items()
    if key != "name"
}
UKQCD24.update({"kwargs": {"label": "RBC-UKQCD24", "fmt": "o", "color": "k"}})

NB_R = []
N1 = norm_factors(rotate=NPR_to_SUSY)[0]
const = (B_K_3GeV * N1) * ((m_s_MS_3GeV + m_d_MS_3GeV) ** 2) / (m_K**2)
B1 = UKQCD24["B1"]
M1 = B1 * (m_K**2) * (f_K**2) * N1
for idx in range(1, 5):
    Bi = UKQCD24[f"B{idx+1}"]
    Ni = norm_factors(rotate=NPR_to_SUSY)[idx]
    Mi = Bi * (m_K**4) * (f_K**2) * Ni / ((m_s_MS_3GeV + m_d_MS_3GeV) ** 2)
    Ri_other = Mi / M1

    Ri = UKQCD24[f"R{idx+1}"]
    Mi = Ri * M1
    print(
        f"R{idx+1}:{err_disp(Ri.val, Ri.err)},\t"
        + f"R{idx+1} alt: {err_disp(Ri_other.val, Ri_other.err)},\t"
        + f"M{idx+1}: {err_disp(Mi.val, Mi.err)}"
    )

for idx in range(1, 5):
    Ri = UKQCD24[f"R{idx+1}"]
    Ni = norm_factors(rotate=NPR_to_SUSY)[idx]
    Mi = Ri * M1
    Bi_other = Mi / ((m_K**4) * (f_K**2) * Ni / ((m_s_MS_3GeV + m_d_MS_3GeV) ** 2))

    Bi = UKQCD24[f"B{idx+1}"]
    Mi = Bi * (m_K**4) * (f_K**2) * Ni / ((m_s_MS_3GeV + m_d_MS_3GeV) ** 2)
    print(
        f"B{idx+1}:{err_disp(Bi.val, Bi.err)},\t"
        + f"B{idx+1} alt: {err_disp(Bi_other.val, Bi_other.err)},\t"
        + f"M{idx+1}: {err_disp(Mi.val, Mi.err)}"
    )

for idx in range(1, 5):
    Ri = UKQCD24[f"R{idx+1}"]

    Bi = UKQCD24[f"B{idx+1}"]
    Ni = norm_factors(rotate=NPR_to_SUSY)[idx]

    ki = (Bi / Ri) * Ni
    NB_R.append(ki)
    print(f"{idx+1}:{err_disp(ki.val, ki.err)}," + f" {err_disp(const.val, const.err)}")

fig, ax = plt.subplots(figsize=(4, 2.5))
mass_sum_pred = (NB_R[0] * (m_K**2) / (B1 * N1)) ** 0.5
print(f"(m_s+m_d) in MS-bar using O2: {err_disp(mass_sum_pred.val, mass_sum_pred.err)}")

ax.axhline(const.val, color="0.7")
ax.axhspan(
    const.val + const.err,
    const.val - const.err,
    color="k",
    alpha=0.1,
    label=r"FLAG $N_f=2+1+1$" if N_flav == 4 else r"FLAG $N_f=2+1$",
)
ax.errorbar(
    np.arange(2, 6),
    [k.val for k in NB_R],
    yerr=[k.err for k in NB_R],
    fmt="o",
    capsize=4,
    c="k",
)
ax.legend(frameon=False)
ax.set_xticks([2, 3, 4, 5])
ax.set_ylabel(
    r"$N_i\mathcal{B}_i^{\overline{\mathrm{MS}}}/R_i^{\overline{\mathrm{MS}}}$", size=16
)
ax.set_xlabel(r"$i$", size=16)

filename = "/Users/rajnandinimukherjee/Desktop/draft_plots/summary_plots/NiBioverRi_comparison.pdf"
call_PDF(filename, open=False)

# ===============================================================================================
# plot BSM bags
fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8, 4))
plt.subplots_adjust(wspace=0, hspace=0)

fit_dicts = [ETM12, ETM15, SWME15, UKQCD16, UKQCD24]
num_fits = len(fit_dicts)
for idx in range(4):
    ax[idx].set_title(r"$\mathcal{B}_{" + str(idx + 2) + r"}$", fontsize=14)
    j = 2
    ax[idx].axvspan(
        FLAG21[f"B{idx+2}"].val - FLAG21[f"B{idx+2}"].err,
        FLAG21[f"B{idx+2}"].val + FLAG21[f"B{idx+2}"].err,
        color="k",
        alpha=0.1,
        label=r"FLAG21 $N_f=2+1$",
    )
    for fitter in fit_dicts:
        kwargs = fitter["kwargs"]
        ax[idx].errorbar(
            [fitter[f"B{idx+2}"].val],
            [j],
            xerr=[fitter[f"B{idx+2}"].err],
            capsize=4,
            **kwargs,
        )
        j += 1
    ax[idx].set_ylim([0, num_fits + 2])
    ax[idx].set_yticks([])

handles, labels = ax[-1].get_legend_handles_labels()
fig.legend(
    reversed(handles),
    reversed(labels),
    loc="center right",
    bbox_to_anchor=(1.1, 0.49),
    labelspacing=2.25,
    frameon=False,
)

filename = (
    "/Users/rajnandinimukherjee/Desktop/draft_plots/summary_plots/comparison_plot.pdf"
)
call_PDF(filename, open=True)

# ==================================================================================================
# plot B_K
fig, ax = plt.subplots(figsize=(3, 4))
marker_list = ["o", "d", "^", "s", "v", ">", "<", "p"]
B_K_dict = {
    "RBC-UKQCD24": MS_bar_results["combined"]["B1"]["store"] * NLO_3_to_inf[0, 0],
    "SWME15": stat(val=0.735, err=(0.005**2 + 0.036**2) ** 0.5),
    "ETM15": stat(val=0.717, err=(0.018**2 + 0.016**2) ** 0.5),
    "RBC-UKQCD14": stat(val=0.7499, err=(0.0024**2 + 0.0150**2) ** 0.5),
    "ETM12": stat(val=0.727, err=(0.022**2 + 0.012**2) ** 0.5),
    "Laiho11": stat(val=0.7628, err=(0.0038**2 + 0.0205**2) ** 0.5),
    "BMW11": stat(val=0.7727, err=(0.0081**2 + 0.0084**2) ** 0.5),
    "FLAG21": stat(val=0.7625, err=0.0097),
}
num_B_K = len(B_K_dict.keys())
for key, B_hat in B_K_dict.items():
    if key != "FLAG21":
        kwargs = {"color": "k"} if key == "RBC-UKQCD24" else {}
        idx = list(B_K_dict.keys()).index(key)
        ax.errorbar(
            B_hat.val,
            num_B_K - idx,
            xerr=B_hat.err,
            fmt=marker_list[idx],
            capsize=4,
            label=key,
            **kwargs,
        )
    else:
        ax.axvspan(
            B_hat.val + B_hat.err,
            B_hat.val - B_hat.err,
            color="k",
            alpha=0.1,
            label=r"FLAG21 $N_f=2+1$",
        )
ax.set_yticks([])
ax.set_ylim([0, num_B_K + 1])
ax.set_title(r"$\hat{B}_K$", fontsize=14)

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles[1:] + [handles[0]],
    labels[1:] + [labels[0]],
    loc="center right",
    bbox_to_anchor=(1.45, 0.49),
    labelspacing=1.55,
    frameon=False,
)

filename = "/Users/rajnandinimukherjee/Desktop/draft_plots/summary_plots/B_K_plot.pdf"
call_PDF(filename, open=True)
