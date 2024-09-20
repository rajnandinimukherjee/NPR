from NPR_structures import *
import pickle

C_tab, M_tab = pickle.load(open("pickles/1411_res.p", "rb"))
comp_data = {"C1": C_tab["C1"], "C2": C_tab["C2"], "M1": M_tab["M1"], "M2": M_tab["M2"]}

C_pt2, M_pt2 = pickle.load(open("pickles/3GeV.p", "rb"))
bl_V = pickle.load(open("pickles/bl_V.p", "rb"))

ensembles = ["C1", "C2", "M1", "M2"]
colors = ["blue", "orange", "red", "green"]
all_data = {}
fig1, ax1 = plt.subplots(nrows=5, ncols=5)
fig2, ax2 = plt.subplots(nrows=1, ncols=1)

for i in range(len(ensembles)):
    ens = ensembles[i]
    color = colors[i]
    Z, proj = pickle.load(open(f"pickles/{ens}_res.p", "rb"))
    all_data[ens] = {"Z": Z[(0, 0)], "proj": proj[(0, 0)]}
    momenta = list(Z[(0, 0)].keys())
    momenta.sort()
    momenta = np.array(momenta)
    comp_mom = C_tab["mom"] if ens == "C1" or ens == "C2" else M_tab["mom"]
    pt2 = C_pt2 if ens == "C1" or ens == "C2" else M_pt2

    for i in range(5):
        for j in range(5):
            k, l = i, j
            if mask[k, l]:
                ax1[i, j].scatter(
                    momenta,
                    [all_data[ens]["Z"][m][k, l] for m in momenta],
                    label=ens,
                    c=color,
                    alpha=0.75,
                )
                ax1[i, j].scatter(
                    [3],
                    [pt2[k, l]],
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.75,
                    marker="*",
                )
                ax1[i, j].set_ylabel(
                    "$Z_{" + str(k + 1) + "," + str(l + 1) + "}/Z_V^2$"
                )
                ax1[i, j].set_xlabel("$q/GeV$")
            else:
                ax1[i, j].axis("off")

    ax2.scatter(
        momenta,
        [(all_data[ens]["proj"][m][0, 0] / fq_gamma_F[0, 0]).real for m in momenta],
        label="RM",
        c=color,
        alpha=0.75,
    )
    ax2.scatter(comp_mom, comp_data[ens], c=color, marker="*", label="1411.7017")
handles, labels = ax1[0, 0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="center right")
fig1.suptitle("$Z_{ij}/Z_V^2 =  F(\Lambda^{-1})^T$ for fourquark operators")

ax2.set_xlabel("$q/GeV$")
fig2.legend(handles, labels, loc="center right")
fig2.suptitle(
    "$\Lambda_{VV+AA}}$ in ($\gamma_{\mu}$,$\gamma_{\mu}$) scheme\no:RM,*:1411.7017"
)
plt.show()
