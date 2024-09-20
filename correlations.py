from cont_chir_extrap import *

fit_systematics = pickle.load(open(f"fit_systematics_20_{fit_file}_{scheme}.p", "rb"))
scaling_systematics = pickle.load(
    open(f"scaling_systematics_{scheme}_{fit_file}.p", "rb")
)
MS_bar_results = pickle.load(open("MS_bar_results.p", "rb"))

quantities = {f"R{i+2}": r"$R_" + str(i + 2) + r"$" for i in range(4)}
quantities.update({f"B{i+1}": r"$\mathcal{B}_" + str(i + 1) + r"$" for i in range(5)})
stat_only_values = {
    f"RISMOM_2GeV_{scheme}": {
        key: fit_systematics["central"][key][0] for key in quantities
    },
    f"RISMOM_3GeV_{scheme}": {
        key: scaling_systematics["(2,3)"][key] for key in quantities
    },
    f"MSbar_3GeV_{scheme}": {
        key: MS_bar_results[scheme][key]["central"] for key in quantities
    },
}

N_flav = 3
m_K_pm = stat(val=493.677, err=0.013, btsp="fill") / 1000
m_K_0 = stat(val=498.611, err=0.013, btsp="fill") / 1000
m_K = (m_K_0 + m_K_pm) / 2
f_K = stat(val=155.7, err=0.3 if N_flav == 4 else 0.7, btsp="fill") / 1000
for key in stat_only_values.keys():
    B1 = stat_only_values[key]["B1"]
    N1 = norm_factors()[0]
    M1 = B1 * (m_K.val**2) * (f_K.val**2) * N1
    stat_only_values[key]["M1"] = M1
    for idx in range(1, 5):
        Ri = stat_only_values[key][f"R{idx+1}"]
        stat_only_values[key][f"M{idx+1}"] = Ri * M1

from mpl_toolkits.axes_grid1 import make_axes_locatable

filename = (
    f"/Users/rajnandinimukherjee/Desktop/draft_plots/correlation_data/correlations.h5"
)
labels = [r"$R_" + str(i + 1) + r"$" for i in range(1, 5)]
labels += [r"$\mathcal{B}_" + str(i + 1) + r"$" for i in range(5)]
labels += [r"$\langle O_" + str(i + 1) + r"\rangle$" for i in range(5)]
with h5py.File(filename, "a") as f:
    names = list(stat_only_values[list(stat_only_values.keys())[0]].keys())
    if "observables" not in f:
        name_list = f.create_dataset(name="observables", data=names)
    for key in stat_only_values.keys():
        all_stats = join_stats([val for key, val in stat_only_values[key].items()])
        cov = COV(all_stats.btsp, center=all_stats.val)
        errs_inv = np.diag(1 / all_stats.err)
        corr = errs_inv @ cov @ errs_inv
        if key in f.keys():
            del f[key]

        corr_data = f.create_dataset(name=key, data=corr)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(corr, cmap="RdBu", interpolation="nearest")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.xaxis.tick_top()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        plot_filename = f"/Users/rajnandinimukherjee/Desktop/draft_plots/correlation_data/plot_{key}.pdf"
        call_PDF(plot_filename, open=True)

        print(f"Saved {key} data to {filename}.")
