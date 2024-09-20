import itertools
from scipy.interpolate import interp1d
from NPR_classes import *

# fit_file = 'Felix'
fit_file = "Tobi"

all_bag_data = h5py.File(f"kaon_bag_fits_{fit_file}.h5", "r")
bag_ensembles = [key for key in all_bag_data.keys() if key in UKQCD_ens]


def load_info(key, ens, ops=operators, meson="ls", **kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson == "ls":
        central = np.array([np.array(h5_data[op][key]["central"]).item() for op in ops])
        error = np.array([np.array(h5_data[op][key]["error"]).item() for op in ops])
        bootstraps = np.array(
            [np.array(h5_data[op][key]["Bootstraps"])[:, 0] for op in ops]
        ).T

    elif meson == "ll":
        central = np.array(h5_data[key]["central"]).item()
        error = np.array(h5_data[key]["error"]).item()
        bootstraps = np.array(h5_data[key]["Bootstraps"])[:, 0]

    if key == "gr-O-gr":
        central[1] *= -1
        bootstraps[:, 1] = -bootstraps[:, 1]
        central[3] *= -1
        bootstraps[:, 3] = -bootstraps[:, 3]
        central[4] *= -1
        bootstraps[:, 4] = -bootstraps[:, 4]

    return stat(val=central, err=error, btsp=bootstraps)


class Z_analysis:
    N_boot = N_boot
    N_ops = len(operators)

    def __init__(
        self,
        ensemble,
        action=(0, 0),
        norm="V",
        mask=fq_mask.copy(),
        scheme="gamma",
        sea_mass_idx=0,
        run_extrap=False,
        **kwargs,
    ):
        self.ens = ensemble
        self.action = action
        self.scheme = scheme
        self.ainv = stat(
            val=params[self.ens]["ainv"],
            err=params[self.ens]["ainv_err"],
            btsp="seed",
            seed=ensemble_seeds[self.ens],
        )
        self.a_sq = self.ainv ** (-2)
        self.mask = mask

        self.f_pi = f_pi_PDG / self.ainv

        try:
            self.m_pi = load_info("m_0", self.ens, operators, meson="ll")
            if self.m_pi.N_boot != N_boot:
                self.m_pi = stat(val=self.m_pi.val, err=self.m_pi.err, btsp="fill")
            self.m_f_sq = stat(
                val=(self.m_pi.val / self.f_pi.val) ** 2,
                err="fill",
                btsp=(self.m_pi.btsp**2) / (self.f_pi.btsp**2),
            )
        except KeyError:
            if self.ens == "C1M":
                self.m_pi = stat(val=0.16079, err=0.00054, btsp="fill")
            elif self.ens == "M1M":
                self.m_pi = stat(
                    val=0.12005,
                    err=0.00065,
                    btsp="fill",
                )
            else:
                print(
                    f"{self.ens} m_pi data not identified,"
                    + " check for consistent ensemble naming"
                )

        self.norm = norm
        if norm == "11":
            self.mask[0, 0] = False

        if sea_mass_idx == "extrap":
            filename = f"Z_ij_{self.norm}_amq_0_{self.scheme}.h5"
            if self.ens[-1] != "0":
                if run_extrap:
                    self.valence_extrapolation(**kwargs)
                else:
                    self.load_extrap(filename, **kwargs)
            else:
                self.sea_m = "{:.4f}".format(params[self.ens]["masses"][0])
                self.masses = (self.sea_m, self.sea_m)
                self.load_fq_Z(norm=self.norm, **kwargs)

                am_q = eval(self.sea_m)
                alt_ens = self.ens[0] + "1M"
                alt_ens = Z_analysis(
                    alt_ens,
                    action=self.action,
                    scheme=self.scheme,
                    norm=self.norm,
                    mask=self.mask,
                    sea_mass_idx="extrap",
                    run_extrap=run_extrap,
                    **kwargs,
                )
                self.slopes = alt_ens.slopes
                self.am = alt_ens.am
                self.N_mom = len(self.am.val)
                extrap_Z = [
                    self.Z[mom] - self.slopes[mom] * am_q for mom in range(self.N_mom)
                ]
                self.Z = join_stats(extrap_Z)
                # print(f'Using {alt_ens.ens} valence extrapolation slope '+\
                # f'to extrpolate {self.ens} data at valence mass {am_q}.')

            self.sea_m = "{:.4f}".format(0.0)
            self.masses = (self.sea_m, self.sea_m)
            if "save_extrap" in kwargs:
                self.save_extrap(filename, **kwargs)

        else:
            self.sea_m = "{:.4f}".format(params[self.ens]["masses"][sea_mass_idx])
            self.masses = (self.sea_m, self.sea_m)
            self.load_fq_Z(norm=self.norm, **kwargs)

    def save_extrap(self, filename, **kwargs):
        file = h5py.File(filename, "a")
        if self.ens in file:
            del file[self.ens]

        ens_grp = file.create_group(self.ens)

        ens_grp.create_dataset("ap", data=self.am.val)
        ens_grp.create_dataset("Z/central", data=self.Z.val)
        ens_grp.create_dataset("Z/errors", data=self.Z.err)
        ens_grp.create_dataset("Z/bootstrap", data=self.Z.btsp)
        ens_grp.create_dataset("slope/central", data=self.slopes.val)
        ens_grp.create_dataset("slope/errors", data=self.slopes.err)
        ens_grp.create_dataset("slope/bootstrap", data=self.slopes.btsp)

        file.close()
        # print(f'Saved valence extrapolated Z factors for '+\
        #        f'{self.ens} to {filename}.')

    def load_extrap(self, filename, **kwargs):
        # print(f'Loading valence extrap data for {self.ens} from {filename}.')
        file = h5py.File(filename, "r")[self.ens]
        self.am = stat(val=file["ap"][:], err=np.zeros(len(file["ap"][:])), btsp="fill")
        self.N_mom = len(self.am.val)
        self.Z = stat(
            val=np.array(file["Z/central"][:]),
            err=np.array(file["Z/errors"][:]),
            btsp=np.array(file["Z/bootstrap"][:]),
        )
        self.slopes = stat(
            val=np.array(file["slope/central"][:]),
            err=np.array(file["slope/errors"][:]),
            btsp=np.array(file["slope/bootstrap"][:]),
        )

    def valence_extrapolation(self, plot_valence_extrap=False, filename="", **kwargs):

        am_l = "{:.4f}".format(params[self.ens]["aml_sea"])
        am_l_twice = "{:.4f}".format(params[self.ens]["aml_sea"] * 2)
        am_s_half = "{:.4f}".format(params[self.ens]["ams_sea"] / 2)

        order_am = [am_l, am_l_twice, am_s_half]
        names_am = [r"$am_l$", r"$2am_l$", r"$am_s/2$"]
        # print(f'Using valence masses {str(order_am)}'+\
        #        f' to extrapolate {self.ens} Z-factors')

        val_am = list(sorted(set(order_am)))
        try:
            Zs = join_stats(
                [
                    self.load_fq_Z(
                        masses=(m, m), norm=self.norm, pass_val=True, **kwargs
                    )
                    for m in val_am
                ]
            )
        except ValueError:
            Z_list = [
                self.load_fq_Z(masses=(m, m), norm=self.norm, pass_val=True, **kwargs)
                for m in val_am
            ]
            if self.ens == "F1M":
                Z_0 = self.load_fq_Z(
                    masses=(val_am[0], val_am[0]),
                    norm=self.norm,
                    pass_val=True,
                    **kwargs,
                )
                N_mom = len(self.am.val)
                Z_list[-1] = stat(
                    val=Z_list[-1].val[:N_mom],
                    err=Z_list[-1].err[:N_mom],
                    btsp=Z_list[-1].btsp[:, :N_mom],
                )
                Zs = join_stats(Z_list)

        if order_am == val_am:
            ordered_names = names_am
        else:
            ordered_names = [names_am[order_am.index(m)] for m in val_am]

        if plot_valence_extrap:
            fig1, ax1 = plt.subplots(
                nrows=self.N_ops, ncols=self.N_ops, figsize=(16, 16)
            )
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.suptitle(f"Extrapolation in valence quark mass for {self.ens}", y=0.9)

        def valence_extrap_ansatz(ams, param, **kwargs):
            return param[0] + param[1] * ams

        x = stat(
            val=[eval(mass) for mass in val_am], err=np.zeros(len(val_am)), btsp="fill"
        )
        extrap_Z = []
        slopes = []
        for mom_idx, mom in enumerate(self.am.val):
            Z_vals = np.zeros(shape=(self.N_ops, self.N_ops))
            Z_btsp = np.zeros(shape=(N_boot, self.N_ops, self.N_ops))

            Z_slope_vals = np.zeros(shape=(self.N_ops, self.N_ops))
            Z_slope_btsp = np.zeros(shape=(N_boot, self.N_ops, self.N_ops))

            if plot_valence_extrap:
                fig, ax = plt.subplots(
                    nrows=self.N_ops, ncols=self.N_ops, figsize=(16, 16)
                )
                plt.subplots_adjust(hspace=0, wspace=0)
            for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                if self.mask[i, j]:
                    y = stat(
                        val=Zs.val[:, mom_idx, i, j],
                        err=Zs.err[:, mom_idx, i, j],
                        btsp=Zs.btsp[:, :, mom_idx, i, j],
                    )
                    res = fit_func(x, y, valence_extrap_ansatz, [1, 1e-1])

                    Z_vals[i, j] = res[0].val
                    Z_btsp[:, i, j] = res[0].btsp

                    Z_slope_vals[i, j] = res[1].val
                    Z_slope_btsp[:, i, j] = res[1].btsp

                    if plot_valence_extrap:
                        ax[i, j].errorbar(x.val, y.val, yerr=y.err, fmt="o", capsize=4)
                        ax[i, j].axvline(0.0, c="k", linestyle="dashed", alpha=0.2)
                        ax[i, j].errorbar(
                            [0.0],
                            Z_vals[i, j],
                            yerr=st_dev(Z_btsp[:, i, j], mean=Z_vals[i, j]),
                            fmt="o",
                            capsize=4,
                            c="r",
                        )
                        xmin, xmax = ax[i, j].get_xlim()
                        xgrain = np.linspace(xmin, xmax, 50)
                        ygrain = res.mapping(xgrain)
                        ax[i, j].fill_between(
                            xgrain,
                            ygrain.val + ygrain.err,
                            ygrain.val - ygrain.err,
                            color="r",
                            alpha=0.2,
                        )
                        if j == 2 or j == 4:
                            ax[i, j].yaxis.tick_right()
                        if i == 1 or i == 3:
                            ax[i, j].set_xticks([])
                        else:
                            ax[i, j].set_xlabel(r"$am_q^{\mathrm{val}}$")

                        if i == 0 or i == 1 or i == 3:
                            ax_twin = ax[i, j].twiny()
                            ax_twin.set_xlim(ax[i, j].get_xlim())
                            ax_twin.set_xticks([0] + [xi for xi in x.val])
                            ax_twin.set_xticklabels(["0"] + ordered_names)
                else:
                    if plot_valence_extrap:
                        ax[i, j].axis("off")

            extrap_Z.append(stat(val=Z_vals, err="fill", btsp=Z_btsp))
            slopes.append(stat(val=Z_slope_vals, err="fill", btsp=Z_slope_btsp))

            if plot_valence_extrap:
                plt.suptitle(
                    f"Extrapolation in valence quark mass for ap = {mom}", y=0.9
                )

        self.Z = join_stats(extrap_Z)
        self.slopes = join_stats(slopes)

        if plot_valence_extrap:
            x = self.ainv * self.am
            for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                if self.mask[i, j]:
                    for mass_idx, mass in enumerate(val_am):
                        y = stat(
                            val=Zs.val[mass_idx, :, i, j],
                            err=Zs.err[mass_idx, :, i, j],
                        )
                        ax1[i, j].errorbar(
                            x.val,
                            y.val,
                            yerr=y.err,
                            capsize=4,
                            fmt="o",
                            label=names_am[mass_idx] + " : " + mass,
                        )

                    ax1[i, j].errorbar(
                        x.val,
                        self.Z.val[:, i, j],
                        yerr=self.Z.err[:, i, j],
                        fmt="o",
                        capsize=4,
                        color="k",
                        label=r"$\lim\,am_q^{\mathrm{val}}\to 0$",
                    )
                    ax1[i, j].legend()
                    if j == 2 or j == 4:
                        ax1[i, j].yaxis.tick_right()
                    if i == 1 or i == 3:
                        ax1[i, j].set_xticks([])
                    else:
                        ax1[i, j].set_xlabel(r"$\mu$ [GeV]")
                else:
                    ax1[i, j].axis("off")

            if filename == "":
                filename = f"valence_extrap_{self.ens}.pdf"

            call_PDF(filename, open=True)

    def interpolate(
        self,
        m,
        xaxis="mu",
        ainv=None,
        plot=False,
        fittype="linear",
        filename="plots/Z_scaling.pdf",
        **kwargs,
    ):
        if xaxis == "mu":
            if ainv is None:
                x = self.am * self.ainv
            else:
                x = self.am * ainv
        elif xaxis == "amu":
            x = self.am

        if plot:
            fig, ax = self.plot_Z(xaxis=xaxis, pass_plot=plot, filename=filename)

        matrix = np.zeros(shape=(self.N_ops, self.N_ops))
        errors = np.zeros(shape=(self.N_ops, self.N_ops))
        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=self.Z.val[:, i, j],
                    err=self.Z.err[:, i, j],
                    btsp=self.Z.btsp[:, :, i, j],
                )

                if fittype == "linear":
                    indices = np.sort(self.closest_n_points(m, x.val, n=2))
                    x_1, y_1 = x[indices[0]], y[indices[0]]
                    x_2, y_2 = x[indices[1]], y[indices[1]]
                    slope = (y_2 - y_1) / (x_2 - x_1)
                    intercept = y_1 - slope * x_1

                    pred = intercept + slope * m
                    x_grain = np.linspace(x.val[indices[0]], x.val[indices[-1]], 20)
                    pred_grain = join_stats([intercept + slope * g for g in x_grain])

                elif fittype == "quadratic":
                    indices = np.sort(self.closest_n_points(m, x.val, n=3))
                    x_1, y_1 = x[indices[0]], y[indices[0]]
                    x_2, y_2 = x[indices[1]], y[indices[1]]
                    x_3, y_3 = x[indices[2]], y[indices[2]]

                    a = (
                        y_1 / ((x_1 - x_2) * (x_1 - x_3))
                        + y_2 / ((x_2 - x_1) * (x_2 - x_3))
                        + y_3 / ((x_3 - x_1) * (x_3 - x_2))
                    )

                    b = (
                        -y_1 * (x_2 + x_3) / ((x_1 - x_2) * (x_1 - x_3))
                        - y_2 * (x_1 + x_3) / ((x_2 - x_1) * (x_2 - x_3))
                        - y_3 * (x_1 + x_2) / ((x_3 - x_1) * (x_3 - x_2))
                    )

                    c = (
                        y_1 * x_2 * x_3 / ((x_1 - x_2) * (x_1 - x_3))
                        + y_2 * x_1 * x_3 / ((x_2 - x_1) * (x_2 - x_3))
                        + y_3 * x_1 * x_2 / ((x_3 - x_1) * (x_3 - x_2))
                    )

                    pred = a * (m**2) + b * m + c
                    x_grain = np.linspace(x.val[indices[0]], x.val[indices[-1]], 20)
                    pred_grain = join_stats([a * (g**2) + b * g + c for g in x_grain])

                else:
                    print("Choose fittype linear or quadratic.")
                    break

                matrix[i, j] = pred.val
                errors[i, j] = pred.err

                if plot:
                    ax[i, j].errorbar(
                        m,
                        matrix[i, j],
                        yerr=errors[i, j],
                        c="r",
                        fmt="o",
                        label=err_disp(pred.val, pred.err),
                        capsize=4,
                    )
                    ax[i, j].fill_between(
                        x_grain,
                        pred_grain.val + pred_grain.err,
                        pred_grain.val - pred_grain.err,
                        color="r",
                        alpha=0.1,
                        label=fittype,
                    )
                    ax[i, j].legend()

        if plot:
            call_PDF(filename)

        if self.mask[0, 0] == False:
            matrix[0, 0] = 1

        Z = stat(val=matrix, err=errors, btsp="fill")

        if "rotate" in kwargs:
            rot_mtx = kwargs["rotate"]
            Z = stat(
                val=rot_mtx @ Z.val @ np.linalg.inv(rot_mtx),
                err="fill",
                btsp=np.array(
                    [
                        rot_mtx @ Z.btsp[k,] @ np.linalg.inv(rot_mtx)
                        for k in range(N_boot)
                    ]
                ),
            )

        return Z

    def scale_evolve(self, mu2, mu1, **kwargs):
        Z1 = self.interpolate(mu1, type="mu", **kwargs)
        Z2 = self.interpolate(mu2, **kwargs)
        sigma = stat(
            val=Z2.val @ np.linalg.inv(Z1.val),
            err="fill",
            btsp=np.array(
                [Z2.btsp[k,] @ np.linalg.inv(Z1.btsp[k,]) for k in range(N_boot)]
            ),
        )
        return sigma

    def plot_Z(
        self, xaxis="mu", filename="plots/Z_scaling.pdf", pass_plot=False, **kwargs
    ):
        x = self.am if xaxis == "am" else (self.am * self.ainv)

        fig, ax = plt.subplots(nrows=self.N_ops, ncols=self.N_ops, figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
            if self.mask[i, j]:
                y = stat(
                    val=self.Z.val[:, i, j],
                    err=self.Z.err[:, i, j],
                    btsp=self.Z.btsp[:, :, i, j],
                )
                ax[i, j].errorbar(
                    x.val, y.val, yerr=y.err, xerr=x.err, fmt="o", capsize=4
                )

                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
                if i == 1 or i == 3:
                    ax[i, j].set_xticks([])
            else:
                ax[i, j].axis("off")
        plt.suptitle(
            r"$Z_{ij}^{"
            + self.ens
            + r"}/Z_{"
            + self.norm
            + r"}$ vs renormalisation scale $\mu$",
            y=0.9,
        )

        if pass_plot:
            return fig, ax
        else:
            call_PDF(filename)
            print(f"Saved plot to {filename}.")

    def plot_sigma(
        self, xaxis="mu", mu1=None, mu2=3.0, filename="plots/Z_running.pdf", **kwargs
    ):
        x = self.am.val if xaxis == "am" else (self.am * self.ainv).val
        xerr = self.am.err if xaxis == "am" else (self.am * self.ainv).err

        if mu1 == None:
            sigmas = [self.scale_evolve(mu2, mom) for mom in list(x)]
            sig_str = (
                r"$\sigma_{ij}^{" + self.ens + r"}(" + str(mu2) + r"\leftarrow\mu)$"
            )
        else:
            sigmas = [self.scale_evolve(mom, mu1) for mom in list(x)]
            sig_str = (
                r"$\sigma_{ij}^{" + self.ens + r"}(\mu\leftarrow " + str(mu1) + r")$"
            )

        fig, ax = plt.subplots(
            nrows=self.N_ops, ncols=self.N_ops, sharex="col", figsize=(16, 16)
        )
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if self.mask[i, j]:
                y = [sig.val[i, j] for sig in sigmas]
                yerr = [sig.err[i, j] for sig in sigmas]

                ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o", capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis("off")
        if self.bag:
            plt.suptitle(sig_str + r"for $Z_{ij}/Z_{A/P}^2$", y=0.9)
        else:
            plt.suptitle(sig_str + r"for $Z_{ij}/Z_A^2$", y=0.9)

        call_PDF(filename)

    def load_fq_Z(
        self, norm="A", masses=None, resid_mask=True, pass_val=False, **kwargs
    ):
        if masses == None:
            masses = self.masses

        a1, a2 = self.action
        datafile = f"{datapath}/action{a1}_action{a2}/"
        datafile += "__".join(
            [
                "NPR",
                self.ens,
                params[self.ens]["baseactions"][a1],
                params[self.ens]["baseactions"][a2],
                self.scheme,
            ]
        )
        datafile += ".h5"
        data = h5py.File(datafile, "r")[str(masses)]

        key = "fourquark"
        self.am = stat(
            val=data[key]["ap"][:], err=np.zeros(len(data[key]["ap"][:])), btsp="fill"
        )
        self.N_mom = len(self.am.val)
        Z_ij_Z_q_2 = stat(
            val=(data[key]["central"][:]).real,
            err=data[key]["errors"][:],
            btsp=(data[key]["bootstrap"][:]).real,
        )

        if resid_mask:
            F = fq_qslash_F if self.scheme == "qslash" else fq_gamma_F
            Z_mask = np.zeros(shape=Z_ij_Z_q_2.val.shape)
            Z_mask_boot = np.zeros(shape=Z_ij_Z_q_2.btsp.shape)
            try:
                for m in range(self.N_mom):
                    Lambda = (np.linalg.inv(Z_ij_Z_q_2.val[m]) @ F).T
                    Lambda = fq_mask * Lambda
                    Z_mask[m,] = F @ np.linalg.inv(Lambda.T)
                    for k in range(N_boot):
                        Lambda_k = (np.linalg.inv(Z_ij_Z_q_2.btsp[k, m]) @ F).T
                        Lambda_k = fq_mask * Lambda_k
                        Z_mask_boot[
                            k,
                            m,
                        ] = F @ np.linalg.inv(Lambda_k.T)
                Z_ij_Z_q_2 = stat(val=Z_mask, err="fill", btsp=Z_mask_boot)
            except np.linalg.LinAlgError:
                print(self.mask)
                raise

        key = "bilinear"
        if norm == "bag":
            Z_ij_bag = Z_ij_Z_q_2.val
            Z_ij_bag[:, :, 0] = np.array(
                [
                    Z_ij_bag[m, :, 0] * (data[key]["A"]["central"][m] ** (-2))
                    for m in range(self.N_mom)
                ]
            )
            Z_ij_bag[:, :, 1:] = np.array(
                [
                    Z_ij_bag[m, :, 1:] * (data[key]["S"]["central"][m] ** (-2))
                    for m in range(self.N_mom)
                ]
            )

            Z_ij_bag_btsp = Z_ij_Z_q_2.btsp
            Z_ij_bag_btsp[:, :, :, 0] = np.array(
                [
                    [
                        Z_ij_bag_btsp[k, m, :, 0]
                        * (data[key]["A"]["bootstrap"][k, m] ** (-2))
                        for m in range(self.N_mom)
                    ]
                    for k in range(N_boot)
                ]
            )
            Z_ij_bag_btsp[:, :, :, 1:] = np.array(
                [
                    [
                        Z_ij_bag_btsp[k, m, :, 1:]
                        * (data[key]["S"]["bootstrap"][k, m] ** (-2))
                        for m in range(self.N_mom)
                    ]
                    for k in range(N_boot)
                ]
            )

            Z = stat(val=Z_ij_bag, err="fill", btsp=Z_ij_bag_btsp)
            if not pass_val:
                self.Z = Z
            else:
                return Z

        elif norm == "11":
            Z = stat(
                val=[
                    Z_ij_Z_q_2.val[m, :, :] / Z_ij_Z_q_2.val[m, 0, 0]
                    for m in range(self.N_mom)
                ],
                err="fill",
                btsp=[
                    [
                        Z_ij_Z_q_2.btsp[k, m, :, :] / Z_ij_Z_q_2.btsp[k, m, 0, 0]
                        for m in range(self.N_mom)
                    ]
                    for k in range(N_boot)
                ],
            )
            if not pass_val:
                self.Z = Z
            else:
                return Z

        elif norm in bilinear.currents:
            Z_bl_Z_q = stat(
                val=data[key][norm]["central"][:],
                err=data[key][norm]["errors"][:],
                btsp=data[key][norm]["bootstrap"][:],
            )
            Z = stat(
                val=[
                    Z_ij_Z_q_2.val[m, :, :] * (Z_bl_Z_q.val[m] ** (-2))
                    for m in range(self.N_mom)
                ],
                err="fill",
                btsp=np.array(
                    [
                        [
                            Z_ij_Z_q_2.btsp[k, m, :, :] * (Z_bl_Z_q.btsp[k, m] ** (-2))
                            for m in range(self.N_mom)
                        ]
                        for k in range(N_boot)
                    ]
                ),
            )
            if not pass_val:
                self.Z = Z
            else:
                return Z
        else:
            print("Normalisation not recognised!")

    def closest_n_points(self, target, values, n, **kwargs):
        diff = np.abs(np.array(values) - np.array(target))
        sort = np.sort(diff)
        closest_idx = []
        for n_idx in range(n):
            nth_closest_point = list(diff).index(sort[n_idx])
            closest_idx.append(nth_closest_point)
        return closest_idx


class bag_analysis:
    def __init__(self, ensemble, obj="bag", action=(0, 0), **kwargs):
        self.ens = ensemble
        self.action = action
        self.ra = ratio_analysis(self.ens)
        self.obj = obj
        if obj == "bag":
            norm = "bag"
            self.bag = self.ra.B_N
            if self.ens == "F1M":
                if fit_file == "Tobi":
                    corr_file = h5py.File("F1M_correction_factors_Tobi.h5", "r")
                    corr_data = corr_file["ratios_0.022170_over_0.021440"]
                    corrections = join_stats(
                        [
                            stat(
                                val=corr_data[f"B{i+1}"]["central"][:][0],
                                err=corr_data[f"B{i+1}"]["error"][:][0],
                                btsp=np.array(corr_data[f"B{i+1}"]["Bootstraps"][:])[
                                    :, 0
                                ],
                            )
                            for i in range(5)
                        ]
                    )
                else:
                    corrections = stat(
                        val=[1.00496, 1.004231, 1.003035, 1.003208, 1.003593],
                        err=[0.00010, 0.000062, 0.000061, 0.000077, 0.000067],
                        btsp="fill",
                    )
                self.bag = corrections * self.bag

        elif obj == "ratio":
            norm = "11"
            self.bag = self.ra.ratio
            if self.ens == "F1M":
                if fit_file == "Tobi":
                    corr_file = h5py.File("F1M_correction_factors_Tobi.h5", "r")
                    corr_data = corr_file["ratios_0.022170_over_0.021440"]
                    one = stat(val=1, err=0, btsp="fill")
                    corrections = join_stats(
                        [one]
                        + [
                            stat(
                                val=corr_data[f"R{i+1}"]["central"][:][0],
                                err=corr_data[f"R{i+1}"]["error"][:][0],
                                btsp=np.array(corr_data[f"R{i+1}"]["Bootstraps"][:])[
                                    :, 0
                                ],
                            )
                            for i in range(1, 5)
                        ]
                    )
                else:
                    corrections = stat(
                        val=[1.0, 0.97007, 0.96891, 0.96908, 0.96945],
                        err=[0.0, 0.00017, 0.00017, 0.00019, 0.00019],
                        btsp="fill",
                    )
                self.bag = corrections * self.bag
        elif obj == "ratio2":
            norm = "11/AS"
            self.bag = self.ra.ratio2

        self.ainv = stat(
            val=params[self.ens]["ainv"],
            err=params[self.ens]["ainv_err"],
            btsp="seed",
            seed=ensemble_seeds[self.ens],
        )
        self.a_sq = self.ainv ** (-2)
        self.ms_phys = (
            stat(
                val=params[self.ens]["ams_phys"],
                err=params[self.ens]["ams_phys_err"],
                btsp="seed",
                seed=ensemble_seeds[self.ens],
            )
            * self.ainv
        )

        self.ms_sea = self.ainv * params[self.ens]["ams_sea"]
        self.ms_diff = (self.ms_sea - self.ms_phys) / self.ms_phys

        self.f_pi = f_pi_PDG / self.ainv

        self.m_pi = load_info("m_0", self.ens, meson="ll")
        if self.m_pi.N_boot != N_boot:
            self.m_pi = stat(val=self.m_pi.val, err=self.m_pi.err, btsp="fill")
        self.m_f_sq = (self.m_pi**2) / (self.f_pi**2)

    def ansatz(self, param, operator, fit="central", log=False, **kwargs):
        op_idx = operators.index(operator)
        if fit == "central":
            a_sq = self.a_sq.val
            m_f_sq = self.m_f_sq.val
            PDG = m_f_sq_PDG.val
            ms_diff = self.ms_diff.val
        else:
            k = kwargs["k"]
            a_sq = self.a_sq.btsp[k]
            m_f_sq = self.m_f_sq.btsp[k]
            PDG = m_f_sq_PDG.btsp[k]
            ms_diff = self.ms_diff.btsp[k]

        def mpi_dep(m_f_sq):
            f = param[2] * m_f_sq
            if "addnl_terms" in kwargs:
                if kwargs["addnl_terms"] == "m4":
                    f += param[3] * (m_f_sq**2)
            return f

        func = param[0] + param[1] * a_sq + (mpi_dep(m_f_sq) - mpi_dep(PDG))
        if "addnl_terms" in kwargs:
            if kwargs["addnl_terms"] == "a4":
                func += param[3] * (a_sq**2)
            elif kwargs["addnl_terms"] == "del_ms":
                func += param[3] * ms_diff
        if log:
            chir_log_coeffs = chiral_logs(obj=self.obj, **kwargs)
            chir_log_coeffs = chir_log_coeffs / ((4 * np.pi) ** 2)
            log_ratio = m_f_sq * (f_pi_PDG.val**2) / (Lambda_QCD**2)
            log_term = chir_log_coeffs[op_idx] * np.log(log_ratio)
            func += param[0] * log_term * m_f_sq

        return func


class ratio_analysis:
    def __init__(self, ens, action=(0, 0), **kwargs):
        self.ens = ens
        self.op_recon()

    def op_recon(self, **kwargs):
        self.m_P = load_info("m_0", self.ens)
        self.ZP_L_0 = load_info("pLL_0", self.ens)
        self.ZA_L_0 = load_info("aLL_0", self.ens)
        if fit_file == "Tobi":
            self.ZP_L_0 = self.ZP_L_0 * (self.m_P) ** 0.5
            self.ZA_L_0 = self.ZA_L_0 * (self.m_P) ** 0.5

        self.gr_O_gr = load_info("gr-O-gr", self.ens)
        self.N_boot = self.gr_O_gr.N_boot

        self.Ni = norm_factors()
        B1 = stat(
            val=self.gr_O_gr.val[0] / (self.Ni[0] * self.ZA_L_0.val[0] ** 2),
            err="fill",
            btsp=np.array(
                [
                    self.gr_O_gr.btsp[k, 0] / (self.Ni[0] * self.ZA_L_0.btsp[k, 0] ** 2)
                    for k in range(self.N_boot)
                ]
            ),
        )

        B2_5 = [
            stat(
                val=self.gr_O_gr.val[i] / (self.Ni[i] * self.ZP_L_0.val[i] ** 2),
                err="fill",
                btsp=np.array(
                    [
                        self.gr_O_gr.btsp[k, i]
                        / (self.Ni[i] * self.ZP_L_0.btsp[k, i] ** 2)
                        for k in range(self.N_boot)
                    ]
                ),
            )
            for i in range(1, len(operators))
        ]

        Bs = [B1] + B2_5
        self.bag = stat(
            val=[B.val for B in Bs],
            err=[B.err for B in Bs],
            btsp=np.array([[B.btsp[k] for B in Bs] for k in range(self.N_boot)]),
        )
        Ni_diag = stat(
            val=np.diag(self.Ni),
            err="fill",
            btsp=np.array([np.diag(self.Ni) for k in range(self.N_boot)]),
        )
        self.B_N = Ni_diag @ self.bag
        self.ratio = self.gr_O_gr / self.gr_O_gr[0]
        if self.B_N.N_boot != N_boot:
            self.B_N = stat(val=self.B_N.val, err=self.B_N.err, btsp="fill")
            self.ratio = stat(val=self.ratio.val, err=self.ratio.err, btsp="fill")
