from basics import *
from check_data import *

datapath = "/".join(os.getcwd().split("/")[:-1]) + "/data"


class valence:
    eta_h_gamma = ("Gamma5", "Gamma5")
    Z_A_gamma = ("Gamma5", "GammaTGamma5")
    path = datapath

    def __init__(self, ens, action=(0, 0), smeared=False, **kwargs):
        self.ens = ens
        self.action = action
        self.info = params[self.ens]
        self.ainv = stat(val=self.info["ainv"], err=self.info["ainv_err"], btsp="fill")
        self.T = int(self.info["TT"])
        self.all_masses = ["{:.4f}".format(m) for m in self.info["masses"]]
        self.N_mass = len(self.all_masses)
        self.mass_names = {
            self.all_masses[0]: "l",
            self.all_masses[1]: "l_double",
            self.all_masses[2]: "s_half",
            self.all_masses[3]: "s",
        }
        self.mass_names.update(
            {self.all_masses[m]: f"c{m-4}" for m in range(4, self.N_mass)}
        )

        self.sm = "_sm" if smeared else ""
        self.smeared = smeared
        self.cfgs = self.info["valence_cfgs"]
        df = check_valence(
            self.ens + "S" if ens in ["C1", "M1"] else self.ens, pass_only=True
        )
        if smeared:
            if self.ens[:2] == "M1":
                for mass in self.all_masses[-3:]:
                    df[mass].replace(16, 32, inplace=True)
            df.replace(32, 100, inplace=True)
        else:
            if self.ens[:2] == "M1":
                for mass in self.all_masses[-3:]:
                    df[mass].replace(16, 0, inplace=True)
            df.replace(16, 100, inplace=True)

        self.mass_cfgs = {m: df.index[df[m].eq(100)].to_list() for m in self.all_masses}
        for mass in self.all_masses.copy():
            if len(self.mass_cfgs[mass]) == 0:
                self.all_masses.remove(mass)

        self.cfgs_df = df

    def Z_A_correlator(
        self,
        mass,
        load=True,
        meson_num=33,
        N_src=16,
        save=True,
        end_only=False,
        fit_start=19,
        fit_end=28,
        **kwargs,
    ):

        a1, a2 = self.action
        fname = f"{self.path}/action{a1}_action{a2}/"
        fname += "__".join(
            [
                "NPR",
                self.ens,
                self.info["baseactions"][a1],
                self.info["baseactions"][a2],
            ]
        )
        fname += "_mSMOM.h5"
        grp_name = f"{str((mass, mass))}/PA0"

        if load:
            datapath = path + self.ens
            datapath += "S/results" if self.ens[-1] not in ["M", "S"] else "/results"
            massname = self.mass_names[mass]
            R = "R16" if massname[0] == "c" else "R08"
            cfgs = self.mass_cfgs[mass]

            t_src_range = range(0, self.T, int(self.T / N_src))
            corr = np.zeros(shape=(len(cfgs), N_src, self.T))

            for cf_idx, cf in enumerate(cfgs):
                filename = f"{datapath}/{cf}/conserved/"
                for t_idx, t_src in enumerate(t_src_range):
                    f_string = f"prop_{massname}_{R}_Z2_t" + "{:02d}".format(t_src)
                    f_string += "_p+0_+0_+0" + self.sm + f".{cf}.h5"
                    data = h5py.File(filename + f_string, "r")["wardIdentity"]
                    corr[cf_idx, t_idx, :] = np.roll(
                        np.array(data["PA0"][:]["re"]), -t_src
                    )

            corr = np.mean(corr, axis=1)
            corr = stat(val=np.mean(corr, axis=0), err="fill", btsp=bootstrap(corr))

            meson = self.meson_correlator(
                mass, meson_num=meson_num, fit_corr=False, load=False
            )

            corr_rolled = stat(
                val=np.roll(corr.val, 1),
                err=np.roll(corr.err, 1),
                btsp=np.roll(corr.btsp, 1, axis=1),
            )
            meson_rolled = stat(
                val=np.roll(meson.val, -1),
                err=np.roll(meson.err, -1),
                btsp=np.roll(meson.btsp, -1, axis=1),
            )
            corr_new = (corr + corr_rolled) / (meson * 2)
            corr_new = corr_new + (corr * 2) / (meson + meson_rolled)
            corr_new = corr_new * 0.5

            folded_corr = (corr_new[1:] + corr_new[::-1][:-1])[: int(self.T / 2)] * 0.5

            if end_only:
                fit = folded_corr[-4]
            else:
                fit_points = np.arange(fit_start, fit_end)

                def constant_Z_A(t, param, **kwargs):
                    return param[0] * np.ones(len(t))

                x = stat(val=fit_points, btsp="fill")
                y = folded_corr[fit_points]
                res = fit_func(x, y, constant_Z_A, [1, 0])
                fit = res[0] if res[0].val != 0 else folded_corr[-1]
            print(f"For am_q={mass}, Z_A={err_disp(fit.val,fit.err)}")

            if save:
                f = h5py.File(fname, "a")
                if grp_name in f:
                    del f[grp_name]
                if grp_name + "_sm" in f:
                    del f[grp_name + "_sm"]
                mes_grp = f.create_group(grp_name)
                mes_grp.attrs["configs"] = cfgs
                mes_grp.create_dataset("central", data=corr.val)
                mes_grp.create_dataset("errors", data=corr.err)
                mes_grp.create_dataset("bootstrap", data=corr.btsp)

                mes_grp.create_dataset("fit/central", data=fit.val)
                mes_grp.create_dataset("fit/errors", data=fit.err)
                mes_grp.create_dataset("fit/bootstrap", data=fit.btsp)
                f.close()
                print(f"Saved amres data and fit to {grp_name} in {fname}")
        else:
            f = h5py.File(fname, "r")[grp_name]
            corr = stat(
                val=np.array(f["central"][:]),
                err=np.array(f["errors"][:]),
                btsp=np.array(f["bootstrap"][:]),
            )
            fit = stat(
                val=np.array(f["fit/central"]),
                err=np.array(f["fit/errors"]),
                btsp=np.array(f["fit/bootstrap"][:]),
            )
        return corr, fit

    def compute_Z_A(
        self,
        masses=None,
        plot=False,
        meson_num=33,
        leave_out=None,
        xlabel="",
        filename="",
        **kwargs,
    ):

        self.Z_A = {}
        if masses == None:
            masses = self.all_masses

        for mass in masses:
            corr, fit = self.Z_A_correlator(mass, **kwargs)
            meson = self.meson_correlator(
                mass, meson_num=meson_num, load=False, fit_corr=False
            )
            corr_rolled = stat(
                val=np.roll(corr.val, 1),
                err=np.roll(corr.err, 1),
                btsp=np.roll(corr.btsp, 1, axis=1),
            )
            meson_rolled = stat(
                val=np.roll(meson.val, -1),
                err=np.roll(meson.err, -1),
                btsp=np.roll(meson.btsp, -1, axis=1),
            )
            corr_new = (corr + corr_rolled) / (meson * 2)
            corr_new = corr_new + (corr * 2) / (meson + meson_rolled)
            corr_new = corr_new * 0.5
            folded_corr = (corr_new[1:] + corr_new[::-1][:-1])[: int(self.T / 2)] * 0.5

            self.Z_A[mass] = fit
            if plot:
                fig, ax = plt.subplots(figsize=(3, 2))
                start = np.where(
                    (folded_corr.val > fit.val - 5 * fit.err)
                    & (folded_corr.val < fit.val + 8 * fit.err)
                )[0][0]
                x = np.arange(start, int(self.T / 2) - 2)
                y = folded_corr[start:-2]
                ax.errorbar(x, y.val, yerr=y.err, fmt="o", capsize=4, color="k", ms=4)
                ax.axhspan(
                    self.Z_A[mass].val + self.Z_A[mass].err,
                    self.Z_A[mass].val - self.Z_A[mass].err,
                    facecolor="tab:purple",
                    edgecolor="None",
                    alpha=0.4,
                )
                ax.set_xlabel(r"$t/a$")
                ax.set_ylabel(r"$Z_A^\mathrm{eff}(t)$")
                # ax.set_ylabel(r'$\frac{\langle J_\mathcal{A}(t) J_5(0)\rangle}{\langle J_A(t)J_5(0)\rangle}$',
                #              fontsize=12)
                label = r"$am_\mathrm{input}=$" + mass
                ax.text(
                    0.5,
                    1.0,
                    label,
                    bbox=dict(facecolor="white", edgecolor="None"),
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                )

        if "amres" in self.__dict__.keys() and plot:
            fig, ax = plt.subplots(figsize=(3, 2))
            x = join_stats([self.amres[mass] + eval(mass) for mass in self.all_masses])
            if leave_out == None:
                leave_out = len(x.val)
            y = join_stats([self.Z_A[mass] for mass in self.all_masses])
            ax.errorbar(
                x.val[:leave_out],
                y.val[:leave_out],
                yerr=y.err[:leave_out],
                xerr=x.err[:leave_out],
                fmt="o",
                capsize=4,
                color="k",
                ms=4,
            )
            if xlabel == "":
                xlabel = r"$am_q=am_\mathrm{input}+am_\mathrm{res}$"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$Z_A$")
            # ax.text(0.5, 0.1, self.ens,
            #           va='center', ha='center',
            #           transform=ax.transAxes)

        if plot:
            if filename == "":
                filename = f"{self.ens}_Z_A.pdf"
            call_PDF(filename, open=True)

    def amres_correlator(
        self,
        mass,
        load=True,
        cfgs=None,
        meson_num=1,
        N_src=16,
        save=True,
        end_only=False,
        num_end_points=5,
        **kwargs,
    ):

        a1, a2 = self.action
        fname = f"{self.path}/action{a1}_action{a2}/"
        fname += "__".join(
            [
                "NPR",
                self.ens,
                self.info["baseactions"][a1],
                self.info["baseactions"][a2],
            ]
        )
        fname += "_mSMOM.h5"
        grp_name = f"{str((mass, mass))}/PJ5q"

        if load:
            datapath = path + self.ens
            datapath += "S/results" if self.ens[-1] not in ["M", "S"] else "/results"
            massname = self.mass_names[mass]
            R = "R16" if massname[0] == "c" else "R08"
            cfgs = self.mass_cfgs[mass]

            t_src_range = range(0, self.T, int(self.T / N_src))
            corr = np.zeros(shape=(len(cfgs), N_src, self.T))

            for cf_idx, cf in enumerate(cfgs):
                filename = f"{datapath}/{cf}/conserved/"
                for t_idx, t_src in enumerate(t_src_range):
                    f_string = f"prop_{massname}_{R}_Z2_t" + "{:02d}".format(t_src)
                    f_string += "_p+0_+0_+0" + self.sm + f".{cf}.h5"
                    data = h5py.File(filename + f_string, "r")["wardIdentity"]
                    corr[cf_idx, t_idx, :] = np.roll(
                        np.array(data["PJ5q"][:]["re"]), -t_src
                    )

            corr = np.mean(corr, axis=1)
            corr = stat(val=np.mean(corr, axis=0), err="fill", btsp=bootstrap(corr))

            meson, meson_fit = self.meson_correlator(
                mass, meson_num=meson_num, load=False
            )
            corr_new = corr / meson
            folded_corr = (corr_new[1:] + corr_new[::-1][:-1])[: int(self.T / 2)] * 0.5

            if end_only:
                fit = folded_corr[-4]
            else:
                fit_points = np.arange(int(self.T / 2))[-num_end_points:]

                def constant_mass(t, param, **kwargs):
                    return param[0] * np.ones(len(t))

                x = stat(val=fit_points, btsp="fill")
                y = folded_corr[fit_points]
                res = fit_func(x, y, constant_mass, [0.1, 0])
                fit = res[0] if res[0].val != 0 else folded_corr[-1]
            print(f"For am_q={mass}, am_res={err_disp(fit.val,fit.err)}")

            if save:
                f = h5py.File(fname, "a")
                if grp_name in f:
                    del f[grp_name]
                if grp_name + "_sm" in f:
                    del f[grp_name + "_sm"]
                mes_grp = f.create_group(grp_name)
                mes_grp.attrs["configs"] = cfgs
                mes_grp.create_dataset("central", data=corr.val)
                mes_grp.create_dataset("errors", data=corr.err)
                mes_grp.create_dataset("bootstrap", data=corr.btsp)

                mes_grp.create_dataset("fit/central", data=fit.val)
                mes_grp.create_dataset("fit/errors", data=fit.err)
                mes_grp.create_dataset("fit/bootstrap", data=fit.btsp)
                f.close()
                print(f"Saved amres data and fit to {grp_name} in {fname}")
        else:
            f = h5py.File(fname, "r")[grp_name]
            corr = stat(
                val=np.array(f["central"][:]),
                err=np.array(f["errors"][:]),
                btsp=np.array(f["bootstrap"][:]),
            )
            fit = stat(
                val=np.array(f["fit/central"]),
                err=np.array(f["fit/errors"]),
                btsp=np.array(f["fit/bootstrap"][:]),
            )
        return corr, fit

    def compute_amres(self, masses=None, plot=False, xlabel="", filename="", **kwargs):
        self.amres = {}
        if masses == None:
            masses = self.all_masses

        for mass in masses:
            corr, fit = self.amres_correlator(mass, **kwargs)
            meson, meson_fit = self.meson_correlator(mass, meson_num=1, load=False)
            corr_new = corr / meson
            folded_corr = (corr_new[1:] + corr_new[::-1][:-1])[: int(self.T / 2)] * 0.5

            self.amres[mass] = fit
            if plot:
                fig, ax = plt.subplots(figsize=(3, 2))
                start = np.where(folded_corr.val > fit.val - 5 * self.amres[mass].err)[
                    0
                ][0]
                x = np.arange(start, int(self.T / 2))
                y = folded_corr[start:]
                ax.errorbar(x, y.val, yerr=y.err, fmt="o", capsize=4, color="k", ms=4)
                ax.axhspan(
                    self.amres[mass].val + self.amres[mass].err,
                    self.amres[mass].val - self.amres[mass].err,
                    facecolor="tab:purple",
                    edgecolor="None",
                    alpha=0.4,
                )
                # ax.errorbar([x[-4]], [y.val[-4]],
                #            yerr=[y.err[-4]],
                #            fmt='o', capsize=4, color='tab:orange', ms=4)
                ax.set_xlabel(r"$t/a$")
                ax.set_ylabel(
                    r"$\frac{\langle J_{5q}(t)J_5(0)\rangle}{\langle J_5(t)J_5(0)\rangle}$",
                    fontsize=12,
                )
                label = r"$am_\mathrm{input}=$" + mass
                ax.text(
                    0.5,
                    1.0,
                    label,
                    bbox=dict(facecolor="white", edgecolor="None"),
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                )

        if plot:
            fig, ax = plt.subplots(figsize=(3, 2))
            x = stat(
                val=[eval(mass) for mass in self.all_masses],
                err=np.zeros(len(self.all_masses)),
                btsp="fill",
            )
            y = join_stats([self.amres[mass] for mass in self.all_masses])
            ax.errorbar(
                x.val,
                y.val,
                yerr=y.err,
                xerr=x.err,
                fmt="o",
                capsize=4,
                color="k",
                ms=4,
            )
            if xlabel == "":
                xlabel = r"$am_\mathrm{input}$"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$am_{res}$")
            # ax.text(0.5, 0.9, self.ens,
            #           va='center', ha='center',
            #           transform=ax.transAxes)
            if plot:
                if filename == "":
                    filename = f"{self.ens}_amres.pdf"
                call_PDF(filename, open=True)

    def meson_correlator(
        self,
        mass,
        load=True,
        cfgs=None,
        meson_num=1,
        N_src=16,
        save=True,
        fit_corr=True,
        end_only=False,
        fit_start=15,
        fit_end=30,
        **kwargs,
    ):

        a1, a2 = self.action
        fname = f"{self.path}/action{a1}_action{a2}/"
        fname += "__".join(
            [
                "NPR",
                self.ens,
                self.info["baseactions"][a1],
                self.info["baseactions"][a2],
            ]
        )
        fname += "_mSMOM.h5"
        grp_name = f"{str((mass, mass))}/meson_{meson_num}/corr"

        if load:
            datapath = path + self.ens
            datapath += "S/results" if self.ens[-1] not in ["M", "S"] else "/results"
            massname = self.mass_names[mass]
            R = "R16" if massname[0] == "c" else "R08"
            cfgs = self.mass_cfgs[mass]

            foldername = f"{massname}_{R}__{massname}_{R}"
            mes = f"meson_{meson_num}"

            t_src_range = range(0, self.T, int(self.T / N_src))
            corr = np.zeros(shape=(len(cfgs), N_src, self.T))

            for cf_idx, cf in enumerate(cfgs):
                filename = f"{datapath}/{cf}/mesons/{foldername}/meson_"
                for t_idx, t_src in enumerate(t_src_range):
                    f_string = f"prop_{massname}_{R}_Z2_t" + "{:02d}".format(t_src)
                    f_string += "_p+0_+0_+0" + self.sm
                    fstring = "__".join([f_string, f_string, f"snk_0_0_0.{cf}.h5"])
                    data = h5py.File(filename + fstring, "r")["meson"][mes]
                    corr[cf_idx, t_idx, :] = np.roll(
                        np.array(data["corr"][:]["re"]), -t_src
                    )

            corr = np.mean(corr, axis=1)
            corr = stat(val=np.mean(corr, axis=0), err="fill", btsp=bootstrap(corr))

            if fit_corr:
                folded_corr = (corr[1:] + corr[::-1][:-1])[: int(self.T / 2)] * 0.5
                div = ((folded_corr[2:] + folded_corr[:-2]) / folded_corr[1:-1]) * 0.5
                m_eff = div.use_func(np.arccosh)

                if end_only:
                    fit = m_eff[-4]
                else:

                    def constant_mass(t, param, **kwargs):
                        return param[0] * np.ones(len(t))

                    x = stat(val=np.arange(fit_start, fit_end), btsp="fill")
                    y = m_eff[fit_start:fit_end]
                    res = fit_func(x, y, constant_mass, [0.1, 0])
                    fit = res[0]
                print(f"For am_q={mass}, am_eta_h={err_disp(fit.val, fit.err)}")

            if save:
                f = h5py.File(fname, "a")
                if grp_name in f:
                    del f[grp_name]
                if grp_name + "_sm" in f:
                    del f[grp_name + "_sm"]
                mes_grp = f.create_group(grp_name)
                mes_grp.attrs["configs"] = cfgs
                mes_grp.create_dataset("central", data=corr.val)
                mes_grp.create_dataset("errors", data=corr.err)
                mes_grp.create_dataset("bootstrap", data=corr.btsp)

                if fit_corr:
                    mes_grp.create_dataset("fit/central", data=fit.val)
                    mes_grp.create_dataset("fit/errors", data=fit.err)
                    mes_grp.create_dataset("fit/bootstrap", data=fit.btsp)

                f.close()
                print(
                    f"Saved meson_{meson_num} corr data "
                    + f"and fit to {grp_name} in {fname}"
                )
        else:
            f = h5py.File(fname, "r")[grp_name]
            corr = stat(
                val=np.array(f["central"][:]),
                err=np.array(f["errors"][:]),
                btsp=np.array(f["bootstrap"][:]),
            )

            if fit_corr:
                fit = stat(
                    val=np.array(f["fit/central"]),
                    err=np.array(f["fit/errors"]),
                    btsp=np.array(f["fit/bootstrap"][:]),
                )

        if fit_corr:
            return corr, fit
        else:
            return corr

    def compute_eta_h(self, plot=False, xlabel="", filename="", **kwargs):
        self.eta_h_masses = {}
        for mass in self.all_masses:
            corr, fit = self.meson_correlator(mass, **kwargs)
            folded_corr = (corr[1:] + corr[::-1][:-1])[: int(self.T / 2)] * 0.5
            div = ((folded_corr[2:] + folded_corr[:-2]) / folded_corr[1:-1]) * 0.5
            m_eff = div.use_func(np.arccosh)

            self.eta_h_masses[mass] = fit

            if plot:
                fig, ax = plt.subplots(figsize=(3, 2))
                start = np.where(m_eff.val < fit.val + 10 * fit.err)[0][0]
                x = np.arange(start + 1, int(self.T / 2) - 1)
                y = m_eff[start:]
                ax.errorbar(x, y.val, yerr=y.err, color="k", fmt="o", capsize=4, ms=4)
                ax.axhspan(
                    fit.val + fit.err,
                    fit.val - fit.err,
                    facecolor="tab:purple",
                    edgecolor="None",
                    alpha=0.4,
                )
                ax.set_xlabel(r"$t/a$")
                ax.set_ylabel(r"$\left(aM_{\eta_h}\right)_\mathrm{eff}$")
                label = r"$am_\mathrm{input}=$" + mass
                ax.text(
                    0.5,
                    1.0,
                    label,
                    bbox=dict(facecolor="white", edgecolor="None"),
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                )

        if "amres" in self.__dict__.keys() and plot:
            fig, ax = plt.subplots(figsize=(3, 2))
            x = join_stats([self.amres[mass] + eval(mass) for mass in self.all_masses])
            y = join_stats(
                [self.ainv * self.eta_h_masses[mass] for mass in self.all_masses]
            )
            ax.errorbar(
                x.val,
                y.val,
                yerr=y.err,
                xerr=x.err,
                fmt="o",
                capsize=4,
                color="k",
                ms=4,
            )
            if xlabel == "":
                xlabel = r"$am_q = am_\mathrm{input}+am_\mathrm{res}$"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$M_{\eta_h}\,[\mathrm{GeV}]$")
            # ax.text(0.5, 0.1, self.ens,
            #           va='center', ha='center',
            #           transform=ax.transAxes)

        if plot:
            if filename == "":
                filename = f"{self.ens}_eta_h.pdf"
            call_PDF(filename, open=True)

    def calc_all(self, load=False, safe=True, **kwargs):
        if not load:
            if safe:
                self.all_masses = ["{:.4f}".format(m) for m in self.info["safe_masses"]]
            else:
                self.all_masses = ["{:.4f}".format(m) for m in self.info["masses"]]

        self.compute_eta_h(load=load, **kwargs)
        self.compute_amres(load=load, **kwargs)
        for mass in self.all_masses:
            self.meson_correlator(
                mass, meson_num=33, fit_corr=False, load=load, **kwargs
            )
        self.compute_Z_A(load=load, **kwargs)

    def interpolate_amres(self, M, start=0, stop=None, plot=False, **kwargs):
        if stop == None:
            stop = len(self.all_masses)

        x = join_stats([self.eta_h_masses[m] for m in self.all_masses[start:stop]])
        x = x * self.ainv
        y = join_stats([self.amres[m] for m in self.all_masses[start:stop]])

        lin_indices = np.sort(self.closest_n_points(M.val, x.val, n=2))
        x_1, y_1 = x[lin_indices[0]], y[lin_indices[0]]
        x_2, y_2 = x[lin_indices[1]], y[lin_indices[1]]
        slope = (y_2 - y_1) / (x_2 - x_1)
        intercept = y_1 - slope * x_1

        lin_pred = intercept + slope * M

        quad_indices = np.sort(self.closest_n_points(M.val, x.val, n=3))
        x_1, y_1 = x[quad_indices[0]], y[quad_indices[0]]
        x_2, y_2 = x[quad_indices[1]], y[quad_indices[1]]
        x_3, y_3 = x[quad_indices[2]], y[quad_indices[2]]

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

        quad_pred = a * (M**2) + b * M + c

        stat_err = max(lin_pred.err, quad_pred.err)
        sys_err = np.abs(quad_pred.val - lin_pred.val) / 2
        pred = stat(
            val=(lin_pred.val + quad_pred.val) / 2,
            err=(stat_err**2 + sys_err**2) ** 0.5,
            btsp="fill",
        )

        if plot:
            fig, ax = plt.subplots()
            ax.errorbar(x.val, y.val, xerr=x.err, yerr=y.err, fmt="o", capsize=4, ms=4)
            ax.errorbar(
                [M.val],
                [pred.val],
                xerr=M.err,
                yerr=pred.err,
                fmt="o",
                capsize=4,
                c="k",
                ms=4,
            )

            ax.set_xlabel(r"$M_{\eta_h}\,[\mathrm{GeV}]$")
            ax.set_ylabel(r"$am_\mathrm{res}$")
            ax.set_title(self.ens)
            filename = f"{self.ens}_amres_variation.pdf"
            call_PDF(filename, open=True)

        return pred

    def closest_n_points(self, target, values, n, **kwargs):
        diff = np.abs(np.array(values) - np.array(target))
        sort = np.sort(diff)
        closest_idx = []
        for n_idx in range(n):
            nth_closest_point = list(diff).index(sort[n_idx])
            closest_idx.append(nth_closest_point)
        return closest_idx

    def create_ens_table(self, **kwargs):
        self.calc_all()
        rv = [r"\begin{tabular}{c|ccc}"]
        rv += [r"\hline"]
        rv += [r"\hline"]

        rv += [
            r"$am_\mathrm{input}$ & $am_\mathrm{res}\times 10^3$ & $aM_{\eta_h}$ & $Z_A$ \\"
        ]
        rv += [r"\hline"]
        for mass in self.all_masses:
            amres, aM, ZA = self.amres[mass], self.eta_h_masses[mass], self.Z_A[mass]
            rv += [
                r"$"
                + "$ & $".join(
                    [mass] + [err_disp(o.val, o.err) for o in [amres * 1000, aM, ZA]]
                )
                + r"$ \\"
            ]

        rv += [r"\hline"]
        rv += [r"\hline"]
        rv += [r"\end{tabular}"]

        ens = self.ens + "S" if self.ens[-1] == "1" else self.ens
        filename = f"/Users/rajnandinimukherjee/PhD/thesis/inputs/MassiveNPR/tables/{ens}_valence.tex"

        f = open(filename, "w")
        f.write("\n".join(rv))
        f.close()
        print(f"valence table written to {filename}")
