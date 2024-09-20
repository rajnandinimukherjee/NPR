from externalleg import *
import pandas as pd

# =====bilinear projectors==============================
bl_gamma_proj = {
    "S": [Gamma["I"]],
    "P": [Gamma["5"]],
    "V": [Gamma[i] for i in dirs],
    "A": [Gamma[i] @ Gamma["5"] for i in dirs],
    "T": sum(
        [
            [Gamma[dirs[i]] @ Gamma[dirs[j]] for j in range(i + 1, 4)]
            for i in range(0, 4 - 1)
        ],
        [],
    ),
}

bl_gamma_F = {
    k: np.trace(np.sum([mtx @ mtx for mtx in bl_gamma_proj[k]], axis=0))
    for k in currents
}


class bilinear:
    currents = ["S", "P", "V", "A", "T"]
    schemes = ["gamma", "qslash"]
    obj = "bilinears"
    prefix = "bi_"

    def __init__(
        self,
        ensemble,
        prop1,
        prop2,
        scheme="gamma",
        mres=None,
        Z_A_input=None,
        cfgs=None,
        **kwargs,
    ):

        if ensemble[:3]=='KEK':
            data = KEK_path+ensemble+'/results'
        else:
            data = path+ensemble
            data += "S/results" if ensemble[-1] not in ["M", "S"] else "/results"
        self.ens = ensemble
        self.L = params[ensemble]["XX"]
        if cfgs == None:
            self.cfgs = sorted(os.listdir(data)[1:])
        else:
            self.cfgs = cfgs

        self.N_cf = len(self.cfgs)
        self.filename = prop1.filename + "__" + prop2.filename
        self.h5_path = f"{data}/{self.cfgs[0]}/NPR/{self.obj}/{self.prefix}{self.filename}.{self.cfgs[0]}.h5"
        self.N_bl = len(h5py.File(self.h5_path, "r")["Bilinear"].keys())

        self.bilinears = np.array(
            [
                np.empty(shape=(self.N_cf, 12, 12), dtype="complex128")
                for i in range(self.N_bl)
            ],
            dtype=object,
        )
        for cf in range(self.N_cf):
            c = self.cfgs[cf]
            h5_path = f"{data}/{c}/NPR/{self.obj}/{self.prefix}{self.filename}.{c}.h5"
            h5_data = h5py.File(h5_path, "r")["Bilinear"]
            for i in range(self.N_bl):
                bilinear_i = h5_data[f"Bilinear_{i}"]["corr"][0, 0, :]
                self.bilinears[i][cf, :] = (
                    np.array(bilinear_i["re"] + bilinear_i["im"] * 1j)
                    .swapaxes(1, 2)
                    .reshape((12, 12))
                )
        self.pOut = [
            int(x)
            for x in h5_data["Bilinear_0"]["info"]
            .attrs["pOut"][0]
            .decode()
            .rsplit(".")[:-1]
        ]
        self.pIn = [
            int(x)
            for x in h5_data["Bilinear_0"]["info"]
            .attrs["pIn"][0]
            .decode()
            .rsplit(".")[:-1]
        ]
        self.org_gammas = np.array(
            [
                h5_data[f"Bilinear_{i}"]["info"].attrs["gamma"][0].decode()
                for i in range(self.N_bl)
            ]
        )
        self.gammas = [
            [x for x in list(self.org_gammas[i]) if x in list(gamma.keys())]
            for i in range(self.N_bl)
        ]

        self.prop_in = prop1 if prop1.momentum == self.pIn else prop2
        self.prop_out = prop1 if prop1.momentum == self.pOut else prop2
        self.tot_mom = self.prop_in.total_momentum - self.prop_out.total_momentum
        self.mom_sq = self.prop_in.momentum_squared
        self.q = self.mom_sq**0.5

        self.avg_bilinear = np.array(
            [np.mean(self.bilinears[i,], axis=0) for i in range(self.N_bl)],
            dtype=object,
        )
        self.btsp_bilinear = np.array(
            [bootstrap(self.bilinears[i,], K=N_boot, 
                       seed=ensemble_seeds[ensemble])
             for i in range(self.N_bl)]
        )

        self.m_q = float(self.prop_in.info["am"])
        if mres == None:
            self.mres = stat(val=0, err=0, btsp="fill")
        else:
            self.mres = mres
        self.Z_A_input = Z_A_input

    def gamma_Z(self, operators, **kwargs):
        projected = {
            c: np.trace(
                np.sum(
                    [
                        bl_gamma_proj[c][i] @ operators[c][i]
                        for i in range(len(operators[c]))
                    ],
                    axis=0,
                )
            )
            for c in bilinear.currents
        }

        gamma_Z = {c: (bl_gamma_F[c] / projected[c]).real for c in bilinear.currents}
        return projected, gamma_Z

    def calc_Z_q(self, S_inv, p_vec, **kwargs):
        pslash = np.sum([p_vec[i] * Gamma[dirs[i]] for i in range(len(dirs))], axis=0)
        p_sq = np.linalg.norm(p_vec) ** 2
        Z_q = np.trace(-1j * S_inv @ pslash).real / (12 * p_sq)
        return Z_q

    def qslash_Z(self, operators, S, mres, renorm="mSMOM", Z_A_input=None, **kwargs):

        S_inv_in, S_inv_out = S

        p_vec_in = self.prop_in.total_momentum
        Z_q_in = self.calc_Z_q(S_inv_in, p_vec_in)

        p_vec_out = self.prop_out.total_momentum
        Z_q_out = self.calc_Z_q(S_inv_out, p_vec_out)

        Z_q = (Z_q_in * Z_q_out) ** 0.5
        q_vec = self.tot_mom
        qslash = np.sum([q_vec[i] * Gamma[dirs[i]] for i in range(len(dirs))], axis=0)
        q_sq = np.linalg.norm(q_vec) ** 2

        if Z_A_input != None:
            Z_q_over_Z_V = np.trace(
                np.sum([q_vec[i] * operators["V"][i] for i in range(len(dirs))], axis=0)
                @ qslash
            ).real / (12 * q_sq)
            Z_q = Z_q_over_Z_V * Z_A_input

        Z_P = Z_q * self.gamma_Z(operators)[1]["P"]
        Z_T = Z_q * self.gamma_Z(operators)[1]["T"]
        Z_V = Z_q / (
            np.trace(
                np.sum([q_vec[i] * operators["V"][i] for i in range(len(dirs))], axis=0)
                @ qslash
            ).real
            / (12 * q_sq)
        )
        m_q = self.m_q + mres
        A1 = (
            1j
            * np.trace(
                np.sum([q_vec[i] * operators["A"][i] for i in range(len(dirs))], axis=0)
                @ Gamma["5"]
            ).imag
        )
        A2 = np.trace(
            np.sum([q_vec[i] * operators["A"][i] for i in range(len(dirs))], axis=0)
            @ Gamma["5"]
            @ qslash
        ).real
        P = np.trace(1j * operators["P"][0] @ Gamma["5"] @ qslash).real
        S = (np.trace(S_inv_in) + np.trace(S_inv_out)).real / 2

        if renorm == "SMOM":
            Z_A = 12 * q_sq * Z_q / A2
        else:
            Z_A = (144 * q_sq * (Z_q**2) - 2 * Z_P * S * P) / (
                12 * Z_q * A2 + 1j * Z_P * A1 * P
            )

        Z_m = (S + (Z_A * A1 * 1j) / 2) / (12 * m_q * Z_q)
        Z_mm_q = (S + (Z_A * A1 * 1j) / 2) / (12 * Z_q)

        s_term = np.trace(operators["S"][0])
        mass_term = 4 * m_q * Z_m * Z_P * P
        if renorm == "SMOM":
            Z_S = 12 * Z_q / s_term
        else:
            Z_S = (12 * q_sq * Z_q - mass_term) / (q_sq * s_term)

        qslash_Z = {
            "S": Z_S.real,
            "P": Z_P.real,
            "V": Z_V.real,
            "A": Z_A.real,
            "T": Z_T.real,
            "m": Z_m.real,
            "mam_q": Z_mm_q.real,
            "q": Z_q,
            "m_q": m_q,
        }

        # ainv = params[self.ens]['ainv']
        # if np.round(((ainv**2)*q_sq)**0.5,1)==2.1:
        #    temp_dict = {'Z_A':Z_A, 'Z_q':Z_q,
        #            'S':S, 'A1':A1, 'A2':A2, 'P':P,
        #            'm':m_q, 'Z_m':Z_m.real}
        #    for key in temp_dict:
        #        print(key, temp_dict[key])
        #    pdb.set_trace()
        return qslash_Z

    def construct_operators(self, S_in, S_out, Gs, **kwargs):
        amputated = np.array([S_out @ Gs[b] @ S_in for b in range(self.N_bl)])
        operators = {
            "S": [amputated[self.gammas.index(["I"])]],
            "P": [amputated[self.gammas.index(["5"])]],
            "V": [amputated[self.gammas.index([i])] for i in dirs],
            "A": [amputated[self.gammas.index([i, "5"])] for i in dirs],
            "T": sum(
                [
                    [
                        amputated[self.gammas.index([dirs[i], dirs[j]])]
                        for j in range(i + 1, 4)
                    ]
                    for i in range(0, 4 - 1)
                ],
                [],
            ),
        }
        return operators

    def NPR(self, massive=False, **kwargs):
        # ==central===
        S_in = self.prop_in.inv_propagator.val
        S_out = self.prop_out.inv_outgoing_propagator.val
        operators = self.construct_operators(S_in, S_out, self.avg_bilinear)
        if not massive:
            projected, Z = self.gamma_Z(operators)
        else:
            mres = self.mres.val
            Z_A_input = self.Z_A_input.val if self.Z_A_input != None else None
            Z = self.qslash_Z(
                operators, [S_in, S_out], mres, Z_A_input=Z_A_input, **kwargs
            )

        # ==bootstrap===
        Z_btsp = {c: np.zeros(N_boot) for c in Z.keys()}
        projected_btsp = {c: np.zeros(N_boot, dtype=object) for c in Z.keys()}
        for k in range(N_boot):
            S_in = self.prop_in.inv_propagator.btsp[k,]
            S_out = self.prop_out.inv_outgoing_propagator.btsp[k,]
            operators = self.construct_operators(
                S_in, S_out, self.btsp_bilinear[:, k, :]
            )
            if not massive:
                proj_k, Z_k = self.gamma_Z(operators)
                for c in Z_k.keys():
                    projected_btsp[c][k] = proj_k[c]
                    Z_btsp[c][k] = Z_k[c]
            else:
                mres = self.mres.btsp[k]
                Z_A_input = self.Z_A_input.btsp[k] if self.Z_A_input != None else None
                Z_k = self.qslash_Z(
                    operators, [S_in, S_out], mres, Z_A_input=Z_A_input, **kwargs
                )
                for c in Z_k.keys():
                    Z_btsp[c][k] = Z_k[c].real

        self.Z = {
            key: stat(val=Z[key], err="fill", btsp=Z_btsp[key]) for key in Z.keys()
        }

        if not massive:
            self.proj = {
                key: stat(val=projected[key], err="fill", btsp=projected_btsp[key])
                for key in Z.keys()
            }
