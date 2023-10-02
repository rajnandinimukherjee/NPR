import itertools
from scipy.interpolate import interp1d
from NPR_classes import *

all_bag_data = h5py.File('kaon_bag_fits.h5', 'r')
bag_ensembles = [key for key in all_bag_data.keys() if key in UKQCD_ens]


def load_info(key, ens, ops, meson='ls', **kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson == 'ls':
        central_val = np.array([np.array(h5_data[
            op][key]['central']).item()
            for op in ops])
        error = np.array([np.array(h5_data[
            op][key]['error']).item()
            for op in ops])
        bootstraps = np.zeros(shape=(N_boot, 5))
        for i in range(5):
            op = ops[i]
            bootstraps[:, i] = np.array(h5_data[op][key][
                'Bootstraps'])[:, 0]

    elif meson == 'll':
        central_val = np.array(h5_data[key]['central']).item()
        error = np.array(h5_data[key]['error']).item()
        bootstraps = np.array(h5_data[key]['Bootstraps'])[:, 0]

    if key == 'bag':
        central_val[1] *= -1
        bootstraps[:, 1] = -1*bootstraps[:, 1]
        central_val[-1] *= 2
        bootstraps[:, -1] = 2*bootstraps[:, -1]
    return stat(val=central_val, err='fill', btsp=bootstraps)


class Z_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP', 'TT']
    N_boot = N_boot

    def __init__(self, ensemble, action=(0, 0), bag=False, **kwargs):

        self.ens = ensemble
        self.action = action
        self.bag = bag
        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='fill')
        self.a_sq = stat(
            val=self.ainv.val**(-2),
            err='fill',
            btsp=self.ainv.btsp**(-2)
        )
        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][0])
        self.masses = (self.sea_m, self.sea_m)
        self.f_pi = stat(
            val=f_pi_PDG.val/self.ainv.val,
            btsp=f_pi_PDG.btsp/self.ainv.btsp)

        self.m_pi = load_info('m_0', self.ens, self.operators, meson='ll')
        self.m_f_sq = stat(
            val=(self.m_pi.val/self.f_pi.val)**2,
            err='fill',
            btsp=(self.m_pi.btsp**2)/(self.f_pi.btsp**2)
        )

        self.Z_obj = fourquark_analysis(
            ensemble, loadpath=f'RISMOM/{self.ens}.p')

        if self.action == (0, 1) or self.action == (1, 0):
            self.action == (0, 1)
            print('actions (0,1) and (1,0) have been averaged' +
                  ' into group called "(0,1)"')
            self.Z_obj.merge_mixed()

        momenta = self.Z_obj.momenta[self.action][self.masses]
        self.momenta = stat(
            val=momenta,
            err='fill',
            btsp=np.array([mom*self.ainv.val/self.ainv.btsp
                           for mom in momenta]).T
        )
        self.Z = stat(
            val=self.Z_obj.avg_results[self.action][self.masses],
            err=self.Z_obj.avg_errs[self.action][self.masses],
            btsp='fill'
        )
        self.N_mom = len(self.momenta.val)

        if self.bag:
            bl = bilinear_analysis(
                self.ens, loadpath=f'pickles/{self.ens}_bl.p')
            Z_bl = bl.avg_results[self.action][self.masses]
            Z_bl_err = bl.avg_errs[self.action][self.masses]
            for m in range(self.N_mom):
                z_a = stat(
                    val=Z_bl[m]['A'],
                    err=Z_bl_err[m]['A'],
                    btsp='fill')
                z_p = stat(
                    val=Z_bl[m]['P'],
                    err=Z_bl_err[m]['P'],
                    btsp='fill')
                mult = stat(
                    val=z_a.val/z_p.val,
                    btsp=z_a.btsp/z_p.btsp)

                self.Z.val[m, 1:, 1:] = self.Z.val[m, 1:, 1:]*(mult.val**2)
                self.Z.val[m, :, :] = mask*self.Z.val[m, :, :]

                for k in range(self.N_boot):
                    self.Z.btsp[k, m, 1:, 1:] = self.Z.btsp[
                        k, m, 1:, 1:]*(mult.btsp[k]**2)
                    self.Z.btsp[k, m, :, :] = mask*self.Z.btsp[k, m, :, :]

                self.Z.err[m, :, :] = np.array(
                    [[st_dev(self.Z.btsp[:, m, i, j], self.Z.val[m, i, j])
                      for j in range(5)]
                     for i in range(5)])

    def interpolate(self, mu, **kwargs):
        matrix = np.zeros(shape=(5, 5))
        btsp = np.zeros(shape=(N_boot, 5, 5))
        for i, j in itertools.product(range(5), range(5)):
            if mask[i, j]:
                f = interp1d(
                    self.momenta.val, self.Z.val[:, i, j], fill_value='extrapolate')
                matrix[i, j] = f(mu)

                for k in range(self.N_boot):
                    f = interp1d(
                        self.momenta.btsp[k,], self.Z.btsp[k, :, i, j],
                        fill_value='extrapolate')
                    btsp[k, i, j] = f(mu)
        Z = stat(
            val=matrix,
            err='fill',
            btsp=btsp
        )
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
            Z = stat(
                val=rot_mtx@Z.val@np.linalg.inv(rot_mtx),
                err='fill',
                btsp=np.array([rot_mtx@Z.btsp[k,]@np.linalg.inv(rot_mtx)
                               for k in range(N_boot)])
            )
        return Z

    def scale_evolve(self, mu2, mu1, **kwargs):
        Z1 = self.interpolate(mu1, **kwargs)
        Z2 = self.interpolate(mu2, **kwargs)
        sigma = stat(
            val=mask*Z2.val@np.linalg.inv(Z1.val),
            err='fill',
            btsp=np.array([mask*Z2.btsp[k,]@np.linalg.inv(Z1.btsp[k,])
                           for k in range(N_boot)]))
        return sigma

    def plot_Z(self, filename='plots/Z_scaling.pdf', **kwargs):
        x = self.momenta.val
        xerr = self.momenta.err
        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if mask[i, j]:
                y = self.Z.val[:, i, j]
                yerr = self.Z.err[:, i, j]
                ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr,
                                  fmt='o', capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis('off')
        if self.bag:
            plt.suptitle(
                r'$Z_{ij}^{'+self.ens+r'}/Z_{A/P}^2$ vs renormalisation scale $\mu$', y=0.9)
        else:
            plt.suptitle(
                r'$Z_{ij}^{'+self.ens+r'}/Z_A^2$ vs renormalisation scale $\mu$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)

    def plot_sigma(self, mu1=None, mu2=3.0, filename='plots/Z_running.pdf', **kwargs):
        x = self.momenta.val
        xerr = self.momenta.err

        if mu1 == None:
            sigmas = [self.scale_evolve(mu2, mom)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{'+self.ens + \
                r'}('+str(mu2)+r'\leftarrow\mu)$'
        else:
            sigmas = [self.scale_evolve(mom, mu1)
                      for mom in list(x)]
            sig_str = r'$\sigma_{ij}^{'+self.ens + \
                r'}(\mu\leftarrow '+str(mu1)+r')$'

        N_ops = len(operators)
        fig, ax = plt.subplots(nrows=N_ops, ncols=N_ops,
                               sharex='col',
                               figsize=(16, 16))
        plt.subplots_adjust(hspace=0, wspace=0)

        for i, j in itertools.product(range(N_ops), range(N_ops)):
            if mask[i, j]:
                y = [sig.val[i, j] for sig in sigmas]
                yerr = [sig.err[i, j] for sig in sigmas]

                ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr,
                                  fmt='o', capsize=4)
                if j == 2 or j == 4:
                    ax[i, j].yaxis.tick_right()
            else:
                ax[i, j].axis('off')
        if self.bag:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_{A/P}^2$', y=0.9)
        else:
            plt.suptitle(
                sig_str+r'for $Z_{ij}/Z_A^2$', y=0.9)

        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')

        print(f'Saved plot to {filename}.')
        os.system("open "+filename)

    def output_h5(self, add_mu=[2.0, 3.0], **kwargs):
        filename = 'Z_fq_bag.h5' if self.bag else 'Z_fq_a.h5'
        f = h5py.file(filename, 'r+')
        f_str = f'{self.action}/{self.ens}'

        momenta = self.momenta.val
        Z, Z_err = self.Z.val, self.Z.err

        for mu in add_mu:
            momenta = np.append(momenta, mu)
            Z_mu = self.interpolate(mu)
            Z = np.append(Z, np.resize(Z_mu.val, (1, 5, 5)), axis=0)
            Z_err = np.append(Z_err, np.resize(Z_mu.err, (1, 5, 5)), axis=0)

        f.create_dataset(f_str+'/momenta', data=momenta)
        f.create_dataset(f_str+'/Z', data=Z)
        f.create_dataset(f_str+'/Z_err', data=Z_err)


class bag_analysis:
    operators = operators
    N_boot = N_boot

    def __init__(self, ensemble, action=(0, 0), **kwargs):

        self.ens = ensemble
        self.action = action
        self.ainv = stat(
            val=params[self.ens]['ainv'],
            err=params[self.ens]['ainv_err'],
            btsp='fill')
        self.a_sq = stat(
            val=self.ainv.val**(-2),
            err='fill',
            btsp=self.ainv.btsp**(-2)
        )
        self.bag = load_info('bag', self.ens, self.operators, meson='ls')

        self.f_pi = stat(
            val=f_pi_PDG.val/self.ainv.val,
            err='fill',
            btsp=f_pi_PDG.btsp/self.ainv.btsp)

        self.m_pi = load_info('m_0', self.ens, self.operators, meson='ll')
        self.m_f_sq = stat(
            val=(self.m_pi.val/self.f_pi.val)**2,
            err='fill',
            btsp=(self.m_pi.btsp**2)/(self.f_pi.btsp**2)
        )

        self.Z_info = Z_analysis(self.ens, bag=True)

    def rotate_bag(self, rot_mtx, bag, **kwargs):
        bag_rot = stat(
            val=rot_mtx@bag.val,
            err='fill',
            btsp=[rot_mtx@bag.btsp[k,] for k in range(N_boot)]
        )
        return bag_rot

    def interpolate(self, mu, **kwargs):
        Z_mu = self.Z_info.interpolate(mu, **kwargs)
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
            bag = self.rotate_bag(rot_mtx, self.bag)
        else:
            bag = self.bag

        bag_interp = stat(
            val=Z_mu.val@bag.val,
            err='fill',
            btsp=[Z_mu.btsp[k, :, :]@bag.btsp[k, :]
                  for k in range(N_boot)])

        return bag_interp

    def ansatz(self, params, operator, fit='central', **kwargs):
        op_idx = self.operators.index(operator)
        if fit == 'central':
            a_sq = self.ainv.val**(-2)
            m_f_sq = self.m_f_sq.val
            PDG = m_f_sq_PDG.val
        else:
            k = kwargs['k']
            a_sq = self.ainv.btsp[k]**(-2)
            m_f_sq = self.m_f_sq.btsp[k]
            PDG = m_f_sq_PDG.btsp[k]

        def mpi_dep(m_f_sq):
            f = params[2]*m_f_sq
            if 'addnl_terms' in kwargs:
                if kwargs['addnl_terms'] == 'm4':
                    f += params[3]*(m_f_sq**2)
                elif kwargs['addnl_terms'] == 'log':
                    chir_log_coeffs = np.array([-0.5, -0.5, -0.5, 0.5, 0.5])
                    if 'rotate' in kwargs:
                        chir_log_coeffs = kwargs['rotate']@chir_log_coeffs
                    chir_log_coeffs = chir_log_coeffs/((4*np.pi)**2)
                    Lambda_QCD = 1.0
                    log_ratio = m_f_sq*(f_pi_PDG.val**2)/(Lambda_QCD**2)
                    log_term = chir_log_coeffs[op_idx]*np.log(log_ratio)
                    f += log_term*m_f_sq
            return f

        func = params[0] + params[1]*a_sq + \
            mpi_dep(m_f_sq)-mpi_dep(PDG)
        if 'addnl_terms' in kwargs:
            if kwargs['addnl_terms'] == 'a4':
                func += params[3]*(a_sq**2)

        return func
