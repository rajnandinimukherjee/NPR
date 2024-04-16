from basics import *
from eta_c import *


class valence:
    vpath = 'valence/'
    filename = 'eta_C.h5'
    eta_h_gamma = ('Gamma5', 'Gamma5')

    def __init__(self, ens, **kwargs):
        self.ens = ens
        self.info = params[self.ens]
        self.all_masses = [str(m) for m in info['masses']]
        self.sea_mass = self.all_masses[0]
        self.N_mass = len(self.all_masses)
        self.mass_comb = {self.all_masses[0]: ('l', 'l'),
                          self.all_masses[1]: ('l_double', 'l_double'),
                          self.all_masses[2]: ('s_half', 's_half'),
                          self.all_masses[3]: ('s', 's')}
        self.mass_comb.update(
            {self.all_masses[m]: (f'c{m-1}', f'c{m-1}')
             for m in range(4, self.N_mass)})
        self.mass_dict = {m: {} for m in self.all_masses}
        self.fit_dict = {m: {} for m in self.all_masses}

    def meson_correlator(self, mass, cfgs=None, meson_num=1, **kwargs):
        self.cfgs = info['valence_cfgs'] if cfgs==None else cfgs
        t_src_range = np.arange(0,64,4)





    def load_data(self, mass, key='corr', plot=False,
                  **kwargs):
        file = h5py.File(self.filename, 'r')
        data = np.array(file[self.ens][str(
            self.mass_comb[mass])][key][:])

        if data.shape[0] == 1:
            data = stat(
                val=data[0, :],
                err='fill',
                btsp=np.array(file[self.ens+'_expanded'][str(
                    self.mass_comb[mass])][key][:])
            )
        else:
            data = stat(
                val=np.mean(data, axis=0),
                err='fill',
                btsp=bootstrap(data, K=N_boot)
            )

        if plot:
            plt.figure()
            plt.errorbar(range(len(data.val)), data.val,
                         yerr=data.err, capsize=4, fmt='o')
            plt.xlabel(r'at')
            filename = f'plots/valence_data_plot_{self.ens}.pdf'
            call_PDF(filename)

        return data

    def fit_data(self, mass, key='corr', plot=False,
                 start=12, end=25, pass_plot=False,
                 update=True, **kwargs):
        data = self.load_data(mass, key=key, **kwargs)
        if key == 'PJ5q':
            denom = self.load_data(mass, key='corr')
            data = data/denom
        if key == 'corr':
            data = data.use_func(np.log)

        T = len(data.val)
        T_half = int(T/2)

        def fold(array):
            return 0.5*(array[1:]+array[::-1][:-1])[:T_half]
        folded_data = data.use_func(fold)

        x = stat(np.arange(T_half), btsp='fill')
        y = folded_data

        if key == 'corr':
            guess = [1, 1]

            def ansatz(t, param, **kwargs):
                return param[1] - param[0]*t

        elif key == 'PJ5q':
            guess = [0.1, 0]

            def ansatz(t, param, **kwargs):
                return param[0]*np.ones(len(t))

        res = fit_func(x, y, ansatz, guess,
                       start=start, end=end, **kwargs)
        fit_mass = stat(
            val=res.val[0],
            err=res.err[0],
            btsp=res.btsp[:, 0]
        )

        if update:
            if res.pvalue > 0.05:
                label = 'aM_eta_h' if key == 'corr' else 'am_res'
                self.mass_dict[mass][label] = fit_mass
                self.fit_dict[mass][label] = [start, end]

        if pass_plot:
            plt.figure()
            if key == 'corr':
                y = y.use_func(np.exp)
                y = y.use_func(m_eff, ansatz='exp')
                x.val = x.val[:-1]
                plt.ylabel(r'$aM_{\eta_h}^{eff}$')
            else:
                plt.ylabel(r'$am_{res}$')
            plt.errorbar(x.val[10:], y.val[10:],
                         yerr=y.err[10:],
                         capsize=4, fmt='o')
            plt.fill_between(np.arange(start, end),
                             fit_mass.val+fit_mass.err,
                             fit_mass.val-fit_mass.err,
                             color='0.5')
            plt.text(0.5, 0.08, r'$\chi^2$/DOF:' +
                     str(np.around(res.chi_sq/res.DOF, 3)),
                     va='center', ha='center',
                     transform=plt.gca().transAxes)
            plt.xlabel('at')
            if plot:
                filename = f'plots/valence_data_fit_{self.ens}.pdf'
                call_PDF(filename)

            return res, fit_mass, plt
        else:
            return res, fit_mass

    def fit_all(self, key, plot=False, save=True, **kwargs):
        for mass in self.all_masses:
            res, fit_mass, plt = self.fit_data(
                mass, key=key, pass_plot=True, **kwargs)
            plt.title(r'$am_{in}=$'+str(mass))

        if save:
            self.save_to_H5()

        if plot:
            filename = f'plots/all_fits_{self.ens}.pdf'
            call_PDF(filename)

    def plot_from_dict(self, key, **kwargs):
        label = 'aM_eta_h' if key == 'corr' else 'am_res'
        for mass in self.all_masses:
            if label in self.mass_dict[mass]:
                start, end = self.fit_dict[mass][label]
                res, fit, plt = self.fit_data(
                    mass, key, pass_plot=True,
                    start=start, end=end)
                plt.title(r'$am_{in}=$'+str(mass))

        filename = f'plots/all_fits_{self.ens}.pdf'
        call_PDF(filename)

    def save_to_H5(self, filename='eta_fits.h5'):
        with h5py.File(filename, 'a') as file:
            if self.ens in file.keys():
                del file[self.ens]

            ens = file.create_group(self.ens)
            for mass in self.all_masses:
                for key in ['aM_eta_h', 'am_res']:
                    if key in self.mass_dict[mass].keys():
                        key_group = ens.create_group(mass+'/'+key)

                        obj = self.mass_dict[mass][key]
                        key_group.create_dataset('central', data=obj.val)
                        key_group.create_dataset('errors', data=obj.err)
                        key_group.create_dataset('bootstrap', data=obj.btsp)

                        start, end = self.fit_dict[mass][key]
                        key_group.attrs['fit_start'] = start
                        key_group.attrs['fit_end'] = end
            print(f'Added fit data to {self.ens} in {filename}.')

    def load_from_H5(self, filename='eta_fits.h5'):
        file = h5py.File(filename, 'r')
        if self.ens not in file.keys():
            print(f'Data for {self.ens} not found in {filename}.')
        else:
            for mass in file[self.ens].keys():
                for key in file[self.ens][mass]:
                    folder = file[self.ens][mass][key]

                    obj = stat(
                        val=np.array(folder['central']),
                        err=np.array(folder['errors']),
                        btsp=np.array(folder['bootstrap'][:])
                    )
                    self.mass_dict[mass][key] = obj

                    start = file[self.ens][mass][key].attrs['fit_start']
                    end = file[self.ens][mass][key].attrs['fit_end']
                    self.fit_dict[mass][key] = [start, end]
