from cont_chir_extrap import *


class ens_table:
    N_ops = len(operators)

    def __init__(self, ens, norm='V', sea_mass_idx=0, 
                 action=(0,0), scheme='gamma', **kwargs):
        self.ens = ens
        self.action = action
        self.scheme = scheme
        self.norm = norm
        self.Z_obj = Z_analysis(self.ens, norm=norm, action=action,
                                sea_mass_idx=sea_mass_idx, 
                                scheme=scheme, **kwargs)

        self.am = self.Z_obj.am.val
        self.Z = self.Z_obj.Z
        self.N_mom = len(self.am)
        self.N_mom_half = int(self.N_mom/2)

        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][sea_mass_idx])
        self.masses = (self.sea_m, self.sea_m)
        a1, a2 = self.action
        datafile = f'NPR/action{a1}_action{a2}/'
        datafile += '__'.join(['NPR', self.ens,
                               params[self.ens]['baseactions'][a1],
                               params[self.ens]['baseactions'][a2],
                               self.scheme])
        datafile += '.h5'
        bl_data = h5py.File(datafile, 'r')[str(self.masses)]['bilinear']
        self.Z_A = stat(
            val=bl_data['A']['central'][:],
            err=bl_data['A']['errors'][:],
            btsp=bl_data['A']['bootstrap'][:]
        )
        self.Z_S = stat(
            val=bl_data['S']['central'][:],
            err=bl_data['S']['errors'][:],
            btsp=bl_data['S']['bootstrap'][:]
        )
        self.ratio = self.Z_A/self.Z_S

    def create_Z_table(self, indices=None, filename=None, **kwargs):
        if 'rotate' in kwargs:
            rot_mtx = kwargs['rotate']
            self.Z = stat(
                    val=[rot_mtx@self.Z.val[m]@np.linalg.inv(rot_mtx)
                         for m in range(self.N_mom)],
                    err='fill',
                    btsp=[[rot_mtx@self.Z.btsp[k,m]@np.linalg.inv(rot_mtx)
                           for m in range(self.N_mom)]
                          for k in range(N_boot)]
                    )
        rv = [r'\begin{tabular}{c|'+' '.join(['c']*self.N_mom_half)+r'}']
        rv += [r'\hline']
        rv += [r'\hline']

        if indices == None:
            for row in range(2):
                start = row*self.N_mom_half
                end = (row+1)*self.N_mom_half
                rv += [r'$a\mu$ & $'+'$ & $'.join(
                    [str(np.around(m, 5)) for m in list(self.am[start:end])])+r'$ \\']
                rv += [r'\hline']
                for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                    if mask[i, j]:
                        Z_disp = [err_disp(self.Z.val[m, i, j], self.Z.err[m, i, j])
                                  for m in range(start, end)]
                        rv += [r'$Z_{'+str(i+1)+str(j+1)+r'}/Z_A^2$ & $' +
                               r'$ & $'.join(Z_disp)+r'$ \\']
                rv += [r'\hline']
                ratio_disp = [err_disp(self.ratio.val[m], self.ratio.err[m])
                              for m in range(start, end)]
                rv += [r'$Z_A/Z_S$ & $' +
                       r'$ & $'.join(ratio_disp)+r'$ \\']
                rv += [r'\hline']
        else:
            momenta = r' & '.join(
                [str(np.around(self.am[m], 5)) for m in indices])
            rv += [r'$a\mu$ & '+momenta+r' \\']
            rv += [r'\hline']
            for i, j in itertools.product(range(self.N_ops), range(self.N_ops)):
                if self.Z_obj.mask[i, j]:
                    Z_disp = [err_disp(self.Z.val[m, i, j], self.Z.err[m, i, j])
                              for m in indices]
                    rv += [r'$Z_{'+str(i+1)+str(j+1)+r'}/Z_A^2$ & $' +
                           r'$ & $'.join(Z_disp)+r'$ \\']
                    if i == j:
                        if i == 0 or i == 2 or i == 4:
                            rv += [r'\hline']
            ratio_disp = [err_disp(self.ratio.val[m], self.ratio.err[m])
                          for m in indices]
            rv += [r'$Z_A/Z_S$ & $' +
                   r'$ & $'.join(ratio_disp)+r'$ \\']
            rv += [r'\hline']

        rv += [r'\hline']
        rv += [r'\end{tabular}']

        if filename==None:
            filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/{self.ens}_Z_table.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z table output written to {filename}.')


class extrap_table:
    ens_list = ['C0', 'C1', 'M0', 'M1', 'F1M']

    def __init__(self, norm='V', **kwargs):

        self.norm = norm
        self.Z_fit = Z_fits(self.ens_list, norm=self.norm, **kwargs)
        self.sig = sigma(norm=self.norm, **kwargs)

    def create_Z_table(self, mu, filename=None, **kwargs):
        Z_assign = self.Z_fit.Z_assignment(mu, chiral_extrap=True, **kwargs)

        rv = [r'\begin{tabular}{c|ccccc}']
        rv += [r'\hline']
        rv += [r'\hline']

        ainvs = r' & '.join([err_disp(
            params[ens]['ainv'], params[ens]['ainv_err'])
            for ens in self.ens_list])

        rv += [r'$a^{-1}$ [GeV] & '+ainvs+r' \\']
        rv += [r'\hline']
        for i, j in itertools.product(range(5), range(5)):
            if self.Z_fit.mask[i, j]:
                Zs = r' & '.join([err_disp(
                    Z_assign[e].val[i, j], Z_assign[e].stat_err[i, j],
                    n=2, sys_err=Z_assign[e].sys_err[i,j])
                    for e in self.ens_list])
                if self.norm=='bag':
                    norm = r'Z_A^2' if j==0 else r'Z_S^2'
                else:
                    norm = f'Z_{self.norm}^2'
                Z_name = r'$Z_{'+str(i+1)+str(j+1)+r'}/'+norm+r'$'
                rv += [Z_name+r' & '+Zs+r' \\']
                if i == j:
                    if i != 1 and i != 3:
                        rv += [r'\hline']

        rv += [r'\hline']
        rv += [r'\end{tabular}']

        if filename==None:
            filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/'
            filename += f'tables_{fit_file}/table_extrap_Z_table_{str(int(10*mu))}.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z table output written to {filename}.')

    def create_sig_table(self, mu1, mu2, filename=None, **kwargs):
        Z_dict_mu1 = self.sig.Z_fits.Z_assignment(mu1, **kwargs)
        Z_dict_mu2 = self.sig.Z_fits.Z_assignment(mu2, **kwargs)
        sigmas = {e:stat(
            val=Z_dict_mu2[e].val@np.linalg.inv(Z_dict_mu1[e].val),
            err='fill',
            btsp=np.array([Z_dict_mu2[e].btsp[k]@np.linalg.inv(
                Z_dict_mu1[e].btsp[k]) for k in range(N_boot)])
        ) for e in self.sig.relevant_ensembles}

        rv = [r'\begin{tabular}{c|ccccc}']
        rv += [r'\hline']
        rv += [r'\hline']

        ainvs = r' & '.join([err_disp(
            params[ens]['ainv'], params[ens]['ainv_err'])
            for ens in self.sig.relevant_ensembles])

        rv += [r'$a^{-1}$ [GeV] & '+ainvs+r' \\']
        rv += [r'\hline']
        for i, j in itertools.product(range(5), range(5)):
            if self.Z_fit.mask[i, j]:
                s_vals = r' & '.join([err_disp(
                    sigmas[e].val[i, j], sigmas[e].err[i, j])
                    for e in self.sig.relevant_ensembles])
                s_name = r'$\sigma_{'+str(i+1)+str(j+1)+r'}$'
                rv += [s_name+r' & '+s_vals+r' \\']
                if i == j:
                    if i != 1 and i != 3:
                        rv += [r'\hline']

        rv += [r'\hline']
        rv += [r'\end{tabular}']

        if filename==None:
            filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/'
            filename += f'tables_{fit_file}/sigma_table_{int(mu1*10)}_{int(mu2*10)}.tex'

        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'sigma table output written to {filename}.')

    def print_sigmas(self, mu1, mu2, stepsize=1.0, **kwargs):
        mus = np.arange(mu1, mu2+0.1, stepsize)
        sig = stat(val=np.eye(5), btsp='fill')
        for m in range(1,len(mus)):
            sig = self.sig.calc_running(mus[m-1], mus[m], 
                                        chi_sq_rescale=True, 
                                        **kwargs)@sig

        rv = [r'\begin{bmatrix}']
        for j in range(5):
            rv += [' & '.join([err_disp(sig.val[i,j], sig.err[i,j])
                               for i in range(5)])+r' \\']

        rv += [r'\end{bmatrix}']
        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/'+\
                f'tables_{fit_file}/sigma_matrix_{self.norm}_{np.around(stepsize,2)}.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'sigma table output written to {filename}.')

