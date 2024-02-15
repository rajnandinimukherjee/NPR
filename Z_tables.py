from cont_chir_extrap import *


class ens_table:
    N_ops = len(operators)

    def __init__(self, ens, norm='V'):
        self.ens = ens
        self.Z_obj = Z_analysis(self.ens, norm=norm)

        self.am = self.Z_obj.am.val
        self.Z = self.Z_obj.Z
        self.N_mom = len(self.am)
        self.N_mom_half = int(self.N_mom/2)

        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][0])
        self.masses = (self.sea_m, self.sea_m)
        bl_data = h5py.File('bilinear_Z_gamma.h5', 'r')[
            str((0, 0))][self.ens][str(self.masses)]
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

    def create_Z_table(self, indices=None):
        # table_type = 'table' if self.ens != 'F1M' else 'sidewaystable'
        table_type = 'table'
        rv = [r'\begin{'+table_type+'}']
        rv += [r'\begin{center}']
        rv += [r'\caption{\label{tab:'+self.ens+'renormvals}'+self.ens +
               r': values of $Z_{ij}/Z_A^2$ and $Z_A/Z_S$ at various lattice momenta}']
        rv += [r'\begin{tabular}{c|'+' '.join(['c']*self.N_mom_half)+r'}']
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
                if mask[i, j]:
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
        rv += [r'\end{center}']
        rv += [r'\end{'+table_type+'}']

        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables/{self.ens}_Z_table.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z table output written to {filename}.')


class extrap_table:
    ens_list = ['C0', 'C1', 'M0', 'M1', 'F1M']

    def __init__(self, norm='V', **kwargs):

        self.norm = norm
        self.Z_fit = Z_fits(self.ens_list, norm=self.norm)
        self.sig = sigma(norm=self.norm)

    def create_Z_table(self, mu, **kwargs):
        Z_assign = self.Z_fit.Z_assignment(mu, chiral_extrap=True)

        rv = [r'\begin{table}']
        rv += [r'\caption{Elements of $Z_{ij}^{RI}(\mu={' +\
               str(mu)+r'}\,\mathrm{GeV})/Z_{A/S}^2$' +\
               r' extrapolated to the massless limit. The first parenthesis '+\
               r'is the statistical error and the second is the systematic error. '+\
               r'The systematic error on the finest ensemble also includes the '+\
               r'spread in chiral extrapolations using the chiral slopes from '+\
               r'the four other extrapolations.'+\
               r'\label{tab:ch-extrap-'+str(mu)+'}}']
        rv += [r'\begin{tabular}{c|ccccc}']
        rv += [r'\hline']
        rv += [r'\hline']

        ainvs = r' & '.join([err_disp(
            params[ens]['ainv'], params[ens]['ainv_err'])
            for ens in self.ens_list])

        rv += [r'$a^{-1}$ [GeV] & '+ainvs+r' \\']
        rv += [r'\hline']
        for i, j in itertools.product(range(5), range(5)):
            if mask[i, j]:
                Zs = r' & '.join([err_disp(
                    Z_assign[e].val[i, j], Z_assign[e].stat_err[i, j],
                    n=2, sys_err=Z_assign[e].sys_err[i,j])
                    for e in self.ens_list])
                norm = r'Z_A^2' if i==0 else r'Z_S^2'
                Z_name = r'$Z_{'+str(i+1)+str(j+1)+r'}/'+norm+r'$'
                rv += [Z_name+r' & '+Zs+r' \\']
                if i == j:
                    if i != 1 and i != 3:
                        rv += [r'\hline']

        # Z_names = r' & '.join([Z for Z in [r'$Z_{'+str(i+1) +
        #                                   str(j+1)+'}$' for j in range(5)
        #                                   for i in range(5) if mask[i, j]]])
        # rv += [r'$a^{-1}\,[\mathrm{GeV}]$ & $\mu\,[\mathrm{GeV}]$ &'+Z_names+r' \\']
        # for ens in self.ens_list:
        #    ainv = err_disp(params[ens]['ainv'], params[ens]['ainv_err'])
        #    Zs = r' & '.join([Z for Z in [err_disp(
        #        Z_assign[ens].val[i, j], Z_assign[ens].err[i, j])
        #        for j in range(5) for i in range(5) if mask[i, j]]])
        #    rv += [ainv+r' & '+str(mu)+' & '+Zs+r' \\']
        rv += [r'\hline']
        rv += [r'\end{tabular}']
        rv += [r'\end{table}']

        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables/extrap_Z_table_{str(int(10*mu))}.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z table output written to {filename}.')

    #def create_sig_table(self, mu1, mu2, **kwargs):
