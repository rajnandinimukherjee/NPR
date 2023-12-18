from bag_param_renorm import *


class ens_table:
    N_ops = len(operators)

    def __init__(self, ens, bag=False):
        self.ens = ens
        self.Z_obj = Z_analysis(self.ens, bag=bag)

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

    def create_Z_table(self):
        table_type = 'table' if self.ens != 'F1M' else 'sidewaystable'
        rv = [r'\begin{'+table_type+'}']
        rv += [r'\begin{center}']
        rv += [r'\caption{'+self.ens +
               r': values of $Z_{ij}/Z_A^2$ and $Z_A/Z_S$ at various lattice momenta}']
        rv += [r'\begin{tabular}{c|'+' '.join(['c']*self.N_mom_half)+r'}']
        rv += [r'\hline']
        rv += [r'\hline']
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
                    rv += [r'$'+str(i+1)+str(j+1)+r'$ & $' +
                           r'$ & $'.join(Z_disp)+r'$ \\']
            rv += [r'\hline']
            ratio_disp = [err_disp(self.ratio.val[m], self.ratio.err[m])
                          for m in range(start, end)]
            rv += [r'A/S & $' +
                   r'$ & $'.join(ratio_disp)+r'$ \\']
            rv += [r'\hline']
        rv += [r'\hline']
        rv += [r'\end{tabular}']
        rv += [r'\end{center}']
        rv += [r'\end{'+table_type+'}']

        filename = f'tex/{self.ens}_Z_table.tex'
        f = open(filename, 'w')
        f.write('\n'.join(rv))
        f.close()
        print(f'Z table output written to {filename}.')
