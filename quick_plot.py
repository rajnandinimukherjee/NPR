from bag_param_renorm import *


ens_list = ['M0', 'M1', 'M2', 'M3']
# ens_list = ['C0', 'C1', 'C2']
Z_dict = {ens: Z_analysis(ens, bag=True) for ens in ens_list}
start, end = 3, -1

for i, j in itertools.product(range(len(operators)), range(len(operators))):
    if mask[i, j]:
        fig, ax = plt.subplots(nrows=1, ncols=2,
                               sharey=True,
                               figsize=(8, 3))
        plt.subplots_adjust(wspace=0)
        for ens in ens_list:
            Z_obj = Z_dict[ens]
            Z, amom, ainv = Z_obj.Z, Z_obj.am, Z_obj.ainv

            momenta = amom*ainv
            x = momenta.val[start:end]
            xerr = momenta.err[start:end]
            Z_2_pred = Z_obj.interpolate(2.0)
            Z_25_pred = Z_obj.interpolate(2.5)
            y = Z.val[:, i, j][start:end]
            yerr = Z.err[:, i, j][start:end]
            ax[0].errorbar(x, y, yerr=yerr, xerr=xerr,
                           fmt='o:', capsize=4, label=ens)
            # ax[0].errorbar([2.0], Z_2_pred.val[i, j], yerr=Z_2_pred.err[i, j],
            #             fmt='o', capsize=4, color='k')
            # ax[0].errorbar([2.5], Z_25_pred.val[i, j], yerr=Z_25_pred.err[i, j],
            #             fmt='o', capsize=4, color='k')

            x = amom.val[start:end]
            xerr = amom.err[start:end]
            ax[1].errorbar(x, y, yerr=yerr, xerr=xerr,
                           fmt='o:', capsize=4, label=ens)
        ax[0].set_ylabel(r'$Z_{'+str(i+1)+str(j+1)+'}^{'+','.join(ens_list) +
                         r'}/Z_A^2$')
        ax[0].set_xlabel(r'$\mu$ (GeV)')
        ax[1].set_xlabel(r'$a\mu$')
        ax[1].legend()

filename = 'plots/quickplot_'+'_'.join(ens_list)+'.pdf'
pp = PdfPages(filename)
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')

print(f'Saved plot to {filename}.')
os.system("open "+filename)
