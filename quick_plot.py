from bag_param_renorm import *

fig, ax = plt.subplots(nrows=len(operators),
                       ncols=len(operators),
                       sharex='col',
                       figsize=(16, 16))
plt.subplots_adjust(hspace=0, wspace=0)

ens_list = ['M0', 'M1', 'M2', 'M3']
# ens_list = ['C0', 'C1', 'C2']

for ens in ens_list:
    Z_obj = Z_analysis(ens, bag=False)
    Z = Z_obj.Z
    momenta = Z_obj.momenta
    ainv = stat(
        val=params[ens]['ainv'],
        err=params[ens]['ainv_err'],
        btsp='fill'
    )
    a = stat(
        val=1/ainv.val,
        err='fill',
        btsp=1/ainv.btsp
    )
    momenta = momenta*a

    x = momenta.val[5:]
    xerr = momenta.err[5:]
    for i, j in itertools.product(range(len(operators)), range(len(operators))):
        if mask[i, j]:
            y = Z.val[:, i, j][5:]
            yerr = Z.err[:, i, j][5:]
            ax[i, j].errorbar(x, y, yerr=yerr, xerr=xerr,
                              fmt='o:', capsize=4, label=ens)
            if j == 2 or j == 4:
                ax[i, j].yaxis.tick_right()
        else:
            ax[i, j].axis('off')

plt.suptitle(
    r'$Z_{ij}^{'+','.join(ens_list)+r'}/Z_A^2$ vs renormalisation scale $\mu$', y=0.9)
plt.legend()

filename = 'plots/quickplot.pdf'
pp = PdfPages(filename)
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')

print(f'Saved plot to {filename}.')
os.system("open "+filename)
