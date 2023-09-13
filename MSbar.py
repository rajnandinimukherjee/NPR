from cont_chir_extrap import *
fits = pickle.load(open('cc_extrap_dict.p', 'rb'))


ansatze = list(fits[operator].keys())
ansatz_desc = [fits[operator][a]['kwargs']['title']
               for a in ansatze]
vals = [fits[operator][fit][mu]['phys'] for fit in ansatze]
errs = [fits[operator][fit][mu]['err'] for fit in ansatze]
chis = [fits[operator][fit][mu]['chi_sq_dof'] for fit in ansatze]
cont_slope = [fits[operator][fit][mu]['coeffs'][0] for fit in ansatze]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                       sharey=False)
plt.subplots_adjust(wspace=0)

ax[0].set_title(r'$B_{phys}^{SMOM}(\mu='+str(np.around(mu, 2))+'$ GeV)')
ax[0].errorbar(np.arange(len(ansatze)), vals,
               yerr=errs, fmt='o', capsize=2)
ax[0].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                 ha='right')
# ax[0].set_ylabel(r'$B(a=0, m_\pi=m_\pi^{phys})$ in RI/SMOM')
tick = min(errs)*0.1
for i in range(len(ansatze)):
    ax[0].annotate(str(np.around(chis[i], 2)),
                   (i, vals[i]+errs[i]+tick),
                   ha='center', va='bottom')
    ax[0].annotate(str(int(100*np.around(cont_slope[i], 2)))+r'%',
                   (i, vals[i]-errs[i]-4*tick), fontsize=9,
                   ha='center', va='top')
    if chis[i] > 2.0:
        ax[0].annotate('X', (i, vals[i]), ha='center', va='center')

MS_vals = [fits[operator][fit][mu]['MS']['val'] for fit in ansatze]
MS_errs = [fits[operator][fit][mu]['MS']['err'] for fit in ansatze]
ax[1].errorbar(np.arange(len(ansatze)), MS_vals, yerr=MS_errs,
               fmt='o', capsize=2, c='r')

ansatz_desc.append(r'FLAG $N_f=2+1+1$')
op_idx = operators.index(operator)
FLAG_val, FLAG_err = flag_vals[op_idx], flag_errs[op_idx]
ax[1].errorbar([len(ansatze)], [FLAG_val], yerr=[FLAG_err],
               fmt='o', capsize=2, c='k')

ax[1].set_xticks(np.arange(len(ansatze)+1), ansatz_desc, rotation=45,
                 ha='right')
ax[1].set_title(r'$B_{phys}^{\overline{MS}}(\mu='+str(np.around(mu, 2)) +
                r'\to 3.0$ GeV)')

filename = f'plots/MS_SUSY.pdf'

pp = PdfPages(filename)
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')

print(f'Plot saved to {filename}.')
os.system("open "+filename)
