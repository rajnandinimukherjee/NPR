from cont_chir_extrap import * 

ens = 'C1M'
compare_ens = 'C0'
num_masses = 5
masses = params[ens]['masses'][:num_masses]

Z_dict = {masses[i]:Z_analysis(ens, norm='bag', sea_mass_idx=i)
          for i in range(num_masses)}
comp_Z = Z_analysis(compare_ens, norm='bag')

z = Z_fits(CM, norm='bag')

momenta = set(list(Z_dict[masses[1]].am.val))
for idx in range(num_masses):
    momenta.intersection_update(list(Z_dict[masses[idx]].am.val))

momenta = sorted(list(momenta))
for mom in momenta:
    fig, ax = plt.subplots(5,5,figsize=(16,16))
    plt.subplots_adjust(hspace=0, wspace=0)
    x = masses

    extrap, fit_params = z.Z_chiral_extrap(CM, comp_Z.ainv.val*mom)
    for i,j in itertools.product(range(5), range(5)):
        if fq_mask[i,j]:
            y = join_stats([stat(
                val=Z_dict[m].Z.val[list(Z_dict[m].am.val).index(mom),i,j],
                err=Z_dict[m].Z.err[list(Z_dict[m].am.val).index(mom),i,j],
                btsp=Z_dict[m].Z.btsp[:,list(Z_dict[m].am.val).index(mom),i,j]
                ) for m in masses])

            y_comp = stat(
                    val=comp_Z.Z.val[list(comp_Z.am.val).index(mom),i,j],
                    err=comp_Z.Z.err[list(comp_Z.am.val).index(mom),i,j],
                    btsp=comp_Z.Z.btsp[:,list(comp_Z.am.val).index(mom),i,j]
                    )
            ax[i,j].errorbar(x, y.val, yerr=y.err,
                        fmt='o', capsize=4, label='C1M')
            ax[i,j].errorbar([float(comp_Z.sea_m)], y_comp.val, yerr=y_comp.err,
                             fmt='o', capsize=4, label='C0')
            xmin, xmax = ax[i,j].get_xlim()
            ax[i,j].errorbar([xmax], extrap.val[i,j], yerr=extrap.err[i,j],
                             fmt='o', capsize=4, c='k', label='extrap')
            #ax[i,j].axhline(extrap.val[i,j], linestyle='dashed', c='k', label='extrap')
            #ax[i,j].axhspan(extrap.val[i,j]+extrap.err[i,j],
            #                extrap.val[i,j]-extrap.err[i,j],
            #                color='k', alpha=0.1)
            #ax[i,j].set_xlim((xmin,xmax))
            ax[i,j].legend()
            if j == 2 or j == 4:
                ax[i, j].yaxis.tick_right()

            if i == 1 or i == 3:
                ax[i, j].set_xticks([])
        else:
            ax[i, j].axis('off')
    plt.suptitle(f'{ens} variation with bare mass at ap={mom}', y=0.9)

filename = 'fq_mass_variation.pdf'
call_PDF(filename)
