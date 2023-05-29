from NPR_classes import *

ens_raj=['F1M','KEKC2a','KEKC2b','KEKM1a','KEKM1b','KEKF1','KEKC1S','KEKC1L',
        'C0','C1','C2','M0','M1','M2','M3']
labels=['a2.7m230','a2.5m310-b','a2.5m310-a','a3.6m300-b','a3.6m300-a','a4.5m280',
        'a2.5m230-S','a2.5m230-L','a1.7m140','a1.8m340','a1.8m430','a2.4m140',
        'a2.4m300','a2.4m360','a2.4m410']

fq_dict = {ens:{'fq_obj':fourquark_analysis(ens, loadpath=f'RISMOM/{ens}.p'),
                'action':(0,1) if ens in UKQCD_ens else (0,0)}
           for ens in ens_raj}

fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(12,8))
for op1, op2 in itertools.product(range(5), range(5)):
    ax[0,op2].set_title(operators[op2],fontsize=10)
    ax[op1,0].set_ylabel(operators[op1],fontsize=10)
    
    if mask[op1, op2]:
        if op1==1 or op1==3:
            ax[op1,op2].tick_params('x',labelbottom=False)
        else:
            ax[op1,op2].set_xlabel(r'$\mu$ (GeV)')
            ax[op1,op2].xaxis.set_label_coords(0.1, -0.075)
        for ens in ens_raj:
            fq = fq_dict[ens]['fq_obj']
            action = fq_dict[ens]['action']
            masses = list(fq.momenta[action].keys())[0]
            x = fq.momenta[action][masses]
            y = [mtx[op1,op2] for mtx in fq.avg_results[action][masses]]
            e = [mtx[op1,op2] for mtx in fq.avg_errs[action][masses]]

            label = labels[ens_raj.index(ens)]
            ax[op1,op2].errorbar(x,y,yerr=e,fmt='o',capsize=1,
                                 #markerfacecolor='None',
                                 markersize=2,label=label)
    else:
        ax[op1,op2].xaxis.set_visible(False)
        plt.setp(ax[op1,op2].spines.values(), visible=False)
        ax[op1,op2].tick_params(left=False, labelleft=False)
        
fig.align_ylabels(ax[:,0])
fig.tight_layout()
fig.legend(labels, loc='upper right', bbox_to_anchor=(0.98,0.65),
           ncol=int(len(labels)/4), bbox_transform=fig.transFigure,
           title='Ensembles')
fig.subplots_adjust(hspace=0.2,wspace=0.2)
plt.suptitle(r'$Z_{ij}^{(\gamma,\gamma)}(\mu)/Z_A^2$'+'\n for $B-\overline{B}$ mixing operators', x=0.5, y=0.07)

plt.figure(figsize=(8,6))
for ens in ens_raj:
    fq = fq_dict[ens]['fq_obj']
    action = fq_dict[ens]['action']
    masses = list(fq.momenta[action].keys())[0]
    x = fq.momenta[action][masses]
    y = [mtx[0,0] for mtx in fq.avg_results[action][masses]]
    e = [mtx[0,0] for mtx in fq.avg_errs[action][masses]]

    m = 'o' if ens in UKQCD_ens else 'D'
    label = labels[ens_raj.index(ens)]
    plt.errorbar(x,y,yerr=e,fmt='o',marker=m,capsize=6,label=label)
plt.ylabel(r'$Z_{VV+AA}^{(\gamma_\mu,\gamma_\mu)}/Z_A^2$')
plt.xlabel(r'$\mu$ (GeV)')
plt.legend(bbox_to_anchor=(1.2,1),title='Ensembles')

filename = 'plots/Z_fq.pdf'
pp = PdfPages(filename)
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')
os.system('open '+filename)


