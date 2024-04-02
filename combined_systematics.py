from cont_chir_extrap import *

errors_dict = {'gamma':{'name':r'$\textrm{RI-SMOM}^{(\gamma_\mu,\gamma_\mu)}$'},
               'qslash':{'name':r'$\textrm{RI-SMOM}^{(\slashed{q}, \slashed{q})}$'}}

quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})

RISMOM_results = {'gamma':{}, 'qslash':{}}

expand_str = '_expanded' if expand_err else ''
for scheme in errors_dict.keys():
    fit_systematics = pickle.load(open(f'fit_systematics_20_{fit_file}_{scheme}{expand_str}.p', 'rb'))
    scaling_systematics = pickle.load(open(f'scaling_systematics_{scheme}_{fit_file}{expand_str}.p', 'rb'))
    other_systematics = pickle.load(open(f'other_systematics_{scheme}_{fit_file}{expand_str}.p', 'rb'))
    for key in quantities.keys():
        central = scaling_systematics['(2,3)'][key]
        central_perc_err = np.abs(central.err/central.val)*100
        stat_err_str = '{0:.2f}'.format(central_perc_err)+r'\%'

        central_2 = fit_systematics['central'][key][0]
        chiral_fits = [fit_systematics[fittype][key][0] for fittype in ['no_del_ms', 'C2M3cut','C2M3M2cut','no_log']]
        delta_chirals = [np.abs(((chiral_fit-central_2)/((chiral_fit+central_2)*0.5)).val*100)
                         for chiral_fit in chiral_fits]
        delta_chiral = max(delta_chirals)
        chiral_err_str = '{0:.2f}'.format(delta_chiral)+r'\%'

        rcsb = other_systematics['no_mask'][key][3.0]
        delta_rcsb = np.abs(((rcsb-central)/((rcsb+central)*0.5)).val*100)
        rcsb_err_str = '{0:.2f}'.format(delta_rcsb)+r'\%'

        discr_fits = [scaling_systematics[fittype][key] for fittype in ['(3)', '(2,3,0.5)', '(2,3,0.33)']]
        delta_discrs = [np.abs(((discr_fit-central)/((discr_fit+central)*0.5)).val*100)
                        for discr_fit in discr_fits]
        delta_discr = max(delta_discrs)
        discr_err_str = '{0:.2f}'.format(delta_discr)+r'\%'

        NPR = other_systematics['NPR'][key][3.0]
        delta_NPR = np.abs(((NPR-central)/((NPR+central)*0.5)).val*100)
        NPR_err_str = '{0:.2f}'.format(delta_NPR)+r'\%'

        total_err = (central_perc_err**2+delta_chiral**2+\
                delta_discr**2+delta_rcsb**2+delta_NPR**2)**0.5
        total_err_str = '{0:.2f}'.format(total_err)+r'\%'

        num_digits = int(np.floor(np.abs(
            np.log10(np.abs(total_err*central.val/100)))))+2
        central_val_str = r'$'+('{0:.%df}'%num_digits).format(central.val)+r'$'

        errors_dict[scheme][key] = {'central':central_val_str,
                                    'stat':stat_err_str,
                                    'chiral':chiral_err_str,
                                    'rcsb':rcsb_err_str,
                                    'discr':discr_err_str,
                                    'basis':NPR_err_str,
                                    'total':total_err_str}

        RISMOM_results[scheme][key] = {'central':central,
                                       'chiral':stat(
                                           val=central.val,
                                           err=np.abs(central.val*delta_chiral)/100,
                                           btsp='fill'
                                           ),
                                       'rcsb':stat(
                                           val=central.val,
                                           err=np.abs(central.val*delta_rcsb)/100,
                                           btsp='fill'
                                           ),
                                       'discr':stat(
                                           val=central.val,
                                           err=np.abs(central.val*delta_discr)/100,
                                           btsp='fill'
                                           ),
                                       'basis':stat(
                                           val=central.val,
                                           err=np.abs(central.val*delta_NPR)/100,
                                           btsp='fill'
                                           ),
                                       'total':stat(
                                           val=central.val,
                                           err=np.abs(total_err*central.val/100),
                                           btsp='fill'
                                           )}

MS_bar_results = {'gamma':{'name':r'$\overline{\text{MS}}\leftarrow\text{RI-SMOM}^{(\gamma_\mu, \gamma_\mu)}$'},
                  'qslash':{'name':r'$\overline{\text{MS}}\leftarrow\text{RI-SMOM}^{(\slashed{q}, \slashed{q})}$'},
                  'combined':{'name':r'$\overline{\text{MS}}$'}}

for scheme in MS_bar_results.keys():


    for i in range(5):
        MS_bar_results[scheme][f'B{i+1}'] = {}
    for i in range(4):
        MS_bar_results[scheme][f'R{i+2}'] = {}

    if scheme in ['gamma', 'qslash']:
        N_i = norm_factors(rotate=NPR_to_SUSY)
        R_conv_bag = np.diag(1/N_i)@R_RISMOM_MSbar(
                    3.0, scheme=scheme, obj='bag',
                    rotate=NPR_to_SUSY)@np.diag(N_i)
        R_conv_bag = stat(val=R_conv_bag, err=np.zeros((5,5)), btsp='fill')

        for err_type in ['central', 'chiral', 'rcsb', 'discr', 'basis']:
            bags = join_stats([RISMOM_results[scheme][f'B{idx+1}'][err_type] for idx in range(5)])
            MS_bag = R_conv_bag@bags
            for idx in range(5):
                MS_bar_results[scheme][f'B{idx+1}'][err_type] = MS_bag[idx]

        R_conv_rat = R_RISMOM_MSbar(
                    3.0, scheme=scheme, obj='ratio',
                    rotate=NPR_to_SUSY)
        R_conv_rat = stat(val=R_conv_rat, err=np.zeros((5,5)), btsp='fill')

        for err_type in ['central', 'chiral', 'rcsb', 'discr', 'basis']:
            ratios = join_stats([stat(val=1,err=0,btsp='fill')]+[
                RISMOM_results[scheme][f'R{idx+2}'][err_type] for idx in range(4)])
            MS_rat = R_conv_rat@ratios
            for idx in range(1,5):
                MS_bar_results[scheme][f'R{idx+1}'][err_type] = MS_rat[idx]

        for key in quantities.keys():
            central = MS_bar_results[scheme][key]['central']
            central_perc_err = np.abs(central.err/central.val)*100
            stat_err_str = '{0:.2f}'.format(central_perc_err)+r'\%'

            chiral_perc_err = np.abs(MS_bar_results[scheme][key]['chiral'].err*100/central.val)
            chiral_err_str = '{0:.2f}'.format(chiral_perc_err)+r'\%'

            rcsb_perc_err = np.abs(MS_bar_results[scheme][key]['rcsb'].err*100/central.val)
            rcsb_err_str = '{0:.2f}'.format(rcsb_perc_err)+r'\%'

            discr_perc_err = np.abs(MS_bar_results[scheme][key]['discr'].err*100/central.val)
            discr_err_str = '{0:.2f}'.format(discr_perc_err)+r'\%'

            NPR_perc_err = np.abs(MS_bar_results[scheme][key]['basis'].err*100/central.val)
            NPR_err_str = '{0:.2f}'.format(NPR_perc_err)+r'\%'

            total_err = (central_perc_err**2+chiral_perc_err**2+rcsb_perc_err**2+\
                    discr_perc_err**2+NPR_perc_err**2)**0.5
            total_err_str = '{0:.2f}'.format(total_err)+r'\%'

            total_err = np.abs(total_err*central.val/100)
            num_digits = int(np.floor(np.abs(np.log10(total_err))))+2
            central_val_str = r'$'+('{0:.%df}'%num_digits).format(central.val)+r'$'

            MS_bar_results[scheme][key]['str'] = {'central':central_val_str,
                                                  'stat':stat_err_str,
                                                  'chiral':chiral_err_str,
                                                  'rcsb':rcsb_err_str,
                                                  'discr':discr_err_str,
                                                  'basis':NPR_err_str,
                                                  'total':total_err_str}
    else:
        for key in quantities.keys():
            gamma_central = MS_bar_results['gamma'][key]['central'].val
            qslash_central = MS_bar_results['qslash'][key]['central'].val
            central = (gamma_central+qslash_central)/2
            stat_err = np.max([MS_bar_results[scheme][key]['central'].err
                               for scheme in ['gamma','qslash']])
            stat_err_perc = np.abs(stat_err/central)*100
            stat_err_str = '{0:.2f}'.format(stat_err_perc)+r'\%'

            chiral_err = np.max([MS_bar_results[scheme][key]['chiral'].err
                               for scheme in ['gamma','qslash']])
            chiral_err_perc = np.abs(chiral_err/central)*100
            chiral_err_str = '{0:.2f}'.format(chiral_err_perc)+r'\%'

            rcsb_err = np.max([MS_bar_results[scheme][key]['rcsb'].err
                               for scheme in ['gamma','qslash']])
            rcsb_err_perc = np.abs(rcsb_err/central)*100
            rcsb_err_str = '{0:.2f}'.format(rcsb_err_perc)+r'\%'

            discr_err = np.max([MS_bar_results[scheme][key]['discr'].err
                                for scheme in ['gamma','qslash']])
            discr_err_perc = np.abs(discr_err/central)*100
            discr_err_str = '{0:.2f}'.format(discr_err_perc)+r'\%'

            NPR_err = np.max([MS_bar_results[scheme][key]['basis'].err
                             for scheme in ['gamma','qslash']])
            NPR_err_perc = np.abs(NPR_err/central)*100
            NPR_err_str = '{0:.2f}'.format(NPR_err_perc)+r'\%'

            PT_err_perc = np.abs((gamma_central-qslash_central)/central)*50
            PT_err_str = '{0:.2f}'.format(PT_err_perc)+r'\%'


            total_err = (stat_err_perc**2+chiral_err_perc**2+rcsb_err_perc**2+\
                    discr_err_perc**2+PT_err_perc**2+NPR_err_perc**2)**0.5
            total_err_str = '{0:.2f}'.format(total_err)+r'\%'

            total_sys_err = np.abs(((chiral_err_perc**2+rcsb_err_perc**2+\
                    discr_err_perc**2+PT_err_perc**2+NPR_err_perc**2)**0.5)*central/100)
            key_stat = stat(
                    val=central,
                    err=(stat_err**2+total_sys_err**2)**0.5,
                    btsp='fill'
                    )
            key_stat.disp = err_disp(key_stat.val, stat_err, 
                                     sys_err=np.abs(total_sys_err))

            total_lat_err = np.abs(((chiral_err_perc**2+rcsb_err_perc**2+\
                    discr_err_perc**2+stat_err_perc**2+NPR_err_perc**2)**0.5)*central/100)
            key_stat.alt_disp = err_disp(key_stat.val, total_lat_err,
                                         sys_err=np.abs(PT_err_perc*central/100))

            num_digits = int(np.floor(np.abs(np.log10(key_stat.err))))+2
            central_val_str = r'$'+('{0:.%df}'%num_digits).format(central)+r'$'

            MS_bar_results[scheme][key]['str'] = {'central':central_val_str,
                                                  'stat':stat_err_str,
                                                  'chiral':chiral_err_str,
                                                  'rcsb':rcsb_err_str,
                                                  'discr':discr_err_str,
                                                  'basis':NPR_err_str,
                                                  'PT': PT_err_str,
                                                  'total':total_err_str}
            MS_bar_results[scheme][key]['store'] = key_stat
            print(key, key_stat.alt_disp)
#========================================================================================================
# all_systematics.tex

rv = [r'\begin{tabular}{c|c|cccc|ccccc}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r'scheme & & '+' & '.join([val for key,val in quantities.items()])+r'\\']
rv += [r'\hline']
for scheme in errors_dict.keys():
    rv += [r'\multirow{6}{*}{'+errors_dict[scheme]['name']+r'} & central & '+\
            ' & '.join([errors_dict[scheme][key]['central'] for key in list(quantities.keys())])+r' \\']
    rv += [r'\cline{2-11}']
    for err in ['stat', 'chiral', 'rcsb', 'discr', 'basis', 'total']:
        if err=='total':
            rv += [r'\cline{2-11}']
        rv += [r' & '+err+r' & '+' & '.join([errors_dict[scheme][key][err]
                                             for key in list(quantities.keys())])+r' \\']
    rv += [r'\hline']

rv += [r'\multirow{6}{*}{'+MS_bar_results['combined']['name']+r'} & central & '+\
        ' & '.join([MS_bar_results['combined'][key]['str']['central']
                    for key in list(quantities.keys())])+r' \\']
rv += [r'\cline{2-11}']
for err in ['stat', 'chiral', 'rcsb', 'discr', 'basis', 'PT', 'total']:
    rv += [r' & '+err+r' & '+' & '.join([MS_bar_results['combined'][key]['str'][err]
                                         for key in list(quantities.keys())])+r' \\']
    if err=='PT':
        rv += [r'\cline{2-11}']
rv += [r'\hline']

rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/all_systematics{expand_str}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')


#========================================================================================================
# scheme_systematics.tex

rv = [r'\begin{tabular}{c|c|cccc|ccccc}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r'scheme & & '+' & '.join([val for key,val in quantities.items()])+r'\\']
rv += [r'\hline']
for scheme in ['gamma', 'qslash']:
    rv += [r'\multirow{6}{*}{'+MS_bar_results[scheme]['name']+r'} & central & '+\
            ' & '.join([MS_bar_results[scheme][key]['str']['central']
                        for key in list(quantities.keys())])+r' \\']
    rv += [r'\cline{2-11}']
    for err in ['stat', 'chiral', 'rcsb', 'discr', 'basis', 'total']:
        if err=='total':
            rv += [r'\cline{2-11}']
        rv += [r' & '+err+r' & '+' & '.join([MS_bar_results[scheme][key]['str'][err]
                                             for key in list(quantities.keys())])+r' \\']
    rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/scheme_systematics{expand_str}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')


pickle.dump(MS_bar_results, open(f'MS_bar_results{expand_str}.p','wb'))

#==========================================================================================================
# RI/SMOM ratios

fig, ax = plt.subplots(figsize=(4,2.5))
NB_R = []
for idx in range(1,5):
    Ri = RISMOM_results['gamma'][f'R{idx+1}']['total']
    Bi = RISMOM_results['gamma'][f'B{idx+1}']['total']

    Ni = norm_factors(rotate=NPR_to_SUSY)[idx]

    ki = (Bi/Ri)*Ni
    NB_R.append(ki)
    print(f'{idx+1}:{err_disp(ki.val, ki.err)}')

def constant_ansatz(x, param, **kwargs):
    return param[0]*np.ones(len(x))

x = stat(val=np.arange(2,6), err=np.zeros(4), btsp='fill')
y = join_stats(NB_R)
res = fit_func(x,y,constant_ansatz,guess=[0.5,0],
               correlated=True, start=0,end=4)
ax.axhspan(res.val[0]+res.err[0],
           res.val[0]-res.err[0],
           color='r', alpha=0.2)
ax.axhline(res.val[0], color='r', label=err_disp(res.val[0], res.err[0]))

ax.errorbar(np.arange(2,6),
            [k.val for k in NB_R],
            yerr=[k.err for k in NB_R],
            fmt='o', capsize=4, c='k')
ax.set_ylabel(r'$N_i\mathcal{B}_i^{\mathrm{RI}}/R_i^{\mathrm{RI}}$', size=16)
ax.set_xlabel(r'$i$', size=16)
ax.set_xticks([2,3,4,5])
ax.legend()

N1 = norm_factors()[0]
B1 = RISMOM_results['gamma']['B1']['total']
m_K_pm = stat(val=493.677, err=0.013, btsp='fill')/1000
m_K_0 = stat(val=498.611, err=0.013, btsp='fill')/1000
m_K = (m_K_0+m_K_pm)/2
mass_sum_pred = (res[0]*(m_K**2)/(B1*N1))**0.5
print(f'(m_s+m_d) in RISMOM: {err_disp(mass_sum_pred.val, mass_sum_pred.err)}')

filename = '/Users/rajnandinimukherjee/Desktop/draft_plots/summary_plots/NiBioverRi_comparison_RISMOM.pdf'
call_PDF(filename, open=True)
