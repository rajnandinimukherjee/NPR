from cont_chir_extrap import *

errors_dict = {'gamma':{'name':r'$\textrm{RI-SMOM}^{(\gamma_\mu,\gamma_\mu)}$'},
               'qslash':{'name':r'$\textrm{RI-SMOM}^{(\slashed{q}, \slashed{q})}$'}}

quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})

RISMOM_results = {'gamma':{}, 'qslash':{}}

for scheme in errors_dict.keys():
    fit_systematics = pickle.load(open(f'fit_systematics_20_{fit_file}_{scheme}.p', 'rb'))
    scaling_systematics = pickle.load(open(f'scaling_systematics_{scheme}_{fit_file}.p', 'rb'))
    other_systematics = pickle.load(open(f'other_systematics_{scheme}_{fit_file}.p', 'rb'))
    for key in quantities.keys():
        central = scaling_systematics['(2,3)'][key]
        central_val_str = '{0:.4f}'.format(central.val)
        central_perc_err = np.abs(central.err/central.val)*100
        stat_err_str = '{0:.2f}'.format(central_perc_err)+r'\%'

        central_2 = fit_systematics['central'][key][0]
        chiral_fits = [fit_systematics[fittype][key][0] for fittype in ['C2M3cut','C2M3M2cut','log']]
        delta_chirals = [np.abs(((chiral_fit-central_2)/((chiral_fit+central_2)*0.5)).val*100)
                         for chiral_fit in chiral_fits]
        delta_chiral = max(delta_chirals)
        chiral_err_str = '{0:.2f}'.format(delta_chiral)+r'\%'

        rcsb = other_systematics['no_mask'][key][3.0]
        delta_rcsb = np.abs(((rcsb-central)/((rcsb+central)*0.5)).val*100)
        rcsb_err_str = '{0:.2f}'.format(delta_rcsb)+r'\%'

        discr_fits = [scaling_systematics[fittype][key] for fittype in ['(3)', '(2,3,0.2)']]
        delta_discrs = [np.abs(((discr_fit-central)/((discr_fit+central)*0.5)).val*100)
                        for discr_fit in discr_fits]
        delta_discr = max(delta_discrs)
        discr_err_str = '{0:.2f}'.format(delta_discr)+r'\%'

        total_err = (central_perc_err**2+delta_chiral**2+delta_discr**2+delta_rcsb**2)**0.5
        total_err_str = '{0:.2f}'.format(total_err)+r'\%'

        errors_dict[scheme][key] = {'central':central_val_str,
                                    'stat':stat_err_str,
                                    'chiral':chiral_err_str,
                                    'rcsb':rcsb_err_str,
                                    'discr':discr_err_str,
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
                                           )}

MS_bar_results = {'gamma':{'name':r'$\overline{\text{MS}}\leftarrow\text{RI-SMOM}^{(\gamma_\mu, \gamma_\mu)}$'},
                  'qslash':{'name':r'$\overline{\text{MS}}\leftarrow\text{RI-SMOM}^{(\slashed{q}, \slashed{q})}$'}}

for scheme in RISMOM_results.keys():


    for i in range(5):
        MS_bar_results[scheme][f'B{i+1}'] = {}

    N_i = norm_factors(rotate=NPR_to_SUSY)
    R_conv_bag = np.diag(1/N_i)@R_RISMOM_MSbar(
                3.0, scheme=scheme, obj='bag',
                rotate=NPR_to_SUSY)@np.diag(N_i)
    R_conv_bag = stat(val=R_conv_bag, err=np.zeros((5,5)), btsp='fill')

    for err_type in ['central', 'chiral', 'rcsb', 'discr']:
        bags = join_stats([RISMOM_results[scheme][f'B{idx+1}'][err_type] for idx in range(5)])
        MS_bag = R_conv_bag@bags
        for idx in range(5):
            MS_bar_results[scheme][f'B{idx+1}'][err_type] = MS_bag[idx]

    for i in range(4):
        MS_bar_results[scheme][f'R{i+2}'] = {}

    R_conv_rat = np.diag(1/N_i)@R_RISMOM_MSbar(
                3.0, scheme=scheme, obj='ratio',
                rotate=NPR_to_SUSY)@np.diag(N_i)
    R_conv_rat = stat(val=R_conv_rat, err=np.zeros((5,5)), btsp='fill')

    for err_type in ['central', 'chiral', 'rcsb', 'discr']:
        ratios = join_stats([stat(val=1,err=0,btsp='fill')]+[
            RISMOM_results[scheme][f'R{idx+2}'][err_type] for idx in range(4)])
        MS_rat = R_conv_rat@ratios
        for idx in range(1,5):
            MS_bar_results[scheme][f'R{idx+1}'][err_type] = MS_rat[idx]

    for key in quantities.keys():
        central = MS_bar_results[scheme][key]['central']
        central_val_str = '{0:.4f}'.format(central.val)
        central_perc_err = np.abs(central.err/central.val)*100
        stat_err_str = '{0:.2f}'.format(central_perc_err)+r'\%'

        chiral_perc_err = np.abs(MS_bar_results[scheme][key]['chiral'].err*100/central.val)
        chiral_err_str = '{0:.2f}'.format(chiral_perc_err)+r'\%'

        rcsb_perc_err = np.abs(MS_bar_results[scheme][key]['rcsb'].err*100/central.val)
        rcsb_err_str = '{0:.2f}'.format(rcsb_perc_err)+r'\%'

        discr_perc_err = np.abs(MS_bar_results[scheme][key]['discr'].err*100/central.val)
        discr_err_str = '{0:.2f}'.format(discr_perc_err)+r'\%'

        total_err = (central_perc_err**2+chiral_perc_err**2+rcsb_perc_err**2+discr_perc_err**2)**0.5
        total_err_str = '{0:.2f}'.format(total_err)+r'\%'

        MS_bar_results[scheme][key]['str'] = {'central':central_val_str,
                                              'stat':stat_err_str,
                                              'chiral':chiral_err_str,
                                              'rcsb':rcsb_err_str,
                                              'discr':discr_err_str,
                                              'total':total_err_str}

rv = [r'\begin{tabular}{c|c|cccc|ccccc}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r'scheme & & '+' & '.join([val for key,val in quantities.items()])+r'\\']
rv += [r'\hline']

for scheme in errors_dict.keys():
    rv += [r'\multirow{6}{*}{'+errors_dict[scheme]['name']+r'} & central & '+\
            ' & '.join([errors_dict[scheme][key]['central'] for key in list(quantities.keys())])+r' \\']
    rv += [r'\cline{2-11}']
    for err in ['stat', 'chiral', 'rcsb', 'discr', 'total']:
        rv += [r' & '+err+r' & '+' & '.join([errors_dict[scheme][key][err]
                                             for key in list(quantities.keys())])+r' \\']
        if err=='discr':
            rv += [r'\cline{2-11}']
    rv += [r'\hline']
rv += [r'\hline']
for scheme in MS_bar_results.keys():
    rv += [r'\multirow{6}{*}{'+MS_bar_results[scheme]['name']+r'} & central & '+\
            ' & '.join([MS_bar_results[scheme][key]['str']['central']
                        for key in list(quantities.keys())])+r' \\']
    rv += [r'\cline{2-11}']
    for err in ['stat', 'chiral', 'rcsb', 'discr', 'total']:
        rv += [r' & '+err+r' & '+' & '.join([MS_bar_results[scheme][key]['str'][err]
                                             for key in list(quantities.keys())])+r' \\']
        if err=='discr':
            rv += [r'\cline{2-11}']
    rv += [r'\hline']

rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/all_systematics.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')
