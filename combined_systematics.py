from cont_chir_extrap import *

fit_systematics = pickle.load(open('fit_systematics_20.p', 'rb'))
scaling_systematics = pickle.load(open('scaling_systematics.p', 'rb'))
other_systematics = pickle.load(open('other_systematics.p', 'rb'))

errors_dict = {}
quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})
for key in quantities.keys():
    errors_dict[key] = {}

    central = scaling_systematics['(2,3)'][key]
    central_val_str = '{0:.4}'.format(central.val)
    central_perc_err = np.abs(central.err/central.val)*100
    stat_err_str = '{0:.2f}'.format(central_perc_err)+r'\%'

    central_2 = fit_systematics['central'][key][0]
    chiral_fit = fit_systematics['log'][key][0]
    delta_chiral = np.abs(((chiral_fit-central_2)/((chiral_fit+central_2)*0.5)).val*100) 
    chiral_err_str = '{0:.2f}'.format(delta_chiral)+r'\%'

    rcsb = other_systematics['no_mask'][key][3.0]
    delta_rcsb = np.abs(((rcsb-central)/((rcsb+central)*0.5)).val*100)
    rcsb_err_str = '{0:.2f}'.format(delta_rcsb)+r'\%'

    discr_fit = scaling_systematics['(3)'][key]
    delta_discr = np.abs(((discr_fit-central)/((discr_fit+central)*0.5)).val*100) 
    discr_err_str = '{0:.2f}'.format(delta_discr)+r'\%'

    total_err = (central_perc_err**2+delta_chiral**2+delta_discr**2)**0.5
    total_err_str = '{0:.2f}'.format(total_err)+r'\%'

    errors_dict[key] = {'central':central_val_str,
                        'stat':stat_err_str,
                        'chiral':chiral_err_str,
                        'rcsb':rcsb_err_str,
                        'discr':discr_err_str,
                        'total':total_err_str}

rv = [r'\begin{table}']
rv += [r'\caption{\label{tab:error} central values and combined systematic errors '+\
        r' for ratio and bag parameters at $\mu=3$ GeV in the $\textrm{SMOM}^{(\gamma_\mu,\gamma_\mu)}$ '+\
        r'scheme. We list the errors arising from statistics, chiral extrapolation, '+\
        r'residual chiral symmetry breaking (rcsb) effect, and discretisation and combine '+\
        r'it into total uncertainties.}']
rv += [r'\begin{tabular}{c|c|cccc|ccccc}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r'scheme & & '+' & '.join([val for key,val in quantities.items()])+r'\\']
rv += [r'\hline']
rv += [r'\multirow{6}{*}{$\textrm{SMOM}^{(\gamma_\mu,\gamma_\mu)}$} & central & '+\
        ' & '.join([errors_dict[key]['central'] for key in list(quantities.keys())])+r' \\']
rv += [r'\cline{2-11}']
for err in ['stat', 'chiral', 'rcsb', 'discr', 'total']:
    rv += [r' & '+err+r' & '+' & '.join([errors_dict[key][err] for key in list(quantities.keys())])+r' \\']
    if err=='discr':
        rv += [r'\cline{2-11}']

rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']
rv += [r'\end{table}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables/all_systematics.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')
