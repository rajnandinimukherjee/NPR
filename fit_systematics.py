from cont_chir_extrap import *

mu = 2.0
run = bool(int(input('run?(0:False/1:True): ')))

if run:
    laxis = {1:1, 2:1, 3:1, 4:0, 5:0}
    record_vals = {'central':{'kwargs':{'plot':True, 'open':False,
                                        'ens_list':[b for b in bag_ensembles if b!='C2']}},
                  'del_ms':{'kwargs':{'addnl_terms':'del_ms',
                                      'guess':[1, 1e-1, 1e-2, 1e-3],
                                      'ens_list':[b for b in bag_ensembles if b!='C2'],
                                      'plot':False}},
                  'noC2cut':{'kwargs':{'plot':False}},
                  'C2M3cut':{'kwargs':{'ens_list':[
                      b for b in bag_ensembles if b!='C2' and b!='M3'],'plot':False}},
                  'C2M3M2cut':{'kwargs':{'ens_list':[
                      b for b in bag_ensembles if b!='C2' and b!='M3' and b!='M2'],'plot':False}},
                   'log':{'kwargs':{'ens_list':[b for b in bag_ensembles if b!='C2'],
                                    'addnl_terms':'log', 'plot':False}}
                           }


    b = bag_fits(bag_ensembles, obj='bag')
    for op_idx, op in enumerate(b.operators):
        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/linear_{fit_file}/bag_fits_B{op_idx+1}_{int(mu*10)}.pdf'
        for fit in record_vals.keys():
            record_vals[fit][f'B{op_idx+1}'] = b.fit_operator(mu, op, rotate=NPR_to_SUSY,
                                                        chiral_extrap=True, rescale=True, fs=14,
                                                        figsize=(10,3), label='', legend_axis=laxis[op_idx+1],
                                                        filename=filename, **record_vals[fit]['kwargs'])
            del record_vals[fit][f'B{op_idx+1}'].mapping


    r = bag_fits(bag_ensembles, obj='ratio')
    for op_idx, op in enumerate(r.operators):
        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/linear_{fit_file}/ratio_fits_R{op_idx+2}_{int(mu*10)}.pdf'
        for fit in record_vals.keys():
            record_vals[fit][f'R{op_idx+2}'] = r.fit_operator(mu, op, rotate=NPR_to_SUSY,
                                                             chiral_extrap=True, fs=14,
                                                             figsize=(10,3), label='', legend_axis=0,
                                                             filename=filename, **record_vals[fit]['kwargs'])
            del record_vals[fit][f'R{op_idx+2}'].mapping

    
    for fit in record_vals.keys():
        del record_vals[fit]['kwargs']

    pickle.dump(record_vals, open(f'fit_systematics_{int(mu*10)}_{fit_file}.p','wb'))
else:
    record_vals = pickle.load(open(f'fit_systematics_{int(mu*10)}_{fit_file}.p', 'rb'))


quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})

rv = [r'\begin{tabular}{c|c|c|c|c|c|c|>{\columncolor[gray]{0.95}}c}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r' & no C2S & '+\
        r'\multicolumn{2}{c|}{$+\gamma\delta_{m_s^{\text{sea}}}$, no C2S}'+\
        r' & with C2S & no C2S, M3S & no C2S, M3S, M2S & $+L(m_\pi^2)$, no C2S\\']
rv += [r' & central value & $\gamma$\% & $\delta$ &'+\
        r' $\delta$ & $\delta$ & $\delta$ & $\delta^\text{chiral}$ \\']

rv += [r'\hline']

def pval_color(pval):
    if pval<0.03:
        return 'red'
    elif pval>0.03 and pval<0.05:
        return 'orange'
    else:
        return 'ForestGreen'

for key in quantities.keys():
    name = quantities[key]

    central = record_vals['central'][key]
    Y_0 = central[0]
    central_color = pval_color(central.pvalue)
    central_str = r'\textcolor{'+central_color+r'}{'+err_disp(Y_0.val, Y_0.err)+r'}'

    gamma_fit = record_vals['del_ms'][key]
    gamma_over_Y_0 = gamma_fit[-1]/gamma_fit[0]
    gamma_over_Y_0_str = err_disp(gamma_over_Y_0.val, gamma_over_Y_0.err)
    gamma_color = pval_color(gamma_fit.pvalue)
    gamma_change = np.abs(((gamma_fit[0]-Y_0)/Y_0).val*100)
    gamma_change_str = r'\textcolor{'+gamma_color+r'}{'+'{0:.2f}'.format(gamma_change)+r'\%}'

    noC2cut_fit = record_vals['noC2cut'][key]
    noC2cut_change = np.abs(((noC2cut_fit[0]-Y_0)/Y_0).val*100)
    noC2cut_color = pval_color(noC2cut_fit.pvalue)
    noC2cut_change_str = r'\textcolor{'+noC2cut_color+r'}{'+'{0:.2f}'.format(noC2cut_change)+r'\%}' 
    

    C2M3cut_fit = record_vals['C2M3cut'][key]
    C2M3cut_change = np.abs(((C2M3cut_fit[0]-Y_0)/Y_0).val*100)
    C2M3cut_color = pval_color(C2M3cut_fit.pvalue)
    C2M3cut_change_str = r'\textcolor{'+C2M3cut_color+r'}{'+'{0:.2f}'.format(C2M3cut_change)+r'\%}' 

    C2M3M2cut_fit = record_vals['C2M3M2cut'][key]
    C2M3M2cut_change = np.abs(((C2M3M2cut_fit[0]-Y_0)/Y_0).val*100)
    C2M3M2cut_color = pval_color(C2M3M2cut_fit.pvalue)
    C2M3M2cut_change_str = r'\textcolor{'+C2M3M2cut_color+r'}{'+'{0:.2f}'.format(
            C2M3M2cut_change)+r'\%}' 

    log_fit = record_vals['log'][key]
    log_change = np.abs(((Y_0-log_fit[0])/((Y_0+log_fit[0])*0.5)).val*100)
    log_color = pval_color(log_fit.pvalue)
    log_change_str = r'\textcolor{'+log_color+r'}{'+'{0:.2f}'.format(
            np.abs(log_change))+r'\%}' 

    max_var = max(gamma_change, noC2cut_change, C2M3cut_change, C2M3M2cut_change)
    max_var_str = '{0:.2f}'.format(max_var)+r'\%'

    rv += [' & '.join([name, 
                       central_str, 
                       gamma_over_Y_0_str, 
                       gamma_change_str,
                       noC2cut_change_str, 
                       C2M3cut_change_str,
                       C2M3M2cut_change_str, 
                       log_change_str,
                       #max_var_str,
                       ]) + r'\\']
    if key=='R5':
        rv += [r'\hline']

rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/fit_systematics_{str(int(10*mu))}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')
