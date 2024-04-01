from cont_chir_extrap import *

mu = 2.0
expand_str = '_expanded' if expand_err else ''
chiral_extrap = True
fit_filename = f'fit_systematics_{str(int(mu*10))}_{fit_file}_{scheme}{expand_str}.p'
run = bool(int(input('run?(0:False/1:True): ')))
print(f'Running fit systematics in {scheme} scheme using data from {fit_file}')

if run:
    laxis = {1:1, 2:1, 3:1, 4:0, 5:0}
    record_vals = {'central':{'kwargs':{'plot':True,
                                        'open':False,
                                        'log':True,
                                        'expand_err':expand_err,
                                        'ens_list':[
                                          b for b in bag_ensembles if b!='C2'],
                                        'addnl_terms':'del_ms',
                                        'print_F1M_recons':True,
                                        'guess':[1,1e-1,1e-2,1e-2]}},
                   'no_del_ms':{'kwargs':{'log':True,
                                          'ens_list':[
                                          b for b in bag_ensembles if b!='C2'],
                                          'plot':False}},
                   'noC2cut':{'kwargs':{'plot':False,
                                      'log':True,
                                      'addnl_terms':'del_ms',
                                      'guess':[1,1e-1,1e-2,1e-2]}},
                   'C2M3cut':{'kwargs':{'ens_list':[
                       b for b in bag_ensembles if b!='C2' and b!='M3'],
                                        'plot':False,
                                        'log':True,
                                        'addnl_terms':'del_ms',
                                        'guess':[1,1e-1,1e-2,1e-2]}},
                   'C2M3M2cut':{'kwargs':{'ens_list':[
                       b for b in bag_ensembles if b!='C2' and b!='M3' and b!='M2'],
                                          'plot':False,
                                          'log':True,
                                          'addnl_terms':'del_ms',
                                          'guess':[1,1e-1,1e-2,1e-2]}},
                   'no_log':{'kwargs':{'log':False,
                                       'plot':False,
                                       'ens_list':[
                                          b for b in bag_ensembles if b!='C2'],
                                       'addnl_terms':'del_ms',
                                       'guess':[1,1e-1,1e-2,1e-2]}}
                           }


    b = bag_fits(bag_ensembles, obj='bag', scheme=scheme)
    for op_idx, op in enumerate(b.operators):
        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/new_{fit_file}/bag_fits_B{op_idx+1}_{scheme}{expand_str}_{int(mu*10)}.pdf'
        for fit in record_vals.keys():
            record_vals[fit][f'B{op_idx+1}'] = b.fit_operator(mu, op, rotate=NPR_to_SUSY,
                                                        chiral_extrap=chiral_extrap, rescale=True, fs=14,
                                                        figsize=(10,3), label='', legend_axis=laxis[op_idx+1],
                                                        filename=filename, **record_vals[fit]['kwargs'])
            del record_vals[fit][f'B{op_idx+1}'].mapping


    r = bag_fits(bag_ensembles, obj='ratio', scheme=scheme)
    for op_idx, op in enumerate(r.operators):
        filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/new_{fit_file}/ratio_fits_R{op_idx+2}_{scheme}{expand_str}_{int(mu*10)}.pdf'
        for fit in record_vals.keys():
            record_vals[fit][f'R{op_idx+2}'] = r.fit_operator(mu, op, rotate=NPR_to_SUSY,
                                                             chiral_extrap=chiral_extrap, fs=14,
                                                             figsize=(10,3), label='', legend_axis=1,
                                                             filename=filename, **record_vals[fit]['kwargs'])
            del record_vals[fit][f'R{op_idx+2}'].mapping

    
    for fit in record_vals.keys():
        del record_vals[fit]['kwargs']

    pickle.dump(record_vals, open(fit_filename,'wb'))
else:
    record_vals = pickle.load(open(fit_filename, 'rb'))

other_fitter = 'Felix' if fit_file=='Tobi' else 'Tobi'
other_fit_filename = f'fit_systematics_{str(int(mu*10))}_{other_fitter}_{scheme}{expand_str}.p'
other_record_vals = pickle.load(open(other_fit_filename, 'rb'))


quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})

rv = [r'\begin{tabular}{c|c|c|c|c|c|c|c}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r' & $m_\pi<420$ MeV  & $-\gamma\delta_{m_s}$ & $-L(m_\pi^2)$  '+\
        r' & $m_\pi<440$ MeV & $m_\pi<370$ MeV & $m_\pi<350$ MeV & '+other_fitter+r' fit\\']
rv += [r' & central value & $\delta$ &  $\delta^\text{chiral}$ &'+\
        r' $\delta$ & $\delta$ & $\delta$ & $\delta$ \\']

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
    central_str = r'\textcolor{'+central_color+r'}{$'+err_disp(Y_0.val, Y_0.err)+r'$}'

    no_gamma_fit = record_vals['no_del_ms'][key]
    no_gamma_color = pval_color(no_gamma_fit.pvalue)
    no_gamma_change = np.abs(((no_gamma_fit[0]-Y_0)/Y_0).val*100)
    no_gamma_change_str = r'\textcolor{'+no_gamma_color+r'}{$'+'{0:.2f}'.format(no_gamma_change)+r'$\%}'

    noC2cut_fit = record_vals['noC2cut'][key]
    noC2cut_change = np.abs(((noC2cut_fit[0]-Y_0)/((Y_0+noC2cut_fit[0])*0.5)).val*100)
    noC2cut_color = pval_color(noC2cut_fit.pvalue)
    noC2cut_change_str = r'\textcolor{'+noC2cut_color+r'}{$'+'{0:.2f}'.format(noC2cut_change)+r'$\%}' 
    

    C2M3cut_fit = record_vals['C2M3cut'][key]
    C2M3cut_change = np.abs(((C2M3cut_fit[0]-Y_0)/((Y_0+C2M3cut_fit[0])*0.5)).val*100)
    C2M3cut_color = pval_color(C2M3cut_fit.pvalue)
    C2M3cut_change_str = r'\textcolor{'+C2M3cut_color+r'}{$'+'{0:.2f}'.format(C2M3cut_change)+r'$\%}' 

    C2M3M2cut_fit = record_vals['C2M3M2cut'][key]
    C2M3M2cut_change = np.abs(((C2M3M2cut_fit[0]-Y_0)/((Y_0+C2M3M2cut_fit[0])*0.5)).val*100)
    C2M3M2cut_color = pval_color(C2M3M2cut_fit.pvalue)
    C2M3M2cut_change_str = r'\textcolor{'+C2M3M2cut_color+r'}{$'+'{0:.2f}'.format(
            C2M3M2cut_change)+r'$\%}' 

    no_log_fit = record_vals['no_log'][key]
    no_log_change = np.abs(((Y_0-no_log_fit[0])/((Y_0+no_log_fit[0])*0.5)).val*100)
    no_log_color = pval_color(no_log_fit.pvalue)
    no_log_change_str = r'\textcolor{'+no_log_color+r'}{$'+'{0:.2f}'.format(
            np.abs(no_log_change))+r'$\%}' 
    if key=='R2' or key=='R3':
        no_log_change_str = '-'

    other_fit = other_record_vals['central'][key]
    other_color = pval_color(other_fit.pvalue)
    other_change = np.abs(((other_fit[0]-Y_0)/((Y_0+other_fit[0])*0.5)).val*100)
    other_change_str = r'\textcolor{'+other_color+r'}{$'+'{0:.2f}'.format(other_change)+r'$\%}' 

    rv += [' & '.join([name, 
                       central_str, 
                       no_gamma_change_str,
                       no_log_change_str,
                       noC2cut_change_str, 
                       C2M3cut_change_str,
                       C2M3M2cut_change_str, 
                       other_change_str
                       ]) + r'\\']
    if key=='R5':
        rv += [r'\hline']

rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/fit_systematics_{str(int(10*mu))}_{scheme}{expand_str}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')
