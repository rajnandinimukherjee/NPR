from cont_chir_extrap import *

run = bool(int(input('run?(0:False/1:True): ')))
print(f'Running scaling systematics in {scheme} scheme using data from {fit_file}')

expand_str = '_expanded' if expand_err else ''
scaling_filename = f'scaling_systematics_{scheme}_{fit_file}{expand_str}.p'

if run:
    s_bag = sigma(norm='bag', scheme=scheme)
    record_vals = {'(2,3)':{}, '(2,3,0.2)':{}, '(3)':{}}

    b = bag_fits(bag_ensembles, obj='bag', scheme=scheme)

    bags = [b.fit_operator(2.0, op, rotate=NPR_to_SUSY,
                           log=True, addnl_terms='del_ms',
                           guess=[1,1e-1,1e-2,1e-2],
                           ens_list=[b for b in bag_ensembles if b!='C2'],
                           expand_err=expand_err,
                           chiral_extrap=True)
            for op in b.operators]

    pvalues = [bag.pvalue for bag in bags]
    bags = join_stats([bag[0] for bag in bags])
    sig_bag = s_bag.calc_running(2.0, 3.0, chiral_extrap=True,
                                 rotate=NPR_to_SUSY)
    steps = np.linspace(2.0,3.0,6)
    sig_bag_steps = stat(val=np.eye(len(b.operators)),
                         btsp=[np.eye(len(b.operators))
                               for k in range(N_boot)])
    for i in range(len(steps)-1):
        sig_bag_steps = s_bag.calc_running(steps[i], steps[i+1],
                                           chiral_extrap=True,
                                           rotate=NPR_to_SUSY)@sig_bag_steps

    N_i = norm_factors(rotate=NPR_to_SUSY)
    for i in range(len(bags.val)):
        record_vals['(2,3)'][f'B{i+1}'] = (sig_bag@bags)[i]/N_i[i]
        record_vals['(2,3)'][f'B{i+1}'].pvalue = pvalues[i]
        record_vals['(2,3,0.2)'][f'B{i+1}'] = (sig_bag_steps@bags)[i]/N_i[i]

    bags = [b.fit_operator(3.0, op, rotate=NPR_to_SUSY,
                           log=True, addnl_terms='del_ms',
                           guess=[1,1e-1,1e-2,1e-2],
                           ens_list=[b for b in bag_ensembles if b!='C2'],
                           expand_err=expand_err,
                           chiral_extrap=True)
            for op in b.operators]

    for i in range(len(bags)):
        b3 = bags[i][0]/N_i[i]
        b3.pvalue = bags[i].pvalue
        record_vals['(3)'][f'B{i+1}'] = b3

    del b
    s_rat = sigma(norm='11', scheme=scheme)
    r = bag_fits(bag_ensembles, obj='ratio', scheme=scheme)
    ratios = [r.fit_operator(2.0, op, rotate=NPR_to_SUSY,
                             log=True, addnl_terms='del_ms',
                             guess=[1,1e-1,1e-2,1e-2],
                             ens_list=[b for b in bag_ensembles if b!='C2'],
                             expand_err=expand_err,
                             plot=False, chiral_extrap=True)
              for op in r.operators]

    pvalues = [rat.pvalue for rat in ratios]
    one = stat(val=1, err=0, btsp='fill')
    ratios = join_stats([one]+[rat[0] for rat in ratios])
    sig_rat = s_rat.calc_running(2.0, 3.0, chiral_extrap=True,
                                 rotate=NPR_to_SUSY)
    sig_rat_steps = stat(val=np.eye(5),
                         btsp=[np.eye(5) for k in range(N_boot)])
    for i in range(len(steps)-1):
        sig_rat_steps = s_rat.calc_running(steps[i], steps[i+1],
                                           chiral_extrap=True,
                                           rotate=NPR_to_SUSY)@sig_rat_steps
    for i in range(1, len(ratios.val)):
        record_vals['(2,3)'][f'R{i+1}'] = (sig_rat@ratios)[i]
        record_vals['(2,3)'][f'R{i+1}'].pvalue = pvalues[i-1]
        record_vals['(2,3,0.2)'][f'R{i+1}'] = (sig_rat_steps@ratios)[i]

    ratios = [r.fit_operator(3.0, op, rotate=NPR_to_SUSY,
                             log=True, addnl_terms='del_ms',
                             guess=[1,1e-1,1e-2,1e-2],
                             ens_list=[b for b in bag_ensembles if b!='C2'],
                             expand_err=expand_err,
                             plot=False, chiral_extrap=True)
              for op in r.operators]

    for i in range(len(r.operators)):
        r3 = ratios[i][0]
        r3.pvalue = ratios[i].pvalue
        record_vals['(3)'][f'R{i+2}'] = r3
    del r

    pickle.dump(record_vals, open(scaling_filename, 'wb'))
else:
    record_vals = pickle.load(open(scaling_filename, 'rb'))

other_systematics = pickle.load(open(f'other_systematics_{scheme}_{fit_file}{expand_str}.p','rb'))

quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$\mathcal{B}_'+str(i+1)+r'$' for i in range(5)})

rv = [r'\begin{tabular}{c|c|c|c|c|c}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r' & $\sigma(3\,\mathrm{GeV},3\,\mathrm{GeV})$ & $\sigma('+\
        r'3\,\mathrm{GeV}\xleftarrow{\Delta=0.2}2\,\mathrm{GeV})$ & NPR at 3 GeV '+\
        #r'& mask post-inv '+\
        r'& no mask & SUSY$\leftarrow$NPR \\']
rv += [r' & central value & $\delta$ & $\delta$ & '\
        r'$\delta$ & $\delta$ \\']
rv += [r'\hline']

def pval_color(pval):
    if pval<0.03:
        return 'red'
    elif pval>0.03 and pval<0.05:
        return 'orange'
    else:
        return 'ForestGreen'


mu1, mu2 = 2.0, 3.0
for key in quantities.keys():
    name = quantities[key]

    central = record_vals['(2,3)'][key]
    central_color = pval_color(central.pvalue)
    central_str = err_disp(central.val, central.err)
    central_str = r'\textcolor{'+central_color+r'}{$'+err_disp(central.val, central.err)+r'$}'

    steps = record_vals['(2,3,0.2)'][key]
    steps_change = np.abs(((steps-central)/((steps+central)*0.5)).val*100) 
    steps_change_str = r'\textcolor{'+central_color+r'}{$'+'{0:.2f}'.format(steps_change)+r'$\%}'

    direct = record_vals['(3)'][key]
    direct_color = pval_color(direct.pvalue)
    delta_disc = np.abs(((direct-central)/((direct+central)*0.5)).val*100)
    direct_disc_str = r'\textcolor{'+direct_color+r'}{$'+'{0:.2f}'.format(delta_disc)+r'$\%}'

    #post_inv = other_systematics['mask_post_inv'][key][mu2]
    #post_inv_color = pval_color(other_systematics['mask_post_inv'][key][mu1].pvalue) 
    #post_inv_change = np.abs(((post_inv-central)/((post_inv+central)*0.5)).val*100) 
    #post_inv_change_str = r'\textcolor{'+post_inv_color+r'}{'+'{0:.3f}'.format(post_inv_change)+r'\%}'

    no_mask = other_systematics['no_mask'][key][mu2]
    no_mask_color = pval_color(other_systematics['no_mask'][key][mu1].pvalue) 
    no_mask_change = np.abs(((no_mask-central)/((no_mask+central)*0.5)).val*100) 
    no_mask_change_str = r'\textcolor{'+no_mask_color+r'}{$'+'{0:.2f}'.format(no_mask_change)+r'$\%}'

    NPR = other_systematics['NPR'][key][mu2]
    NPR_color = pval_color(other_systematics['NPR'][key][mu1].pvalue)
    NPR_change = np.abs(((NPR-central)/((NPR+central)*0.5)).val*100) 
    NPR_change_str = r'\textcolor{'+NPR_color+r'}{$'+'{0:.2f}'.format(NPR_change)+r'$\%}'

    rv += [' & '.join([name, 
                       central_str, 
                       steps_change_str, 
                       direct_disc_str,
                       #post_inv_change_str,
                       no_mask_change_str,
                       NPR_change_str
                       ]) + r'\\']
    if key=='R5':
        rv += [r'\hline']

rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/scaling_systematics_{scheme}{expand_str}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')


