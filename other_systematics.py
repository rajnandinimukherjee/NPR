from cont_chir_extrap import *

mu1 = 2.0
mu2 = 3.0
run = bool(int(input('run?(0:False/1:True): ')))
scheme = 'qslash'
print(f'scheme: {scheme}')

if run:
    laxis = {1:1, 2:1, 3:1, 4:0, 5:0}
    record_vals = {'mask_pre_inv':{'kwargs':{'resid_mask':True, 'mask':fq_mask.copy()}},
                   'mask_post_inv':{'kwargs':{'resid_mask':False, 'mask':fq_mask.copy()}},
                   'no_mask':{'kwargs':{'mask':np.ones((5,5),dtype=bool),
                                        'resid_mask':False}}}

    print('NPR basis')
    record_vals['NPR'] = {}
    b = bag_fits(bag_ensembles, obj='bag', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    bags_1 = [b.fit_operator(mu1, op, ens_list=[k for k in bag_ensembles if k!='C2'],
                             chiral_extrap=True, rotate=np.eye(len(b.operators)), plot=False)
            for op in b.operators]

    pvalues = [bag.pvalue for bag in bags_1]
    bags_1 = join_stats([bag[0] for bag in bags_1])
    s_b = sigma(norm='bag', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    sig_b = s_b.calc_running(mu1, mu2, chiral_extrap=True)
    bags_2 = sig_b@bags_1

    NPR_to_SUSY_stat = stat(val=NPR_to_SUSY, err=np.zeros(shape=NPR_to_SUSY.shape), btsp='fill')
    bags_1 = NPR_to_SUSY_stat@bags_1
    bags_2 = NPR_to_SUSY_stat@bags_2
    N_i = norm_factors(rotate=NPR_to_SUSY) 
    for op_idx in range(len(b.operators)):
        b1  = bags_1[op_idx]/N_i[op_idx]
        b1.pvalue = 0.0
        for j in range(len(b.operators)):
            if NPR_to_SUSY[op_idx, j]!=0:
                b1.pvalue += pvalues[j]
        if op_idx==2:
            b1.pvalue = b1.pvalue/2
        b2 = bags_2[op_idx]/N_i[op_idx]
        record_vals['NPR'][f'B{op_idx+1}'] = {mu1: b1, mu2: b2}


    del b
    r = bag_fits(bag_ensembles, obj='ratio', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    ratios_1 = [r.fit_operator(mu1, op, ens_list=[k for k in bag_ensembles if k!='C2'],
                               chiral_extrap=True, rotate=np.eye(5), plot=False)
                for op in r.operators]

    pvalues = [rat.pvalue for rat in ratios_1]
    one = stat(val=1, err=0, btsp='fill')
    ratios_1 = join_stats([one]+[rat[0] for rat in ratios_1])
    s_r = sigma(norm='11', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    sig_r = s_r.calc_running(mu1, mu2, chiral_extrap=True)
    ratios_2 = sig_r@ratios_1

    ratios_1 = NPR_to_SUSY_stat@ratios_1
    ratios_2 = NPR_to_SUSY_stat@ratios_2
    for op_idx in range(len(r.operators)):
        r1 = ratios_1[op_idx+1]
        r1.pvalue = 0.0
        for j in range(len(r.operators)):
            if NPR_to_SUSY[op_idx+1, j+1]!=0:
                r1.pvalue += pvalues[j]
        if op_idx==1:
            r1.pvalue = r1.pvalue/2
        r2 = ratios_2[op_idx+1]
        record_vals['NPR'][f'R{op_idx+2}'] = {mu1: r1, mu2: r2}

    del r




    for fit in list(record_vals.keys())[:-1]:
        print(fit)
        b = bag_fits(bag_ensembles, obj='bag', scheme=scheme, **record_vals[fit]['kwargs'])
        bags_1 = [b.fit_operator(mu1, op, ens_list=[k for k in bag_ensembles if k!='C2'],
                                    chiral_extrap=True, rotate=NPR_to_SUSY, plot=False)
                for op in b.operators]

        s_b = sigma(norm='bag', scheme=scheme, **record_vals[fit]['kwargs'])
        sig_b = s_b.calc_running(mu1, mu2, rotate=NPR_to_SUSY, chiral_extrap=True)
        bags_2 = sig_b@join_stats([bag[0] for bag in bags_1])

        N_i = norm_factors(rotate=NPR_to_SUSY) 
        for op_idx in range(len(b.operators)):
            b1  = bags_1[op_idx]/N_i[op_idx]
            b1.pvalue = bags_1[op_idx].pvalue
            b2 = bags_2[op_idx]/N_i[op_idx]
            record_vals[fit][f'B{op_idx+1}'] = {mu1: b1, mu2: b2}


        del b
        r = bag_fits(bag_ensembles, obj='ratio', scheme=scheme, **record_vals[fit]['kwargs'])
        ratios_1 = [r.fit_operator(mu1, op, ens_list=[k for k in bag_ensembles if k!='C2'],
                                    chiral_extrap=True, rotate=NPR_to_SUSY, plot=False)
                    for op in r.operators]

        for rat in ratios_1:
            del rat.mapping

        one = stat(val=1, err=0, btsp='fill')
        s_r = sigma(norm='11', scheme=scheme, **record_vals[fit]['kwargs'])
        sig_r = s_r.calc_running(mu1, mu2, rotate=NPR_to_SUSY, chiral_extrap=True)
        ratios_2 = sig_r@join_stats([one]+[rat[0] for rat in ratios_1])

        for op_idx in range(len(r.operators)):
            r1 = ratios_1[op_idx]
            r2 = ratios_2[op_idx+1]
            record_vals[fit][f'R{op_idx+2}'] = {mu1: r1, mu2: r2}

        del record_vals[fit]['kwargs'], r


    pickle.dump(record_vals, open('other_systematics_{scheme}.p', 'wb'))
else:
    record_vals = pickle.load(open('other_systematics_{scheme}.p', 'rb'))


quantities = {f'R{i+2}':r'$R_'+str(i+2)+r'$' for i in range(4)}
quantities.update({f'B{i+1}':r'$B_'+str(i+1)+r'$' for i in range(5)})

rv = [r'\begin{table}']
rv += [r'\caption{\label{tab:other_systematics} Values of bag and ratio '+\
       r'parameters at 3 GeV. Central value uses $Z$-factors with chirally '+\
       r'vanishing elements removed(masked) from $(P\Lambda)^T$ before '+\
       r'the inversion $Z = F((P\Lambda)^T)^{-1}$ '+\
       r'. We list the percent shift in the result by masking $Z$ post-inversion'+\
       r' and without masking at all. We also compare with performing the entire '+\
       r'analysis in the NPR basis and then rotating to the SUSY basis.}']
rv += [r'\begin{tabular}{c|c|c|c|c}']
rv += [r'\hline']
rv += [r'\hline']
rv += [r' & mask pre-inv & mask post-inv & no mask & NPR$\to$SUSY \\']
rv += [r' & central value & $\delta$ & $\delta$ & $\delta$ \\']
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

    central = record_vals['mask_pre_inv'][key][mu2]
    central_color = pval_color(record_vals['mask_pre_inv'][key][mu1].pvalue)
    central_str = r'\textcolor{'+central_color+r'}{'+err_disp(central.val, central.err)+r'}'

    post_inv = record_vals['mask_post_inv'][key][mu2]
    post_inv_color = pval_color(record_vals['mask_post_inv'][key][mu1].pvalue) 
    post_inv_change = np.abs(((post_inv-central)/((post_inv+central)*0.5)).val*100) 
    post_inv_change_str = r'\textcolor{'+post_inv_color+r'}{'+'{0:.3f}'.format(post_inv_change)+r'\%}'

    no_mask = record_vals['no_mask'][key][mu2]
    no_mask_color = pval_color(record_vals['no_mask'][key][mu1].pvalue) 
    no_mask_change = np.abs(((no_mask-central)/((no_mask+central)*0.5)).val*100) 
    no_mask_change_str = '{0:.2f}'.format(no_mask_change)+r'\%'
    no_mask_change_str = r'\textcolor{'+no_mask_color+r'}{'+'{0:.2f}'.format(no_mask_change)+r'\%}'

    NPR = record_vals['NPR'][key][mu2]
    NPR_color = pval_color(record_vals['NPR'][key][mu1].pvalue)
    NPR_change = np.abs(((NPR-central)/((NPR+central)*0.5)).val*100) 
    NPR_change_str = r'\textcolor{'+NPR_color+r'}{'+'{0:.2f}'.format(NPR_change)+r'\%}'

    rv += [' & '.join([name, 
                       central_str, 
                       post_inv_change_str,
                       no_mask_change_str,
                       NPR_change_str
                       ]) + r'\\']
    if key=='R5':
        rv += [r'\hline']

rv += [r'\hline']
rv += [r'\hline']
rv += [r'\end{tabular}']
rv += [r'\end{table}']

filename = f'/Users/rajnandinimukherjee/Desktop/draft_plots/tables_{fit_file}/other_systematics_{scheme}.tex'
f = open(filename, 'w')
f.write('\n'.join(rv))
f.close()
print(f'Z table output written to {filename}.')


