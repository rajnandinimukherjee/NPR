from cont_chir_extrap import *

expand_str = '_expanded' if expand_err else ''
fit_filename = f'other_systematics_{scheme}_{fit_file}{expand_str}.p'

mu1 = 2.0
mu2 = 3.0
run = bool(int(input('run?(0:False/1:True): ')))
print(f'Running other systematics in {scheme} scheme using data from {fit_file}')

if run:
    laxis = {1:1, 2:1, 3:1, 4:0, 5:0}
    record_vals = {'mask_pre_inv':{'kwargs':{'resid_mask':True,
                                             'mask':fq_mask.copy()}},
                   'mask_post_inv':{'kwargs':{'resid_mask':False,
                                              'mask':fq_mask.copy(),
                                              'run_extrap':True}},
                   'no_mask':{'kwargs':{'mask':np.ones((5,5),dtype=bool),
                                        'resid_mask':False,
                                        'run_extrap':True}}}

    print('NPR basis')
    record_vals['NPR'] = {}
    b = bag_fits(bag_ensembles, obj='bag', mask=fq_mask.copy(), resid_mask=False,
                 scheme=scheme, run_extrap=True)
    bags_1 = [b.fit_operator(mu1, op, 
                             ens_list=[k for k in bag_ensembles if k!='C2'],
                             log=True, addnl_terms='del_ms',
                             expand_err=expand_err,
                             guess=[1,1e-1,1e-2,1e-2],
                             chiral_extrap=True,
                             rotate=np.eye(len(b.operators)),
                             plot=False)
            for op in b.operators]

    pvalues = [bag.pvalue for bag in bags_1]
    bags_1 = join_stats([bag[0] for bag in bags_1])
    s_b = sigma(norm='bag', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    sig_b = s_b.calc_running(mu1, mu2, chi_sq_rescale=True, chiral_extrap=True)
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
    ratios_1 = [r.fit_operator(mu1, op,
                               ens_list=[k for k in bag_ensembles if k!='C2'],
                               log=True, addnl_terms='del_ms',
                               expand_err=expand_err,
                               guess=[1,1e-1,1e-2,1e-2],
                               chiral_extrap=True,
                               rotate=np.eye(5),
                               plot=False)
                for op in r.operators]

    pvalues = [rat.pvalue for rat in ratios_1]
    one = stat(val=1, err=0, btsp='fill')
    ratios_1 = join_stats([one]+[rat[0] for rat in ratios_1])
    s_r = sigma(norm='11', mask=fq_mask.copy(), resid_mask=False, scheme=scheme)
    sig_r = s_r.calc_running(mu1, mu2, chi_sq_rescale=True, chiral_extrap=True)
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
        bags_1 = [b.fit_operator(mu1, op, 
                                 ens_list=[k for k in bag_ensembles if k!='C2'],
                                 log=True, addnl_terms='del_ms',
                                 guess=[1,1e-1,1e-2,1e-2],
                                 chiral_extrap=True, rotate=NPR_to_SUSY,
                                 expand_err=expand_err,
                                 plot=False)
                for op in b.operators]

        s_b = sigma(norm='bag', scheme=scheme, **record_vals[fit]['kwargs'])
        sig_b = s_b.calc_running(mu1, mu2, rotate=NPR_to_SUSY,
                                 chiral_extrap=True, chi_sq_rescale=True)
        bags_2 = sig_b@join_stats([bag[0] for bag in bags_1])

        N_i = norm_factors(rotate=NPR_to_SUSY) 
        for op_idx in range(len(b.operators)):
            b1  = bags_1[op_idx]/N_i[op_idx]
            b1.pvalue = bags_1[op_idx].pvalue
            b2 = bags_2[op_idx]/N_i[op_idx]
            record_vals[fit][f'B{op_idx+1}'] = {mu1: b1, mu2: b2}


        del b
        r = bag_fits(bag_ensembles, obj='ratio', scheme=scheme, **record_vals[fit]['kwargs'])
        ratios_1 = [r.fit_operator(mu1, op, 
                                   ens_list=[k for k in bag_ensembles if k!='C2'],
                                   log=True, addnl_terms='del_ms',
                                   guess=[1,1e-1,1e-2,1e-2],
                                   expand_err=expand_err,
                                   chiral_extrap=True, rotate=NPR_to_SUSY,
                                   plot=False)
                    for op in r.operators]

        for rat in ratios_1:
            del rat.mapping

        one = stat(val=1, err=0, btsp='fill')
        s_r = sigma(norm='11', scheme=scheme, **record_vals[fit]['kwargs'])
        sig_r = s_r.calc_running(mu1, mu2, rotate=NPR_to_SUSY,
                                 chiral_extrap=True, chi_sq_rescale=True)
        ratios_2 = sig_r@join_stats([one]+[rat[0] for rat in ratios_1])

        for op_idx in range(len(r.operators)):
            r1 = ratios_1[op_idx]
            r2 = ratios_2[op_idx+1]
            record_vals[fit][f'R{op_idx+2}'] = {mu1: r1, mu2: r2}

        del record_vals[fit]['kwargs'], r


    pickle.dump(record_vals, open(fit_filename, 'wb'))
else:
    record_vals = pickle.load(open(fit_filename, 'rb'))
