from cont_chir_extrap import *

Z = Z_fits(bag_ensembles, bag=True)
mu_list = [2.0, 2.2, 2.3, 2.4]
ansatz_kwargs = {
    'a2m2': {'title': r'$a^2$, $m_\pi^2$'},
    'a2m2noC': {'title': r'$a^2$, $m_\pi^2$ (no C)',
                'ens_list': ['M0', 'M1', 'M2', 'M3', 'F1M']},
    'a2a4m2': {'title': r'$a^2$, $a^4$, $m_\pi^2$',
               'guess': [1, 1e-1, 1e-2, 1e-3], 'addnl_terms': 'a4'},
    'a2m2mcut': {'title': r'$a^2$, $m_\pi^2$ (no M3, C2)',
                 'ens_list': ['C0', 'C1', 'M0', 'M1', 'M2', 'F1M']},
    'a2m2m4': {'title': r'$a^2$, $m_\pi^2$, $m_\pi^4$',
               'guess': [1, 1e-1, 1e-2, 1e-3], 'addnl_terms': 'm4'},
    'a2m2logm2': {'title': r'$a^2$, $m_\pi^2$, $\log(m_\pi^2/\Lambda^2)$',
                  'addnl_terms': 'log'}
}


def basis_rotation(basis):
    if basis == 'SUSY':
        return NPR_to_SUSY
    else:
        return np.eye(len(operators))


def convert_to_MSbar(B_N, mu2, mu1, rot_mtx=np.eye(len(operators)),
                     obj='bag', **kwargs):
    if mu1 == mu2:
        sigma = stat(
            val=np.eye(len(operators)),
            btsp=np.array([np.eye(len(operators))
                           for k in range(N_boot)])
        )
    else:
        try:
            sign = np.sign(mu2-mu1)
            mus_temp = np.around(np.arange(int(mu1*10),
                                           int(mu2*10)+sign,
                                           sign)*0.1, 1)
            sigma = stat(
                val=np.eye(len(operators)),
                btsp=np.array([np.eye(len(operators))
                               for k in range(N_boot)])
            )
            sig_str = f'B({mu1})'
            for m in range(1, len(mus_temp)):
                mu2_temp, mu1_temp = mus_temp[m], mus_temp[m-1]
                sig_str = f'sig({mu2_temp},{mu1_temp})' + sig_str
                sigma_temp = Z.store[(mu2_temp, mu1_temp)]

                sigma = sigma_temp@sigma
                # sigma.val = sigma_temp.val@sigma.val
                # sigma.btsp = np.array([sigma_temp.btsp[k,]@sigma.btsp[k,]
                #                       for k in range(N_boot)])
            # print(sig_str)
        except KeyError:
            sigma = z.extrap_sigma(mu2, mu1, rotate=rot_mtx)

    R_conv = stat(
        val=rot_mtx@R_RISMOM_MSbar(
            mu2, obj=obj, **kwargs)@np.linalg.inv(rot_mtx),
        btsp='fill')
    B_N_MS = R_conv@sigma@B_N
    return B_N_MS


def rotate_from_NPR(include_C=False, C_folder='without_C',
                    obj='bag', rot_mtx=np.eye(len(operators)),
                    **kwargs):
    NPR_fits = pickle.load(
        open(f'sigmas/{C_folder}/cc_extrap_dict_{obj}_NPR.p', 'rb'))

    from_NPR = {op: {fit: {mu: {} for mu in mu_list}
                     for fit in NPR_fits['VVpAA'].keys()}
                for op in operators}
    for fit in NPR_fits['VVpAA'].keys():
        for mu in mu_list:
            NPR_MS_Bs = stat(
                val=[NPR_fits[op][fit][mu]['MS'].val
                     for op in operators],
                btsp=np.array([NPR_fits[op][fit][mu]['MS'].btsp
                               for op in operators]).T
            )
            NPR_MS_Bs = stat(
                val=rot_mtx@NPR_MS_Bs.val,
                err='fill',
                btsp=np.array([rot_mtx@NPR_MS_Bs.btsp[k, :]
                               for k in range(N_boot)])
            )
            for op_idx, op in enumerate(operators):
                from_NPR[op][fit][mu] = {
                    'MS': stat(
                        val=NPR_MS_Bs.val[op_idx],
                        err=NPR_MS_Bs.err[op_idx],
                        btsp=NPR_MS_Bs.btsp[:, op_idx]),
                    'GOF': True}

            for op_idx, j in itertools.product(range(len(operators)),
                                               range(len(operators))):
                if rot_mtx[op_idx, j] != 0 and \
                        NPR_fits[operators[j]][fit][mu]['pvalue'] < 0.05:
                    from_NPR[operators[op_idx]][fit][mu]['GOF'] = False

    return from_NPR


def operator_summary(operator, fits, basis='NPR', filename=None,
                     open=True, with_FLAG=False, with_alt=True,
                     save=True, obj='bag', rot_mtx=np.eye(len(operators)),
                     **kwargs):

    if basis != 'NPR' and with_alt:
        fits_NPR = rotate_from_NPR(rot_mtx=rot_mtx, obj=obj, **kwargs)

    op_idx = operators.index(operator)
    ansatze = list(fits[operator].keys())
    ansatz_desc = [ansatz_kwargs[a]['title'] for a in ansatze]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4),
                           sharey=False)
    plt.subplots_adjust(wspace=0)

    for m, mu in enumerate(mu_list):
        offset = m*0.2

        vals = [fits[operator][fit][mu]['phys'].val for fit in ansatze]
        errs = [fits[operator][fit][mu]['phys'].err for fit in ansatze]
        pvals = [fits[operator][fit][mu]['pvalue'] for fit in ansatze]

        MS_vals = [fits[operator][fit][mu]['MS'].val for fit in ansatze]
        MS_errs = [fits[operator][fit][mu]['MS'].err for fit in ansatze]

        ax[0].errorbar(np.arange(len(ansatze))+offset, vals,
                       yerr=errs, fmt='o', capsize=2,
                       label=r'$\mu='+str(np.around(mu, 2)) + '$ GeV')
        ax[1].errorbar(np.arange(len(ansatze))+offset, MS_vals, yerr=MS_errs,
                       fmt='o', capsize=2)

        if basis != 'NPR' and with_alt:
            alt_MS_vals = [fits_NPR[operator][fit][mu]['MS'].val
                           for fit in ansatze]
            alt_MS_errs = [fits_NPR[operator][fit][mu]['MS'].err
                           for fit in ansatze]
            alt_MS_GOF = [fits_NPR[operator][fit][mu]['GOF']
                          for fit in ansatze]

            ax[1].errorbar(np.arange(len(ansatze))+offset+0.1, alt_MS_vals,
                           yerr=alt_MS_errs, fmt='d', capsize=2,
                           color=color_list[m])

        for i in range(len(ansatze)):
            if pvals[i] < 0.05:
                ax[0].annotate(r'$\times $', (i+offset, vals[i]), ha='center',
                               va='center', c='white')
                ax[1].annotate(r'$\times $', (i+offset, MS_vals[i]), ha='center',
                               va='center', c='white')
            if basis != 'NPR' and with_alt and not alt_MS_GOF[i]:
                ax[1].annotate(r'$\times $', (i+offset+0.1, alt_MS_vals[i]),
                               ha='center', va='center', c='white')

    ax[0].set_title(r'$\mathcal{B}_{'+str(op_idx+1)+',\, phys}^{SMOM}(\mu)$')
    ax[0].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')

    if basis != 'NPR' and with_FLAG:
        ansatze.append('FLAG')
        ansatz_desc.append(r'FLAG $N_f=2+1$')
        op_idx = operators.index(operator)
        FLAG_val, FLAG_err = flag_vals[op_idx], flag_errs[op_idx]
        ax[1].errorbar([len(ansatze)-1], [FLAG_val], yerr=[FLAG_err],
                       fmt='o', capsize=2, c='k')

    if obj == 'bag':
        char = r'\mathcal{B}'
    elif obj == 'fq_op':
        char = r'\langle O \rangle'

    ax[1].set_title(r'$'+char+r'_{'+str(op_idx+1)+',\, phys}^{\overline{MS}}(' +
                    str(np.around(flag_mus[op_idx], 2))+'$ GeV)')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')
    ax[0].legend()
    if filename == None:
        filename = f'{operator}/{basis}/fit_summary_{obj}.pdf'

    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')

    if open:
        print(f'Plot saved to {filename}.')
        os.system("open "+filename)


def full_summary(basis='SUSY', run=False, include_C=False,
                 with_FLAG=False, calc_running=False,
                 obj='bag', **kwargs):
    rot_mtx = basis_rotation(basis)
    C_folder = 'with_C' if include_C else 'without_C'

    if calc_running:
        Z.store = {}
        mus = np.around(np.arange(16, 31, 1)*0.1, 1)
        N_mus = len(mus)

        for i in tqdm(range(1, N_mus)):
            Z.store[(mus[i], mus[i-1])] = Z.extrap_sigma(
                mus[i], mus[i-1], rotate=rot_mtx,
                include_C=include_C)

            Z.store[(mus[i-1], mus[i])] = Z.extrap_sigma(
                mus[i-1], mus[i], rotate=rot_mtx,
                include_C=include_C)

        pickle.dump(Z.store, open(f'sigmas/{C_folder}/sigmas_{basis}.p', 'wb'))
        print(
            f'Saved npt running matrices to sigmas/{C_folder}/sigmas_{basis}.p')
    else:
        sigma_file = f'sigmas/{C_folder}/sigmas_{basis}.p'
        Z.store = pickle.load(open(sigma_file, 'rb'))

    dict_filename = f'sigmas/{C_folder}/cc_extrap_dict_{obj}_{basis}.p'
    if run:
        fits = {}
        b = bag_fits(bag_ensembles, obj=obj)

        for op in tqdm(operators, desc=f'{obj} fits'):
            op_idx = operators.index(op)
            fits[op] = {}
            for fit in list(ansatz_kwargs.keys())[:-1]:
                fits[op][fit] = {mu: {'filename': op+'/'+basis+'/'+fit+'_' +
                                      str(int(mu*10))+'.pdf'} for mu in mu_list}
                kwargs = ansatz_kwargs[fit]
                kwargs['rotate'] = rot_mtx
                for mu in mu_list:
                    filename = fits[op][fit][mu]['filename']
                    phys, coeffs, chi_sq_dof, pvalue = b.plot_fits(
                        mu, ops=[op], filename=filename, **kwargs)

                    fits[op][fit][mu].update({'phys': phys,
                                              'coeffs': coeffs,
                                              'disp': err_disp(phys.val, phys.err),
                                              'coeff_disp': [err_disp(coeffs.val[0],
                                                                      coeffs.err[0]),
                                                             err_disp(coeffs.val[1],
                                                                      coeffs.err[1])],
                                              'chi_sq_dof': chi_sq_dof,
                                              'pvalue': pvalue})

        for fit in fits['VVpAA'].keys():
            for mu in mu_list:
                obj_ = stat(
                    val=[fits[op][fit][mu]['phys'].val
                         for op in operators],
                    err=[fits[op][fit][mu]['phys'].err
                         for op in operators],
                    btsp=np.array([fits[op][fit][mu]['phys'].btsp
                                   for op in operators]).T,
                )
                for op_idx, op in enumerate(operators):
                    MS = convert_to_MSbar(
                        obj_, flag_mus[op_idx], mu, obj=obj, rot_mtx=rot_mtx)
                    fits[op][fit][mu].update({'MS': stat(
                        val=MS.val[op_idx],
                        err=MS.err[op_idx],
                        btsp=MS.btsp[:, op_idx]
                    )})

        pickle.dump(fits, open(dict_filename, 'wb'))
        print(f'Saved fit dict to {dict_filename}.')
    else:
        print(f'Loaded fit dict from {dict_filename}.')
        fits = pickle.load(open(dict_filename, 'rb'))

    for op in operators:
        operator_summary(op, fits, obj=obj, basis=basis, open=False, with_alt=True,
                         with_FLAG=with_FLAG, rot_mtx=rot_mtx, C_folder=C_folder)

    if obj == 'bag':
        char = r'\mathcal{B}'
    elif obj == 'fq_op':
        char = r'\langle O \rangle'

    rv = [r'\documentclass[12pt]{extarticle}']
    rv += [r'\usepackage[paperwidth=15in,paperheight=7.2in]{geometry}']
    rv += [r'\usepackage{amsmath}']
    rv += [r'\usepackage{hyperref}']
    rv += [r'\usepackage{multirow}']
    rv += [r'\usepackage{pdfpages}']
    rv += [r'\usepackage[utf8]{inputenc}']
    rv += [r'\title{Kaon mixing: chiral and continuum extrapolations}']
    rv += [r'\author{R Mukherjee}'+'\n'+r'\date{\today}']
    rv += [r'\begin{document}']
    rv += [r'\maketitle']
    rv += [r'\tableofcontents']
    rv += [r'\clearpage']

    for i, op in enumerate(operators):
        mu_str = str(np.around(flag_mus[i]))
        rv += [r'\begin{figure}']
        rv += [r'\centering']
        rv += [r'\includegraphics[page=1, width=1.1\textwidth]{'+op+r'/' +
               basis+'/fit_summary_'+obj+r'.pdf}']
        rv += [r'\caption{$'+char+r'_{'+str(i+1)+r'}$\\(left) $'+char+r'_{phys}$' +
               r' in RI/SMOM scheme from fit variations ' +
               r'(fits with $p$-value $<0.05$ marked with ``$\times$"). \\' +
               r'(right) $'+char+r'_{phys}$ in $\overline{MS}$ computed using ' +
               r'$'+char+r'^{\overline{MS}} = R^{\overline{MS}\leftarrow SMOM}('+mu_str+')' +
               r'\sigma_{npt}('+mu_str+r',\mu) '+char+r'^{SMOM}(\mu)$.}']
        rv += [r'\end{figure}']
        rv += [r'\clearpage']

    for i, op in enumerate(operators):
        ansatze = [ansatz_kwargs[fit]['title'] for fit in fits[op].keys()]
        rv += [r'\section{$\mathcal{B}_'+str(i+1)+r'$}']
        rv += [r'\begin{table}[h!]']
        rv += [r'\begin{center}']
        rv += [r'\begin{tabular}{|' +
               ''.join(['c|']*(len(ansatze)+1))+'}']
        rv += [r'\hline']
        rv += [r'$\mu$ (GeV) & '+'& '.join(ansatze)+r'\\']
        rv += [r'\hline']
        for mu in mu_list:
            chi_sqs = [r'& \hyperlink{'+fits[op][fit][mu]['filename']+r'.1}{\textbf{'
                       + fits[op][fit][mu]['disp']+r'}: ' +
                       str(np.around(fits[op][fit][mu]['chi_sq_dof'], 3)) +
                       r' ('+str(np.around(fits[op][fit][mu]['pvalue'], 3)) +
                       r')}' for fit in fits[op].keys()]
            rv += [str(mu)+' '.join(chi_sqs)+r'\\']
        rv += [r'\hline']
        rv += [r'\end{tabular}']
        rv += [r'\caption{Physical point value from chiral and ' +
               r'continuum extrapolation at renormalisation scale $\mu$. ' +
               r'Entries are \textbf{value(error)}: ' +
               r'$\chi^2/\text{DOF}$ ($p$-value).}']
        rv += [r'\end{center}']
        rv += [r'\end{table}']

        rv += [r'\begin{table}[h!]']
        rv += [r'\begin{center}']
        rv += [r'\begin{tabular}{|c c|' +
               ''.join(['c|']*(len(ansatze)))+'}']
        rv += [r'\hline']
        rv += [r'$\mu$ (GeV) &  & '+'& '.join(ansatze)+r'\\']
        rv += [r'\hline']
        for mu in mu_list:
            rv += [r'\multirow{2}{0.5in}{'+str(mu)+r'} & $\alpha$ & ' + '& '.join(
                [fits[op][fit][mu]['coeff_disp'][0] for fit in fits[op].keys()])+r'\\']
            rv += [r' & $\beta$ & ' + '& '.join([fits[op][fit][mu][
                'coeff_disp'][1] for fit in fits[op].keys()])+r'\\']
            rv += [r'\hline']
        rv += [r'\end{tabular}']
        rv += [r'\caption{Fit values of coefficients in $Q = Q_{phys} + \mathbf{\alpha}'
               + r' a^2 + \mathbf{\beta}\left(\frac{m_\pi^2}{f_\pi^2}-'
               + r'\frac{m_{\pi,PDG}^2}{f_\pi^2}\right) + \ldots$.}']
        rv += [r'\end{center}']
        rv += [r'\end{table}']

        for fit in fits[op].keys():
            for mu in mu_list:
                filename = fits[op][fit][mu]['filename']
                rv += [r'\includepdf[link, pages=-]{'+filename+'}']
        rv += [r'\clearpage']

    rv += [r'\end{document}']

    filename = f'extrap_{obj}_{basis}'
    f = open('tex/'+filename+'.tex', 'w')
    f.write('\n'.join(rv))
    f.close()

    os.system(f"pdflatex tex/{filename}.tex")
    os.system(f"open {filename}.pdf")
