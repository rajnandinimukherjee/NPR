from cont_chir_extrap import *
mu_list = [2.0, 2.2, 2.3, 2.4]
flag_mus = [2.0, 3.0, 3.0, 3.0, 3.0]
flag_vals = [0.5570, 0.502, 0.766, 0.926, 0.720]
flag_errs = [0.0071, 0.014, 0.032, 0.019, 0.038]

basis = str(input('Choose basis (SUSY/NPR): '))
run = input('Generate new plots? (True:1/False:0): ')

Z = Z_fits(bag_ensembles, bag=True)
sigma_file = 'sigmas_SUSY.p' if basis == 'SUSY' else 'sigmas_NPR.p'
Z.store = pickle.load(open(sigma_file, 'rb'))


def convert_to_MSbar(bag, mu2, mu1, **kwargs):
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

                sigma.val = sigma_temp.val@sigma.val
                sigma.btsp = np.array([sigma_temp.btsp[k,]@sigma.btsp[k,]
                                       for k in range(N_boot)])
            # print(sig_str)
        except KeyError:
            rot_mtx = NPR_to_SUSY if basis == 'SUSY'\
                else np.eye(len(operators))
            sigma = z.extrap_sigma(mu2, mu1, rotate=rot_mtx)

    R_conv = R_RISMOM_MSbar(mu2)
    if basis == 'SUSY':
        R_conv = NPR_to_SUSY@R_conv@np.linalg.inv(NPR_to_SUSY)

    bag_MS = stat(
        val=R_conv@sigma.val@bag.val,
        err='fill',
        btsp=np.array([R_conv@sigma.btsp[k,]@bag.btsp[k,]
                       for k in range(N_boot)])
    )
    return bag_MS


def rotate_to_SUSY(**kwargs):
    NPR_fits = pickle.load(open('cc_extrap_dict_NPR.p', 'rb'))

    from_NPR = {op: {fit: {mu: {} for mu in mu_list}
                     for fit in NPR_fits['VVpAA'].keys()}
                for op in operators}
    for fit in NPR_fits['VVpAA'].keys():
        for mu in mu_list:
            NPR_MS_Bs = stat(
                val=[NPR_fits[op][fit][mu]['MS']['val']
                     for op in operators],
                btsp=np.array([NPR_fits[op][fit][mu]['MS']['btsp']
                               for op in operators]).T
            )
            NPR_MS_Bs = stat(
                val=NPR_to_SUSY@NPR_MS_Bs.val,
                err='fill',
                btsp=np.array([NPR_to_SUSY@NPR_MS_Bs.btsp[k, :]
                               for k in range(N_boot)])
            )
            for op_idx, op in enumerate(operators):
                from_NPR[op][fit][mu] = {
                    'val': NPR_MS_Bs.val[op_idx],
                    'err': NPR_MS_Bs.err[op_idx],
                    'btsp': NPR_MS_Bs.btsp[:, op_idx],
                    'GOF': True}

            for op_idx, j in itertools.product(range(len(operators)),
                                               range(len(operators))):
                if NPR_to_SUSY[op_idx, j] != 0 and \
                        NPR_fits[operators[j]][fit][mu]['pvalue'] < 0.05:
                    from_NPR[operators[op_idx]][fit][mu]['GOF'] = False

    return from_NPR


fits_NPR = rotate_to_SUSY()


def operator_summary(operator, basis='NPR', filename=None,
                     open=True, with_alt=True, **kwargs):
    op_idx = operators.index(operator)
    ansatze = list(fits[operator].keys())
    ansatz_desc = [fits[operator][a]['kwargs']['title']
                   for a in ansatze]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4),
                           sharey=False)
    plt.subplots_adjust(wspace=0)

    for m, mu in enumerate(mu_list):
        offset = m*0.2

        vals = [fits[operator][fit][mu]['phys'] for fit in ansatze]
        errs = [fits[operator][fit][mu]['err'] for fit in ansatze]
        pvals = [fits[operator][fit][mu]['pvalue'] for fit in ansatze]
        cont_slope = [fits[operator][fit][mu]['coeffs'][0] for fit in ansatze]

        MS_vals = [fits[operator][fit][mu]['MS']['val'] for fit in ansatze]
        MS_errs = [fits[operator][fit][mu]['MS']['err'] for fit in ansatze]

        ax[0].errorbar(np.arange(len(ansatze))+offset, vals,
                       yerr=errs, fmt='o', capsize=2,
                       label=r'$\mu='+str(np.around(mu, 2)) + '$ GeV')
        ax[1].errorbar(np.arange(len(ansatze))+offset, MS_vals, yerr=MS_errs,
                       fmt='o', capsize=2)

        if basis == 'SUSY' and with_alt:
            alt_MS_vals = [fits_NPR[operator][fit][mu]['val']
                           for fit in ansatze]
            alt_MS_errs = [fits_NPR[operator][fit][mu]['err']
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
            if basis == 'SUSY' and with_alt and not alt_MS_GOF[i]:
                ax[1].annotate(r'$\times $', (i+offset+0.1, alt_MS_vals[i]),
                               ha='center', va='center', c='white')

    ax[0].set_title(r'$B_{phys}^{SMOM}(\mu)$')
    ax[0].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')

    if basis == 'SUSY':
        ansatze.append('FLAG')
        ansatz_desc.append(r'FLAG $N_f=2+1$')
        op_idx = operators.index(operator)
        FLAG_val, FLAG_err = flag_vals[op_idx], flag_errs[op_idx]
        ax[1].errorbar([len(ansatze)-1], [FLAG_val], yerr=[FLAG_err],
                       fmt='o', capsize=2, c='k')

    ax[1].set_title(r'$B_{phys}^{\overline{MS}}(' +
                    str(np.around(flag_mus[op_idx], 2))+'$ GeV)')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')
    ax[0].legend()
    if filename == None:
        filename = f'{operator}/{basis}/fit_summary.pdf'

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


if bool(int(run)):
    fits = {}
    b = bag_fits(bag_ensembles)

    for op in operators:
        fits[op] = {}
        for key in ['a2m2', 'a2m2noC', 'a2a4m2', 'a2m2m4']:  # , 'a2m2logm2']:
            fits[op][key] = {mu: {'filename': op+'/'+basis+'/'+key+'_'+str(
                int(mu*10))+'.pdf'}
                for mu in mu_list}

        fits[op]['a2m2']['kwargs'] = {'title': r'$a^2$, $m_\pi^2$'}
        fits[op]['a2m2noC']['kwargs'] = {'title': r'$a^2$, $m_\pi^2$ (no C)',
                                         'ens_list': ['M0', 'M1', 'M2',
                                                      'M3', 'F1M']}
        guess = [1e-1, 1e-2, 1e-3, 1e-3]
        fits[op]['a2a4m2']['kwargs'] = {'title': r'$a^2$, $a^4$, $m_\pi^2$',
                                        'guess': guess, 'addnl_terms': 'a4'}
        fits[op]['a2m2m4']['kwargs'] = {'title': r'$a^2$, $m_\pi^2$, ' +
                                        r'$m_\pi^4$',
                                        'guess': guess, 'addnl_terms': 'm4'}
        # fits[op]['a2m2logm2']['kwargs'] = {'title': r'$a^2$, $m_\pi^2$, ' +
        #                                   r'$\log(m_\pi^2/\Lambda^2)$',
        #                                   'addnl_terms': 'log'}

    for i, op in enumerate(operators):
        print(f'Generating plots for operator B{i+1}')
        for fit in fits[op].keys():
            kwargs = fits[op][fit]['kwargs']
            if basis == 'SUSY':
                kwargs['rotate'] = NPR_to_SUSY
            for mu in mu_list:
                filename = fits[op][fit][mu]['filename']
                phys, coeffs, chi_sq_dof, pvalue = b.plot_fits(
                    mu, ops=[op], filename=filename, **kwargs)
                fits[op][fit][mu].update({'phys': phys.val,
                                          'err': phys.err,
                                          'btsp': phys.btsp,
                                          'coeffs': coeffs.val,
                                          'coeffs_err': coeffs.err,
                                          'disp': err_disp(phys.val, phys.err),
                                          'coeff_disp': [err_disp(coeffs.val[0],
                                                                  coeffs.err[0]),
                                                         err_disp(coeffs.val[1],
                                                                  coeffs.err[1])],
                                          'chi_sq_dof': chi_sq_dof,
                                          'pvalue': pvalue})

    bar = Bar('Converting to MS-bar',
              max=len(operators)*len(mu_list)*len(list(fits['VVpAA'].keys())))
    for fit in fits['VVpAA'].keys():
        for mu in mu_list:
            bag = stat(
                val=np.array([fits[op][fit][mu]['phys']
                              for op in operators]),
                err=np.array([fits[op][fit][mu]['err']
                              for op in operators]),
                btsp=np.array([fits[op][fit][mu]['btsp']
                               for op in operators]).T,
            )
            for op_idx, op in enumerate(operators):
                MS = convert_to_MSbar(bag, flag_mus[op_idx], mu)
                fits[op][fit][mu].update({'MS': {'val': MS.val[op_idx],
                                                 'err': MS.err[op_idx],
                                                 'btsp': MS.btsp[:, op_idx]}})
                bar.next()
    bar.finish()

    for op in operators:
        operator_summary(op, basis=basis, open=False)

    pickle.dump(fits, open('cc_extrap_dict_'+basis+'.p', 'wb'))
else:
    fits = pickle.load(open('cc_extrap_dict_'+basis+'.p', 'rb'))


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
           basis+'/fit_summary.pdf}']
    rv += [r'\caption{$B_{'+str(i+1)+r'}$\\(left) $B_{phys}$' +
           r' in RI/SMOM scheme from fit variations ' +
           r'(fits with $p$-value $<0.05$ marked with ``$\times$"). \\' +
           r'(right) $B_{phys}$ in $\overline{MS}$ computed using ' +
           r'$B^{\overline{MS}} = R^{\overline{MS}\leftarrow SMOM}('+mu_str+')' +
           r'\sigma_{npt}('+mu_str+r',\mu) B^{SMOM}(\mu)$.}']
    rv += [r'\end{figure}']
    rv += [r'\clearpage']

for i, op in enumerate(operators):
    ansatze = [fits[op][fit]['kwargs']['title'] for fit in fits[op].keys()]
    rv += [r'\section{$B_'+str(i+1)+r'$}']
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
    rv += [r'\caption{Fit values of coefficients in $B = B_{phys} + \mathbf{\alpha}'
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

filename = f'extrap_{basis}'
f = open('tex/'+filename+'.tex', 'w')
f.write('\n'.join(rv))
f.close()

os.system(f"pdflatex tex/{filename}.tex")
os.system(f"open {filename}.pdf")
