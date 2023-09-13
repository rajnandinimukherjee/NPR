from cont_chir_extrap import *
b = bag_fits(bag_ensembles)
F1M = Z_analysis('F1M', bag=True)
mu_list = [2.4, 2.0, 1.8, 1.5]
MS_bar_mu = 3.0
flag_mus = [2.0, 3.0, 3.0, 3.0, 3.0]
flag_vals = [0.524*F1M.scale_evolve(3.0, 2.0)[0][0, 0],
             0.46, 0.79, 0.78, 0.49]
flag_errs = [0.025, 0.04, 0.07, 0.06, 0.06]
bag_signs = np.diag([1, -1, 1, -1, -1])


def convert_to_MSbar(bags, bags_err, bag_btsp, mu1, mu2, **kwargs):
    F1M_Z = Z_analysis('F1M', bag=True)
    sig, sig_err, sig_btsp = F1M_Z.scale_evolve(mu2, mu1)
    R_conv = R_RISMOM_MSbar(mu2)
    bag_MS = R_conv@sig@bag
    bag_MS_btsp = np.array([R_conv@sig_btsp[k,]@bag_btsp[:, k]
                            for k in range(bag_analysis.N_boot)])
    bag_MS_err = np.array([st_dev(bag_MS_btsp[:, i], bag_MS[i])
                           for i in range(len(bag_MS))])
    return bag_MS, bag_MS_err, bag_MS_btsp


def operator_summary(operator, mu=2.0, filename=None,
                     open=True, FLAG=False, **kwargs):
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
        chis = [fits[operator][fit][mu]['chi_sq_dof'] for fit in ansatze]
        pvals = [fits[operator][fit][mu]['pvalue'] for fit in ansatze]
        cont_slope = [fits[operator][fit][mu]['coeffs'][0] for fit in ansatze]

        MS_vals = [fits[operator][fit][mu]['MS']['val'] for fit in ansatze]
        MS_errs = [fits[operator][fit][mu]['MS']['err'] for fit in ansatze]
        pvals = [fits[operator][fit][mu]['pvalue'] for fit in ansatze]

        ax[0].errorbar(np.arange(len(ansatze))+offset, vals,
                       yerr=errs, fmt='o', capsize=2)
        ax[1].errorbar(np.arange(len(ansatze))+offset, MS_vals, yerr=MS_errs,
                       fmt='o', capsize=2, label=r'$\mu='+str(np.around(mu, 2))
                       + '$ GeV')

        tick = min(errs)*0.1
        for i in range(len(ansatze)):
            # ax[0].annotate(str(np.around(chis[i], 2)),
            #               (i, vals[i]+errs[i]+tick),
            #               ha='center', va='bottom')
            # ax[0].annotate(str(np.around(cont_slope[i], 2)),
            #               (i+offset, vals[i]-errs[i]-4*tick), fontsize=9,
            #               ha='center', va='top')
            if pvals[i] < 0.05:
                ax[0].annotate(r'$\times $', (i+offset, vals[i]), ha='center',
                               va='center', c='white')
                ax[1].annotate(r'$\times $', (i+offset, MS_vals[i]), ha='center',
                               va='center', c='white')

    ax[0].set_title(r'$B_{phys}^{SMOM}(\mu)$')
    ax[0].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')

    if FLAG:
        ansatze.append('FLAG')
        ansatz_desc.append(r'FLAG $N_f=2+1+1$')
        op_idx = operators.index(operator)
        FLAG_val, FLAG_err = flag_vals[op_idx], flag_errs[op_idx]
        ax[1].errorbar([len(ansatze)-1], [FLAG_val], yerr=[FLAG_err],
                       fmt='o', capsize=2, c='k', label=r'$\mu=' +
                       str(np.around(flag_mus[op_idx]))+r'$ GeV')

    ax[1].set_title(r'$B_{phys}^{\overline{MS}}(\mu\to 3.0$ GeV)')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
                     ha='right')
    ax[1].legend()
    if filename == None:
        filename = f'plots/{operator}_fit_summary.pdf'

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


run = input('Generate new plots? (True:1/False:0): ')
if bool(int(run)):
    basis = str(input('Choose basis (SUSY/NPR): '))
    fits = {}
    for op in operators:
        fits[op] = {}
        for key in ['a2m2', 'a2m2noC', 'a2a4m2', 'a2m2m4', 'a2m2logm2']:
            fits[op][key] = {mu: {'filename': op+'/'+key+'_'+str(
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
        fits[op]['a2m2logm2']['kwargs'] = {'title': r'$a^2$, $m_\pi^2$, ' +
                                           r'$\log(m_\pi^2/\Lambda^2)$',
                                           'addnl_terms': 'log'}

    for i, op in enumerate(operators):
        print(f'Generating plots for operator B{i+1}')
        for fit in fits[op].keys():
            kwargs = fits[op][fit]['kwargs']
            if basis == 'SUSY':
                kwargs['rotate'] = NPR_to_SUSY
            for mu in mu_list:
                filename = fits[op][fit][mu]['filename']
                phys, err, btsp, coeffs, coeffs_err, chi_sq_dof, pvalue = \
                    b.plot_fits(mu, ops=[op], filename=filename, **kwargs)
                fits[op][fit][mu].update({'phys': phys,
                                          'err': err,
                                          'btsp': btsp,
                                          'coeffs': coeffs,
                                          'coeffs_err': coeffs_err,
                                          'disp': err_disp(phys, err),
                                          'coeff_disp': [err_disp(coeffs[0],
                                                                  coeffs_err[0]),
                                                         err_disp(coeffs[1],
                                                                  coeffs_err[1])],
                                          'chi_sq_dof': chi_sq_dof,
                                          'pvalue': pvalue})

    for fit in fits['VVpAA'].keys():
        for mu in mu_list:
            bag = np.array([fits[op][fit][mu]['phys'] for op in operators])
            bag_err = np.array([fits[op][fit][mu]['err'] for op in operators])
            bag_btsp = np.array([fits[op][fit][mu]['btsp']
                                 for op in operators])
            MS, MS_err, MS_btsp = convert_to_MSbar(bag, bag_err, bag_btsp,
                                                   mu, MS_bar_mu)
            for op_idx, op in enumerate(operators):
                fits[op][fit][mu]['MS'] = {'val': MS[op_idx],
                                           'err': MS_err[op_idx],
                                           'btsp': MS_btsp[:, op_idx]}
    for op in operators:
        if basis == 'SUSY':
            show_flag = True
        else:
            show_flag = False
        operator_summary(op, FLAG=show_flag, open=False)
    pickle.dump(fits, open('cc_extrap_dict.p', 'wb'))
else:
    fits = pickle.load(open('cc_extrap_dict.p', 'rb'))


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

for i, op in enumerate(operators):
    rv += [r'\clearpage']
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
    rv += [r'\caption{Fit values of coefficients in $B = B_0(1 + \mathbf{\alpha}'
           + r' a^2 + \mathbf{\beta} \frac{m_\pi^2}{f_\pi^2} + \ldots)$.}']
    rv += [r'\end{center}']
    rv += [r'\end{table}']

    rv += [r'\begin{figure}']
    rv += [r'\centering']
    rv += [r'\includegraphics[page=1, width=1.1\textwidth]{plots/' +
           op+'_fit_summary.pdf}']
    rv += [r'\caption{\\(left) $B_{phys}$ in RI/SMOM scheme from fit variations ' +
           r'(fits with $p$-value $<0.05$ marked with ``$\times$"). \\' +
           r'(right) $B_{phys}$ in $\overline{MS}$ computed using ' +
           r'$B^{\overline{MS}} = R^{\overline{MS}\leftarrow SMOM}(3.0)' +
           r'\sigma_{npt}^{F1M}(3.0, 2.0) B^{SMOM}$.}']
    rv += [r'\end{figure}']
    rv += [r'\clearpage']

    for fit in fits[op].keys():
        for mu in mu_list:
            filename = fits[op][fit][mu]['filename']
            rv += [r'\includepdf[link, pages=-]{'+filename+'}']

rv += [r'\end{document}']

f = open('tex/extrap.tex', 'w')
f.write('\n'.join(rv))
f.close()

os.system("pdflatex tex/extrap.tex")
os.system("open extrap.pdf")
