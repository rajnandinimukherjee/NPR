from cont_chir_extrap import *
b = bag_fits(bag_ensembles)
mu_list = [2.4, 2.0, 1.8, 1.5]
MS_bar_mu = 3.0


def convert_to_MSbar(bags, bags_err, bag_btsp, mu1, mu2, **kwargs):
    F1M_Z = Z_analysis('F1M', bag=True)
    sig, sig_err, sig_btsp = F1M_Z.scale_evolve(mu1, mu2)
    R_conv = R_RISMOM_MSbar(mu2)
    bag_MS = R_conv@sig@bag
    bag_MS_btsp = np.array([R_conv@sig_btsp[k,]@bag_btsp[:, k]
                            for k in range(bag_analysis.N_boot)])
    bag_MS_err = np.array([st_dev(bag_MS_btsp[:, i], bag_MS[i])
                           for i in range(len(bag_MS))])
    return bag_MS, bag_MS_err, bag_MS_btsp


def operator_summary(operator, mu, filename=None,
                     open=True, **kwargs):
    ansatze = list(fits[operator].keys())
    ansatz_desc = [fits[operator][a]['kwargs']['title']
                   for a in ansatze]
    vals = [fits[operator][fit][mu]['phys'] for fit in ansatze]
    errs = [fits[operator][fit][mu]['err'] for fit in ansatze]
    chis = [fits[operator][fit][mu]['chi_sq_dof'] for fit in ansatze]
    cont_slope = [fits[operator][fit][mu]['coeffs'][0] for fit in ansatze]

    fig = plt.subplots(figsize=(5, 4))
    plt.title(r'$\mu='+str(np.around(mu, 2))+'$ GeV')
    plt.errorbar(np.arange(len(ansatze)), vals, yerr=errs, fmt='o', capsize=2)
    # plt.gca().set_xticklabels(ansatze)
    plt.xticks(np.arange(len(ansatze)), ansatz_desc, rotation=45,
               ha='right')
    plt.ylabel(r'$B(a=0, m_\pi=m_\pi^{phys})$ in RI/SMOM')
    tick = min(errs)*0.1
    for i in range(len(ansatze)):
        plt.annotate(str(np.around(chis[i], 2)),
                     (i, vals[i]+errs[i]+tick),
                     ha='center', va='bottom')
        plt.annotate(str(int(100*np.around(cont_slope[i], 2)))+r'%',
                     (i, vals[i]-errs[i]-tick), fontsize=9,
                     ha='center', va='top')

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

    for op in fits.keys():
        print(f'Generating plots for operator {op}')
        for fit in fits[op].keys():
            kwargs = fits[op][fit]['kwargs']
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

    pickle.dump(fits, open('cc_extrap_dict.p', 'wb'))
else:
    fits = pickle.load(open('cc_extrap_dict.p', 'rb'))

for fit in fits['VVpAA'].keys():
    for mu in mu_list:
        bag = np.array([fits[op][fit][mu]['phys'] for op in operators])
        bag_err = np.array([fits[op][fit][mu]['err'] for op in operators])
        bag_btsp = np.array([fits[op][fit][mu]['btsp'] for op in operators])
        MS, MS_err, MS_btsp = convert_to_MSbar(bag, bag_err, bag_btsp,
                                               mu, MS_bar_mu)
        for op_idx, op in enumerate(operators):
            fits[op][fit][mu]['MS'] = {'val': MS[op_idx],
                                       'err': MS_err[op_idx],
                                       'btsp': MS_btsp[:, op_idx]}

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

for op in operators:
    rv += [r'\clearpage']
    ansatze = [fits[op][fit]['kwargs']['title'] for fit in fits[op].keys()]
    rv += [r'\section{'+op+'}']
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
