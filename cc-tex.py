from cont_chir_extrap import *
b = bag_fits(bag_ensembles)
mu_list = [2.4, 2.0, 1.8, 1.5]


def err_disp(num, err, n=2, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    err_dec_place = int(np.floor(np.log10(np.abs(err))))
    err_n_digits = int(err*10**(-(err_dec_place+1-n)))
    num_dec_place = int(np.floor(np.log10(np.abs(num))))
    if num_dec_place < err_dec_place:
        print('Error is larger than measurement')
        return str(np.around(num, 3))
    else:
        num_sf = num*10**(-(num_dec_place))
        num_trunc = round(num_sf, num_dec_place-(err_dec_place+1-n))
        digs = -err_dec_place+n-1
        str_num_trunc = str(num)[:digs+2]
        # num_nontrunc = round(num,-err_dec_place+n-1)
        # pdb.set_trace()
        # return str(num_trunc)+'('+str(err_n_digits)+')E%+d'%num_dec_place
        return str_num_trunc+'('+str(err_n_digits)+')'


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
        for fit in fits[op].keys():
            kwargs = fits[op][fit]['kwargs']
            for mu in mu_list:
                filename = fits[op][fit][mu]['filename']
                phys, err, coeffs, coeffs_err, chi_sq_dof, pvalue = b.plot_fits(
                    mu, ops=[op], filename=filename, **kwargs)
                fits[op][fit][mu].update({'phys': phys,
                                          'err': err,
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
