import os
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import odeint
from coeffs import *
import itertools
import matplotlib.colors as mc
from scipy.linalg import expm
import pdb
from scipy.interpolate import interp1d
from NPR_structures import *
# import seaborn as sns
# sns.set_theme()
import pickle

schemes = ['1']
N_boot = 20
UKQCD_ens = ['C0', 'C1', 'C2',
             'M0', 'M1', 'M2', 'M3',
             'F1M', 'F1S']
KEK_ens = ['KEKC2a', 'KEKC2b',
           'KEKM1a', 'KEKM1b',
           'KEKC1S', 'KEKC1L',
           'KEKF1']
all_ens = UKQCD_ens+KEK_ens
ensembles = all_ens


def err_disp(num, err, n=2, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    err_dec_place = int(np.floor(np.log10(np.abs(err))))
    err_n_digits = int(err*10**(-(err_dec_place+1-n)))
    num_dec_place = int(np.floor(np.log10(np.abs(num))))
    if num_dec_place < err_dec_place:
        print('Error is larger than measurement')
        return 0
    else:
        num_sf = num*10**(-(num_dec_place))
        num_trunc = round(num_sf, num_dec_place-(err_dec_place+1-n))
        digs = -err_dec_place+n-1
        str_num_trunc = str(num)[:digs+2]
        # num_nontrunc = round(num,-err_dec_place+n-1)
        # pdb.set_trace()
        # return str(num_trunc)+'('+str(err_n_digits)+')E%+d'%num_dec_place
        return str_num_trunc+'('+str(err_n_digits)+')'


def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)


def sigma(ens, mu1=2, mu2=3, **kwargs):
    res = all_data[ens]['1']['Z_fac']
    err = all_data[ens]['1']['Z_err']

    all_data[ens]['1']['sigma'] = {}
    all_data[ens]['1']['sigma_err'] = {}

    for k in res.keys():
        Z_1 = res[k][mu1]
        Z_2 = res[k][mu2]
        sig = Z_2@np.linalg.inv(Z_1)
        all_data[ens]['1']['sigma'][k] = sig

        var_Z1 = np.zeros(shape=(5, 5, N_boot))
        var_Z2 = np.zeros(shape=(5, 5, N_boot))
        for i, j in itertools.product(range(5), range(5)):
            if mask[i, j]:
                var_Z1[i, j, :] = np.random.normal(
                    Z_1[i, j], err[k][mu1][i, j], N_boot)
                var_Z2[i, j, :] = np.random.normal(
                    Z_2[i, j], err[k][mu2][i, j], N_boot)
        sigmas = np.array([var_Z2[:, :, b]@np.linalg.inv(var_Z1[:, :, b])
                           for b in range(N_boot)])
        errors = np.zeros(shape=(5, 5))
        for i, j in itertools.product(range(5), range(5)):
            if mask[i, j]:
                errors[i, j] = st_dev(sigmas[:, i, j], mean=sig[i, j])

        all_data[ens]['1']['sigma_err'][k] = errors


def get_data(ens, s, **kwargs):
    # file = f'all_res/{ens}_{s}_scheme.p'
    file = f'RISMOM/{ens}.p'
    momenta, results, errs = pickle.load(open(file, 'rb'))
    return momenta, results, errs


def extrap(momenta, results, errs, ens, point=3, **kwargs):
    matrix = {}
    errors = {}
    x = momenta
    rg = [0, 1] if ens in UKQCD_ens else [0]
    for k, l in itertools.product(rg, rg):
        matrix[(k, l)] = np.zeros(shape=(5, 5))
        errors[(k, l)] = np.zeros(shape=(5, 5))
        for i, j in itertools.product(range(5), range(5)):
            if mask[i, j]:
                y = [results[(k, l)][m][i, j] for m in momenta]
                f = interp1d(x, y, fill_value='extrapolate')
                matrix[(k, l)][i, j] = f(point)

                e = [errs[(k, l)][m][i, j] for m in momenta]
                np.random.seed(1)
                ys = np.random.multivariate_normal(y, np.diag(e)**2, N_boot)
                store = []
                for Y in ys:
                    f = interp1d(x, Y, fill_value='extrapolate')
                    store.append(f(point))
                errors[(k, l)][i, j] = st_dev(
                    np.array(store), mean=matrix[(k, l)][i, j])
    return matrix, errors


def merge_mixed(ens, all_data, **kwargs):
    data = all_data[ens]['1']
    mixed_res = {m: (data['Z_fac'][(0, 1)][m]+data['Z_fac'][(1, 0)][m])/2.0
                 for m in data['mom']}
    mixed_err = {m: (data['Z_err'][(0, 1)][m]+data['Z_err'][(1, 0)][m])
                 for m in data['mom']}
    return mixed_res, mixed_err


# alpha_3 = 4*np.pi*(1-1.00414)/r_11
mus = np.linspace(2, 3, 20)
gs = [g(m) for m in mus]
# alphas = [alpha(m) for m in mus]

# ===leading order perturbative running=====
beta_0 = Bcoeffs(3)[0]
exponent = gamma_0/beta_0


def pt_running_lo(mu1, mu2, **kwargs):
    frac = g(mu1)/g(mu2)
    return expm(exponent*np.log(frac))


order_1_running = np.array([pt_running_lo(2, mu) for mu in mus])

# ===next-to-leading order perturbative running=====
# ====calculating gamma_1_RISMOM=========
beta_0, beta_1 = Bcoeffs(3)[:2]
gamma_1_RISMOM = r_mtx@gamma_0 - gamma_0@r_mtx + \
    gamma_1_MS(f=3) + 2*beta_0*r_mtx


def gamma(mu, order):
    mult = (g(mu)**2)/(16*np.pi**2)
    expansion = [gamma_0*mult, gamma_1_RISMOM*(mult**2)]
    return sum(expansion[:order])


def beta_func(Z, mu, order):
    Z_mtx = Z.reshape([5, 5])
    dZ_dmu = -gamma(mu, order)@(Z_mtx)/mu
    return dZ_dmu.reshape(-1)


def pt_running(ens, order, action=(0, 0), **kwargs):
    Z_2 = all_data[ens]['1']['Z_fac'][action][2]
    Z_mus = odeint(beta_func, Z_2.reshape(-1), mus, args=(order,))
    o2r = np.array([Z_2@np.linalg.inv(Z.reshape([5, 5])) for Z in Z_mus])
    return o2r


# ===NLO running using diagonalisation===============
J = np.loadtxt('J.txt', delimiter=',')
L = np.loadtxt('L.txt', delimiter=',')


def K(mu, **kwargs):
    return np.identity(5) + (J+L*np.log(g(mu)))*(g(mu)**2/(16*np.pi**2))


def pt_running_nlo(mu2, mu1, **kwargs):
    o2r = K(mu2)@pt_running_lo(mu1, mu2)@np.linalg.inv(K(mu1))
    # o2r = pt_running_lo(mu1,mu2)
    # o2r += (g(mu1)**2/(16*np.pi**2))*(J + L*np.log(g(mu1)))@order_1_diag
    # o2r -= (g(mu2)**2/(16*np.pi**2))*order_1_diag@(J + L*np.log(g(mu2)))
    return o2r


order_2_running = np.array([pt_running_nlo(2, mu) for mu in mus])

# ===non-perturbative running=====


def npt_running(ens, action=(0, 0), **kwargs):
    data = all_data[ens]['1']
    mom, Z_facs, Z_errs = data['mom'], data['Z_fac'], data['Z_err']
    running_mtx = np.zeros(shape=(len(mus), 5, 5))
    for m_idx in range(len(mus)):
        mu = mus[m_idx]
        mtx, err = extrap(mom, Z_facs, Z_errs, ens, point=mu)
        running_mtx[m_idx, :, :] = Z_facs[action][2]@np.linalg.inv(mtx[action])
    return running_mtx


all_data = {ens: {} for ens in ensembles}
for ens, s, in itertools.product(ensembles, schemes):
    momenta, results, errs = get_data(ens, s)
    all_data[ens][s] = {'mom': momenta,
                        'Z_fac': results,
                        'Z_err': errs}
    # if ens in UKQCD_ens:
    #    mixed_res, mixed_err = merge_mixed(ens, all_data)
    #    all_data[ens][s]['Z_fac'][(0,1)] = mixed_res
    #    all_data[ens][s]['Z_fac'].pop((1,0))
    #    all_data[ens][s]['Z_err'][(0,1)] = mixed_err
    #    all_data[ens][s]['Z_err'].pop((1,0))
    sigma(ens)

    # if ens in KEK_ens:
    #    results[(0,0)] = results.pop((1,1))
    #    errs[(0,0)] = errs.pop((1,1))

    # extrap_3, err_3 = extrap(momenta, results, errs, ens, point=3)
    # extrap_2, err_2 = extrap(momenta, results, errs, ens, point=2)
    # all_data[ens][s] = {'mom':momenta,
    #                    'Z_fac':results,
    #                    'Z_err':errs,
    #                    'extrap_3':extrap_3,
    #                    'err_3':err_3,
    #                    'extrap_2':extrap_2,
    #                    'err_2':err_2}
    # momenta = list(momenta)
    # momenta.append(3)
    # momenta.append(2)
    # momenta.sort()
    # print(momenta)
    # for k in results.keys():
    #    results[k].update({2:extrap_2[k], 3:extrap_3[k]})
    #    errs[k].update({2:err_2[k], 3:err_3[k]})
    # pickle.dump([np.array(momenta),results, errs], open('RISMOM/'+ens+'.p','wb'))


def plot_actions(ens, s,  **kwargs):

    momenta = all_data[ens][s]['mom']
    results = all_data[ens][s]['Z_fac']
    errs = all_data[ens][s]['Z_err']

    plt.figure()
    for key in results.keys():
        x = momenta
        y = [results[key][m][0, 0] for m in momenta]
        e = [errs[key][m][0, 0] for m in momenta]

        plt.errorbar(x, y, yerr=e, fmt='o', capsize=4, label=key)
    plt.legend()
    plt.xlabel('$q/GeV$')
    plt.title(f'Block 1 Ensemble {ens}')

    fig, ax = plt.subplots(2, 2, sharex=True)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for key in results.keys():
            x = momenta
            y = [results[key][m][k, l] for m in momenta]
            e = [errs[key][m][k, l] for m in momenta]
            ax[i, j].errorbar(x, y, yerr=e, fmt='o', capsize=4, label=key)
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle(f'Block 2 Ensemble {ens}')

    fig, ax = plt.subplots(2, 2, sharex=True)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for key in results.keys():
            x = momenta
            y = [results[key][m][k, l] for m in momenta]
            e = [errs[key][m][k, l] for m in momenta]
            ax[i, j].errorbar(x, y, yerr=e, fmt='o', capsize=4, label=key)
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    handles, labels = ax[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle(f'Block 3 Ensemble {ens}')

    plt.show()


def plot_ratios(ens, **kwargs):

    colors = {'0': 'b', '1': 'r'}
    leg = {'0': 'RI/MOM', '1': 'RI/SMOM'}
    momenta = all_data[ens]['1']['mom']
    ratio = {m: np.zeros(shape=(5, 5)) for m in momenta}

    plt.figure()
    for s in schemes:
        results = all_data[ens][s]['Z_fac']
        errs = all_data[ens][s]['Z_err']
        x = momenta
        y_kl = np.array([[[results[(k, l)][m][0, 0] for m in momenta]
                        for l in range(2)] for k in range(2)])
        y = y_kl[0, 0, :]*y_kl[1, 1, :]/(y_kl[0, 1, :]*y_kl[1, 0, :])
        for i in range(len(momenta)):
            ratio[momenta[i]][0, 0] = y[i]

        plt.scatter(x, y, label=leg[s], c=colors[s])

    plt.xlabel('$q/GeV$')
    plt.title(f'Block 1 Ensemble {ens}')

    fig, ax = plt.subplots(2, 2, sharex=True)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for s in schemes:
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = np.array([[[results[(a, b)][m][k, l] for m in momenta]
                            for b in range(2)] for a in range(2)])
            y = y_kl[0, 0, :]*y_kl[1, 1, :]/(y_kl[0, 1, :]*y_kl[1, 0, :])
            for m in range(len(momenta)):
                ratio[momenta[m]][k, l] = y[m]

            ax[i, j].scatter(x, y, label=leg[s], c=colors[s])
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    plt.suptitle(f'Block 2 Ensemble {ens}')

    fig, ax = plt.subplots(2, 2, sharex=True)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for s in schemes:
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = np.array([[[results[(a, b)][m][k, l] for m in momenta]
                            for b in range(2)] for a in range(2)])
            y = y_kl[0, 0, :]*y_kl[1, 1, :]/(y_kl[0, 1, :]*y_kl[1, 0, :])
            for m in range(len(momenta)):
                ratio[momenta[m]][k, l] = y[m]

            ax[i, j].scatter(x, y, label=leg[s], c=colors[s])
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    plt.suptitle(f'Block 3 Ensemble {ens}')

    # plt.show()
    return ratio


cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(all_ens)), cmap.N)


def plot_ens(**kwargs):
    s = '1'
    fig = plt.figure()
    ax = plt.subplot(111)
    for ens in ensembles:
        a, b = (0, 0) if ens in KEK_ens else (1, 1)
        momenta = all_data[ens][s]['mom']
        results = all_data[ens][s]['Z_fac']
        errs = all_data[ens][s]['Z_err']
        x = momenta
        y_kl = [results[(a, b)][m][0, 0] for m in momenta]

        ax.scatter(x, y_kl,
                   color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                   label=ens)

    ax.set_xlabel('$q/GeV$')
    ax.legend(bbox_to_anchor=(1.02, 1.03))
    plt.savefig('/Users/rajnandinimukherjee/Desktop/OSM.pdf')
    # plt.suptitle(r'$Z_{O_{\text{SM}}}(q,a)/Z_V^2$')

    fig, ax = plt.subplots(2, 2)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for ens in ensembles:
            a, b = (0, 0) if ens in KEK_ens else (1, 1)
            momenta = all_data[ens][s]['mom']
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = [results[(a, b)][m][k, l] for m in momenta]
            ax[i, j].scatter(x, y_kl,
                             color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                             label=ens)
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(1.04,0.75))
    plt.suptitle(f'Block 2 action (1,1)')

    fig, ax = plt.subplots(2, 2)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for ens in ensembles:
            a, b = (0, 0) if ens in KEK_ens else (1, 1)
            momenta = all_data[ens][s]['mom']
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = [results[(a, b)][m][k, l] for m in momenta]
            ax[i, j].scatter(x, y_kl,
                             color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                             label=ens)
            if i == 1:
                ax[i, j].set_xlabel('$q/GeV$')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(1.04,0.75))
    plt.suptitle(f'Block 3 action (1,1)')

    pp = PdfPages('plots/summary.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    os.system("open plots/summary.pdf")


def print_mtx(vals, errs=None, **kwargs):
    N, M = vals.shape
    string_mtx = np.empty(shape=(5, 5), dtype=object)
    for n in range(N):
        for m in range(M):
            if mask[n, m]:
                if errs is None:
                    string_mtx[n, m] = "{0:0.5f}".format(vals[n, m])
                else:
                    string_mtx[n, m] = err_disp(vals[n, m], errs[n, m], n=2)
            else:
                string_mtx[n, m] = '0'
    # print(string_mtx)
    return string_mtx


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('\'', '').replace(
        '[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def make_table(ens, save=True, **kwargs):
    data = all_data[ens]['1']
    rv = [r'\begin{center}']
    rv += [r'\begin{tabular}{c|c|c|c}']
    # & $\sigma_{pt}^{NLO}(2,3)$ \\']
    rv += [ens + r' & $Z$(2 GeV) & $Z$(3 GeV) & $\sigma_{npt}(2,3)$ \\']
    rv += [r'\hline']
    rv += [str(k) + ' & $' + bmatrix(print_mtx(data['Z_fac'][k][2],
           data['Z_err'][k][2])) + '$ & $' + bmatrix(print_mtx(data['Z_fac'][k][3],
                                                               data['Z_err'][k][3])) + '$ & $' + bmatrix(print_mtx(data['sigma'][k],
                                                                                                                   data['sigma_err'][k])) + r'$ &\\' for k in data['Z_fac'].keys()]
    # + '$ & $' + bmatrix(print_mtx(pt_running_nlo(2,3))) #+ r'$ & \\'
    rv += [r'\hline']
    rv += [r'\end{tabular}']
    rv += [r'\end{center}']

    if save:
        f = open('tex/'+ens+'.tex', 'w')
        f.write('\n'.join(rv))
        f.close()
    else:
        return '\n'.join(rv)


def make_results_tex(**kwargs):
    rv = [r'\documentclass[9pt]{extarticle}']
    rv += [r'\usepackage[paperwidth=20in,paperheight=6in]{geometry}']
    rv += [r'\usepackage{amsmath}']
    rv += [r'\usepackage[utf8]{inputenc}']
    rv += [r'\title{NPR $Z_{ij}/Z_V^2$}'+'\n' +
           r'\author{Rajnandini Mukherjee}'+'\n'+r'\date{\today}']
    rv += [r'\begin{document}']
    rv += [r'\maketitle']
    rv += ['\n\clearpage\n' + make_table(ens, save=False) for ens in ensembles]
    rv += [r'\end{document}']

    f = open('tex/results.tex', 'w')
    f.write('\n'.join(rv))
    f.close()

    os.system("pdflatex tex/results.tex")
    os.system("open results.pdf")


def plot_running(ens, action=(0, 0), **kwargs):
    npt = npt_running(ens)
    s = '1'
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.yaxis.tick_right()
    plt.plot(mus**2, order_1_running[:, 0, 0],
             label='LO', c='k', linestyle='dashed')
    plt.plot(mus**2, order_2_running[:, 0, 0], label='NLO mtx', c='k')
    plt.plot(mus**2, pt_running(ens, 2, action)
             [:, 0, 0], label='NLO int', c='r')
    plt.plot(mus**2, npt[:, 0, 0], label='npt', c='b')

    ax.set_xlabel('$q^2/GeV^2$')
    ax.legend(bbox_to_anchor=(1.02, 0.8))
    plt.suptitle(ens+' $\sigma(2GeV,q)_{11}$')

    fig, ax = plt.subplots(2, 2)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        ax[i, j].plot(mus**2, order_1_running[:, k, l],
                      label='LO', c='k', linestyle='dashed')
        ax[i, j].plot(mus**2, order_2_running[:, k, l], label='NLO mtx', c='k')
        ax[i, j].plot(mus**2, pt_running(ens, 2, action)
                      [:, k, l], label='NLO int', c='r')
        ax[i, j].plot(mus**2, npt[:, k, l], label='npt', c='b')
        ax[i, j].yaxis.tick_right()
        if i == 1:
            ax[i, j].set_xlabel('$q^2/GeV^2$')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.8, 1.08))
    plt.suptitle(ens+r' $\sigma(2GeV,q)_{2\times3}$')

    fig, ax = plt.subplots(2, 2)
    for i, j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        ax[i, j].plot(mus**2, order_1_running[:, k, l],
                      label='LO', c='k', linestyle='dashed')
        ax[i, j].plot(mus**2, order_2_running[:, k, l], label='NLO mtx', c='k')
        ax[i, j].plot(mus**2, pt_running(ens, 2, action)
                      [:, k, l], label='NLO int', c='r')
        ax[i, j].plot(mus**2, npt[:, k, l], label='npt', c='b')
        ax[i, j].yaxis.tick_right()
        if i == 1:
            ax[i, j].set_xlabel('$q^2/GeV^2$')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.8, 1.08))
    plt.suptitle(ens+r' $\sigma(2GeV,q)_{4\times5}$')

    pp = PdfPages('plots/running.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    os.system("open plots/running.pdf")

# =================================================================
