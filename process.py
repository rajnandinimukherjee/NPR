from NPR_structures import *
import pickle

ensembles = ['C0','C1','C2',
             'M0','M1','M2','M3',
             'F1M','F1S',
             'KEKC2a','KEKC2b',
             #'KEKM1a','KEKM1b',
             'KEKC1S','KEKC1L',
             'KEKF1']
schemes = ['1']
N_boot = 20
UKQCD_ens = ['C0','C1','C2',
             'M0','M1','M2','M3',
             'F1M','F1S']
KEK_ens = ['KEKC2a','KEKC2b',
           'KEKM1a','KEKM1b',
           'KEKC1S','KEKC1L',
           'KEKF1']
all_ens = UKQCD_ens+KEK_ens
#from random import shuffle, seed
#seed(1)
#shuffle(all_ens)

def err_disp(num, err, n=2, **kwargs):
    ''' converts num and err into num(err) in scientific notation upto n digits
    in error, can be extended for accepting arrays of nums and errs as well, for
    now only one at a time'''

    err_dec_place = int(np.floor(np.log10(np.abs(err))))
    err_n_digits = int(err*10**(-(err_dec_place+1-n)))
    num_dec_place = int(np.floor(np.log10(np.abs(num))))
    if num_dec_place<err_dec_place:
        print('Error is larger than measurement')
        return 0
    else:
        num_sf = num*10**(-(num_dec_place))
        num_trunc = round(num_sf, num_dec_place-(err_dec_place+1-n))
        return str(num_trunc)+'('+str(err_n_digits)+')E%+d'%num_dec_place

def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)

import itertools
from scipy.interpolate import interp1d
import pdb

def get_data(ens, s, **kwargs):
    #file = f'all_res/{ens}_{s}_scheme.p'
    file = f'RISMOM/{ens}.p'
    momenta, results, errs = pickle.load(open(file, 'rb'))
    return momenta, results, errs

def extrap(momenta, results, errs, ens, point=3, **kwargs):
    matrix = {}
    errors = {}
    x = momenta
    rg = [0,1] if ens in UKQCD_ens else [0]
    for k,l in itertools.product(rg, rg):
        matrix[(k,l)] = np.zeros(shape=(5,5))
        errors[(k,l)] = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5), range(5)):
            if mask[i,j]:
                y = [results[(k,l)][m][i,j] for m in momenta]
                f = interp1d(x,y,fill_value='extrapolate')
                matrix[(k,l)][i,j] = f(point)

                e = [errs[(k,l)][m][i,j] for m in momenta]
                np.random.seed(1)
                ys = np.random.multivariate_normal(y,np.diag(e)**2,N_boot)
                store = []
                for Y in ys:
                    f = interp1d(x,Y,fill_value='extrapolate')
                    store.append(f(point))
                errors[(k,l)][i,j] = st_dev(np.array(store), mean=matrix[(k,l)][i,j])
    return matrix, errors

from matching import *
R_MS = R_MSbar(3,alpha_3)
all_data = {ens:{} for ens in ensembles}
for ens, s, in itertools.product(ensembles, schemes):
    momenta, results, errs = get_data(ens, s)
    all_data[ens][s] = {'mom':momenta,
                        'Z_fac':results,
                        'Z_err':errs,
                        'Z_MS':{},
                        'err_MS':{}}
    for k in all_data[ens][s]['Z_fac'].keys():
        all_data[ens][s]['Z_MS'][k] = {m:R_MS@all_data[ens][s]['Z_fac'][k][m]
                                       for m in momenta}
        all_data[ens][s]['err_MS'][k] = {m:R_MS@all_data[ens][s]['Z_err'][k][m]
                                       for m in momenta}
    sigma(ens)


    #if ens in KEK_ens:
    #    results[(0,0)] = results.pop((1,1))
    #    errs[(0,0)] = errs.pop((1,1))

    #extrap_3, err_3 = extrap(momenta, results, errs, ens, point=3)
    #extrap_2, err_2 = extrap(momenta, results, errs, ens, point=2)
    #all_data[ens][s] = {'mom':momenta,
    #                    'Z_fac':results, 
    #                    'Z_err':errs,
    #                    'extrap_3':extrap_3,
    #                    'err_3':err_3,
    #                    'extrap_2':extrap_2,
    #                    'err_2':err_2}
    #momenta = list(momenta)
    #momenta.append(3)
    #momenta.append(2)
    #momenta.sort()
    #print(momenta)
    #for k in results.keys():
    #    results[k].update({2:extrap_2[k], 3:extrap_3[k]})
    #    errs[k].update({2:err_2[k], 3:err_3[k]})
    #pickle.dump([np.array(momenta),results, errs], open('RISMOM/'+ens+'.p','wb'))


def plot_actions(ens, s,  **kwargs):

    momenta = all_data[ens][s]['mom']
    results = all_data[ens][s]['Z_fac']
    errs = all_data[ens][s]['Z_err']

    plt.figure()
    for key in results.keys():
        x = momenta
        y = [results[key][m][0,0] for m in momenta]
        e = [errs[key][m][0,0] for m in momenta]
    
        plt.errorbar(x,y,yerr=e,fmt='o',capsize=4,label=key)
    plt.legend()
    plt.xlabel('$q/GeV$')
    plt.title(f'Block 1 Ensemble {ens}')

    fig, ax = plt.subplots(2,2,sharex=True)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for key in results.keys():
            x = momenta
            y = [results[key][m][k,l] for m in momenta]
            e = [errs[key][m][k,l] for m in momenta]
            ax[i,j].errorbar(x,y,yerr=e,fmt='o',capsize=4,label=key)
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle(f'Block 2 Ensemble {ens}')

    fig, ax = plt.subplots(2,2,sharex=True)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for key in results.keys():
            x = momenta
            y = [results[key][m][k,l] for m in momenta]
            e = [errs[key][m][k,l] for m in momenta]
            ax[i,j].errorbar(x,y,yerr=e,fmt='o',capsize=4,label=key)
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    handles, labels = ax[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle(f'Block 3 Ensemble {ens}')
    
    plt.show()


def plot_ratios(ens, **kwargs):

    colors={'0':'b','1':'r'}
    leg={'0':'RI/MOM','1':'RI/SMOM'}
    momenta = all_data[ens]['1']['mom']
    ratio = {m:np.zeros(shape=(5,5)) for m in momenta}

    plt.figure()
    for s in schemes:
        results = all_data[ens][s]['Z_fac']
        errs = all_data[ens][s]['Z_err']
        x = momenta
        y_kl = np.array([[[results[(k,l)][m][0,0] for m in momenta]
                        for l in range(2)] for k in range(2)])
        y = y_kl[0,0,:]*y_kl[1,1,:]/(y_kl[0,1,:]*y_kl[1,0,:])
        for i in range(len(momenta)):
            ratio[momenta[i]][0,0]=y[i]

        plt.scatter(x,y,label=leg[s],c=colors[s])
    
    plt.xlabel('$q/GeV$')
    plt.title(f'Block 1 Ensemble {ens}')

    fig, ax = plt.subplots(2,2,sharex=True)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for s in schemes:
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = np.array([[[results[(a,b)][m][k,l] for m in momenta]
                            for b in range(2)] for a in range(2)])
            y = y_kl[0,0,:]*y_kl[1,1,:]/(y_kl[0,1,:]*y_kl[1,0,:])
            for m in range(len(momenta)):
                ratio[momenta[m]][k,l]=y[m]

            ax[i,j].scatter(x,y,label=leg[s],c=colors[s])
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    plt.suptitle(f'Block 2 Ensemble {ens}')

    fig, ax = plt.subplots(2,2,sharex=True)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for s in schemes:
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = np.array([[[results[(a,b)][m][k,l] for m in momenta]
                            for b in range(2)] for a in range(2)])
            y = y_kl[0,0,:]*y_kl[1,1,:]/(y_kl[0,1,:]*y_kl[1,0,:])
            for m in range(len(momenta)):
                ratio[momenta[m]][k,l]=y[m]

            ax[i,j].scatter(x,y,label=leg[s],c=colors[s])
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    plt.suptitle(f'Block 3 Ensemble {ens}')
    
    #plt.show()
    return ratio

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mc
cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(all_ens)),cmap.N)

def plot_ens(**kwargs):
    s='1'
    fig = plt.figure()
    ax = plt.subplot(111)
    for ens in ensembles:
        a,b = (0,0) if ens in KEK_ens else (1,1)
        momenta = all_data[ens][s]['mom']
        results = all_data[ens][s]['Z_fac']
        errs = all_data[ens][s]['Z_err']
        x = momenta
        y_kl = [results[(a,b)][m][0,0] for m in momenta]

        ax.scatter(x,y_kl,
                   color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                   label=ens)
    
    ax.set_xlabel('$q/GeV$')
    ax.legend(bbox_to_anchor=(1.02, 0.8))
    plt.suptitle(f'Block 1 action (1,1)')

    fig, ax = plt.subplots(2,2)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for ens in ensembles:
            a,b = (0,0) if ens in KEK_ens else (1,1)
            momenta = all_data[ens][s]['mom']
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = [results[(a,b)][m][k,l] for m in momenta]
            ax[i,j].scatter(x,y_kl,
                            color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                            label=ens)
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.04,0.75))
    plt.suptitle(f'Block 2 action (1,1)')

    fig, ax = plt.subplots(2,2)
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for ens in ensembles:
            a,b = (0,0) if ens in KEK_ens else (1,1)
            momenta = all_data[ens][s]['mom']
            results = all_data[ens][s]['Z_fac']
            errs = all_data[ens][s]['Z_err']
            x = momenta
            y_kl = [results[(a,b)][m][k,l] for m in momenta]
            ax[i,j].scatter(x,y_kl,
                            color=cmap(norm(all_ens.index(ens)/len(all_ens))),
                            label=ens)
            if i==1:
                ax[i,j].set_xlabel('$q/GeV$')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.04,0.75))
    plt.suptitle(f'Block 3 action (1,1)')

    pp = PdfPages('plots/summary.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def print_mtx(vals, errs, **kwargs):
    N, M = vals.shape
    string_mtx = np.empty(shape=(5,5),dtype=object)
    for n in range(N):
        for m in range(M):
            if mask[n,m]:
                string_mtx[n,m]=err_disp(vals[n,m],errs[n,m],n=2)
            else:
                string_mtx[n,m]='0'
    print(string_mtx)

def sigma(ens,mu1=2,mu2=3,**kwargs):
    res = all_data[ens]['1']['Z_fac']
    err = all_data[ens]['1']['Z_err']

    all_data[ens]['1']['sigma'] = {}
    all_data[ens]['1']['sigma_err'] = {}

    for k in res.keys():
        Z_1 = res[k][mu1]
        Z_2 = res[k][mu2]
        sig = Z_2@np.linalg.inv(Z_1)
        all_data[ens]['1']['sigma'][k] = sig

        var_Z1 = np.zeros(shape=(5,5,N_boot))
        var_Z2 = np.zeros(shape=(5,5,N_boot))
        for i,j in itertools.product(range(5),range(5)):
            if mask[i,j]:
                var_Z1[i,j,:] = np.random.normal(Z_1[i,j],err[k][mu1][i,j],N_boot)  
                var_Z2[i,j,:] = np.random.normal(Z_2[i,j],err[k][mu2][i,j],N_boot) 
        sigmas = np.array([var_Z2[:,:,b]@np.linalg.inv(var_Z1[:,:,b])
                           for b in range(N_boot)])
        errors = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5),range(5)):
            if mask[i,j]:
                errors[i,j] = st_dev(sigmas[:,i,j], mean=sig[i,j])

        all_data[ens]['1']['sigma_err'][k] = errors











