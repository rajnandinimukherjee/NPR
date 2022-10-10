from NPR_structures import *
import pickle

ensembles = ['C1', 'C2', 'M1', 'M2','F1M']
schemes = ['1']#, '1']#, '2']
N_boot = 30

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
    file = f'all_res/{ens}_{s}_scheme.p'
    momenta, results, errs = pickle.load(open(file, 'rb'))
    return momenta, results, errs

def extrap(momenta, results, errs, point=3, **kwargs):
    matrix = {}
    errors = {}
    x = momenta
    for k,l in itertools.product(range(2), range(2)):
        matrix[(k,l)] = np.zeros(shape=(5,5))
        errors[(k,l)] = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5), range(5)):
            if mask[i,j]:
                y = [results[(k,l)][m][i,j] for m in momenta]
                f = interp1d(x,y)
                #pdb.set_trace()
                matrix[(k,l)][i,j] = f(point)

                e = [errs[(k,l)][m][i,j] for m in momenta]
                np.random.seed(1)
                ys = np.random.multivariate_normal(y,np.diag(e)**2,N_boot)
                store = []
                for Y in ys:
                    f = interp1d(x,Y)
                    store.append(f(point))
                errors[(k,l)][i,j] = st_dev(np.array(store), mean=matrix[(k,l)][i,j])
    return matrix, errors

all_data = {ens:{} for ens in ensembles}
for ens, s, in itertools.product(ensembles, schemes):
    momenta, results, errs = get_data(ens, s)
    extrap_3, err_3 = extrap(momenta, results, errs, point=3)
    all_data[ens][s] = {'mom':momenta,
                        'Z_fac':results, 
                        'Z_err':errs,
                        'extrap_3':extrap_3,
                        'err_3':err_3}


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










