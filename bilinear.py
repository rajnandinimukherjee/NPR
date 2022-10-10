from NPR_structures import *

input_dict  ={0:{'prop':'fgauge_SDWF',
                 'Ls':'16',
                 'M5':'1.80'},
              1:{'prop':'sfgauge_MDWF_sm',
                 'Ls':'12',
                 'M5':'1.00'}}


bl_list = common_cf_files('bilinears', prefix='bi_')
currents = ['S','P','V','A','T']

from numpy.linalg import norm
from tqdm import tqdm


def bl_mixed_action(num1, num2, **kwargs):
    dict1 = input_dict[num1]
    dict2 = input_dict[num2]
    results = {'S':{}, 'P':{}, 'V':{}, 'A':{}, 'T':{}}
    errs = {'S':{}, 'P':{}, 'V':{}, 'A':{}, 'T':{}}
    for i in tqdm(range(len(bl_list))):
        b = bl_list[i]
        prop1_name, prop2_name = b.rsplit('__')
        prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

        if all(prop1_info[k]==v for k,v in dict1.items()):
            if all(prop2_info[k]==v for k,v in dict2.items()):
                prop1 = external(filename=prop1_name)
                prop2 = external(filename=prop2_name)
                mom_diff = prop1.tot_mom-prop2.tot_mom
                # choose RI-SMOM data
                if (prop1.mom_sq==prop2.mom_sq) and (prop1.mom_sq==norm(mom_diff)**2):
                    bl = bilinear(prop1, prop2)
                    bl.errs()
                    for k in results.keys():
                        if bl.q not in results[k].keys():
                            results[k][bl.q] = [(bl.projected[k]/bl.F[k]).real]
                            errs[k][bl.q] = [bl.proj_err[k]]
    momenta = [i for i in results['S'].keys()]
    momenta.sort()
    momenta = np.array(momenta)

    avg_results = {k:np.array([np.mean(v[m]) for m in momenta]) for k,v in results.items()}
    avg_errs = {k:np.array([np.mean(v[m]) for m in momenta]) for k,v in errs.items()}
    return momenta, avg_results, avg_errs


mixed_action = {}
for a in [0,1]:
    for b in [0,1]:
        mom, res, err = bl_mixed_action(a,b)
        mixed_action[(a,b)] = {'mom':mom, 'res':res, 'err':err}

from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots(nrows=2, ncols=5)
for i in range(5):
    val_col = ax[0,i]
    err_col = ax[1,i]
    #val_col.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    current = currents[i]
    for k in mixed_action.keys():
        mom = mixed_action[k]['mom']
        res = mixed_action[k]['res'][current]
        err = mixed_action[k]['err'][current]
        val_col.scatter(mom, np.abs(res), label=k)
        err_col.scatter(mom, err, label=k)
    val_col.title.set_text(current)
    handles, labels = err_col.get_legend_handles_labels()
    #err_col.set_ylim([1e-6, 1e-3])
ax[1,2].set_xlabel('$q/GeV$')
fig.legend(handles, labels, loc='center right')
#fig.tight_layout()
fig.suptitle('C1: $Z/Z_{\psi}$ for sMOM bilinears ($\gamma_{\mu}$ scheme, 15 configs)')

plt.show()















