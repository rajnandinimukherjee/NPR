from NPR_structures import *
from numpy.linalg import norm
import pickle
import itertools
from scipy.interpolate import interp1d
import pdb

ensembles = ['KEKC1S']

phys_ens = ['C0','M0']

scheme = 1

for ens in ensembles:
    print(ens)
    data = path+ens
    info = params[ens]
    if ens in phys_ens:
        sea_mass = '{:.4f}'.format(info['masses'][0])
    else:
        sea_mass = '{:.4f}'.format(info['aml_sea'])

    
    action0 = info['gauges'][0]+'_'+info['baseactions'][0]
    if 'KEK' not in ens:
        action1 = info['gauges'][1]+'_'+info['baseactions'][1]
        input_dict  ={0:{'prop':action0, 'am':sea_mass},
                      1:{'prop':action1, 'am':sea_mass}}
    else:
        input_dict = {1:{'prop':action0, 'am':sea_mass}}


    fq_list = common_cf_files(data, 'fourquarks', prefix='fourquarks_')
    bl_list = common_cf_files(data, 'bilinears', prefix='bi_')


    fq_store = {}
    results = {}
    errs = {}
    action_list = [0,1] if 'KEK' not in ens else [1]
    for d1 in action_list:
        for d2 in action_list:
            dict1 = input_dict[d1]
            dict2 = input_dict[d2]

            results[(d1,d2)] = {}
            fq_store[(d1,d2)] = {}
            errs[(d1,d2)] = {}
            for i in tqdm(range(len(fq_list)), leave=False):
                b = fq_list[i]
                prop1_name, prop2_name = b.rsplit('__')
                prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)
                #pdb.set_trace()

                if all(prop1_info[k]==v for k,v in dict1.items()):
                    if all(prop2_info[k]==v for k,v in dict2.items()):
                        prop1 = external(ens,filename=prop1_name)
                        prop2 = external(ens,filename=prop2_name)
                        mom_diff = prop1.tot_mom-prop2.tot_mom
                        # choose RI-SMOM data
                        if (prop1.mom_sq==prop2.mom_sq) and (scheme*prop1.mom_sq==norm(mom_diff)**2):
                            if np.linalg.norm(mom_diff) not in results.keys():
                                fq = fourquark(ens,prop1, prop2)
                                fq.errs()
                                fq_store[(d1,d2)][fq.q] = fq
                                results[(d1,d2)][fq.q] = fq.Z_V
                                errs[(d1,d2)][fq.q] = fq.Z_V_errs

    momenta = list(results[(1,1)].keys())
    momenta.sort()
    momenta = np.array(momenta)
    pickle.dump([momenta, results, errs],
                open(f'all_res/{ens}_{scheme}_scheme.p','wb'))
