from NPR_structures import *
from numpy.linalg import norm

am_dict = {'C1':'0.0050', 'C2':'0.0100', 'M1':'0.0040', 'M2':'0.0060', 'F1M':'0.0214'}
scheme = 1

ensemble = data.rsplit('/')[-1]
input_dict  ={0:{'prop':'fgauge_SDWF',
                 'am':am_dict[ensemble]},
              1:{'prop':'sfgauge_MDWF_sm',
                 'am':am_dict[ensemble]}}


fq_list = common_cf_files('fourquarks', prefix='fourquarks_')
bl_list = common_cf_files('bilinears', prefix='bi_')


fq_store = {}
results = {}
errs = {}
for d1 in [0,1]:
    for d2 in [0,1]:
        dict1 = input_dict[d1]
        dict2 = input_dict[d2]

        results[(d1,d2)] = {}
        fq_store[(d1,d2)] = {}
        errs[(d1,d2)] = {}
        for i in tqdm(range(len(fq_list))):
            b = fq_list[i]
            prop1_name, prop2_name = b.rsplit('__')
            prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)
            #pdb.set_trace()

            if all(prop1_info[k]==v for k,v in dict1.items()):
                if all(prop2_info[k]==v for k,v in dict2.items()):
                    prop1 = external(filename=prop1_name)
                    prop2 = external(filename=prop2_name)
                    mom_diff = prop1.tot_mom-prop2.tot_mom
                    # choose RI-SMOM data
                    if (prop1.mom_sq==prop2.mom_sq) and (scheme*prop1.mom_sq==norm(mom_diff)**2):
                        if np.linalg.norm(mom_diff) not in results.keys():
                            fq = fourquark(prop1, prop2)
                            fq.errs()
                            fq_store[(d1,d2)][fq.q] = fq
                            results[(d1,d2)][fq.q] = fq.Z_V
                            errs[(d1,d2)][fq.q] = fq.Z_V_errs

momenta = list(results[(0,0)].keys())
momenta.sort()
momenta = np.array(momenta)

import pickle
pickle.dump([momenta, results, errs],
            open(f'all_res/{ensemble}_{scheme}_scheme.p','wb'))












