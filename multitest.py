import datetime as dt
from multiprocessing import Process, current_process
import sys
from numpy.linalg import norm
import pickle
ensembles = ['C1','C2','M1','M2','F1M'] 
scheme = 0
def common_cf_files(data, corr, cfgs, prefix=None):
    file_names = {cf:os.listdir(f'{data}/{cf}/NPR/{corr}/')
                  for cf in cfgs}

    list_of_cf_files = []
    for cf in file_names.keys():
        for i in range(len(file_names[cf])):
            file_names[cf][i] = file_names[cf][i].rsplit(f'.{cf}.h5')[0]
            if prefix != None:
                file_names[cf][i] = file_names[cf][i].rsplit(prefix)[1]
        list_of_cf_files.append(file_names[cf])

    common_files = list(set.intersection(*map(set,list_of_cf_files)))
    common_files.sort()
    return common_files

from NPR_structures import *

def f(ens):
    print('{}: hello {} from {}'.format(
        dt.datetime.now(), ens, current_process().name))
    data = '/home/rm/external/NPR/'+ens
    cfgs = os.listdir(data)[1:]
    cfgs.sort()
    am_dict = {'C1':'0.0050', 'C2':'0.0100', 'M1':'0.0040', 'M2':'0.0060'}#, 'F1M':'0.0214'}


    ensemble = data.rsplit('/')[-1]
    input_dict  ={0:{'prop':'fgauge_SDWF',
                     'am':am_dict[ens],
                     'Ls':'16',
                     'M5':'1.80'},
                  1:{'prop':'sfgauge_MDWF_sm',
                     'am':am_dict[ens],
                     'Ls':'12',
                     'M5':'1.00'}}


    fq_list = common_cf_files(data, 'fourquarks', cfgs, prefix='fourquarks_')
    bl_list = common_cf_files(data, 'bilinears', cfgs, prefix='bi_')


    fq_store = {}
    results = {}
    errs = {}
    d1, d2 = 0, 0
    dict1 = input_dict[d1]
    dict2 = input_dict[d2]

    results[(d1,d2)] = {}
    fq_store[(d1,d2)] = {}
    errs[(d1,d2)] = {}
    for i in range(len(fq_list)):
        b = fq_list[i]
        prop1_name, prop2_name = b.rsplit('__')
        prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

        if all(prop1_info[k]==v for k,v in dict1.items()):
            if all(prop2_info[k]==v for k,v in dict2.items()):
                prop1 = external(data,cfgs,filename=prop1_name)
                prop2 = external(data,cfgs,filename=prop2_name)
                mom_diff = prop1.tot_mom-prop2.tot_mom
                # choose scheme-wise data
                if (prop1.mom_sq==prop2.mom_sq) and (scheme*prop1.mom_sq==norm(mom_diff)**2):
                    if np.linalg.norm(mom_diff) not in results.keys():
                        fq = fourquark(data,cfgs,prop1, prop2)
                        fq.errs()
                        fq_store[(d1,d2)][fq.q] = fq
                        results[(d1,d2)][fq.q] = fq.Z_V
                        errs[(d1,d2)][fq.q] = fq.Z_V_errs

    momenta = list(results[(0,0)].keys())
    momenta.sort()
    momenta = np.array(momenta)

    pickle.dump([momenta, results[(0,0)], errs[(0,0)]],
                open(f'all_res/{ens}_{scheme}_scheme.p','wb'))
    sys.stdout.flush()

if __name__ == '__main__':
    worker_count = 4
    worker_pool = []
    for _ in range(worker_count):
        p = Process(target=f, args=(ensembles[_],))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")
