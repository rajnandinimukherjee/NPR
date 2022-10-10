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

#from NPR_structures import *
NPR_dir = '/work/dp008/dp008/shared/runs/npr/'
print_dir = '/home/dp008/dp008/dc-mukh1/npr/analysis'

def f(ens):
    string = '{}: hello {} from {}'.format(
        dt.datetime.now(), ens, current_process().name)
    #pickle.dump(string,open(f'{ens}.p','wb'))
    print(string)
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

