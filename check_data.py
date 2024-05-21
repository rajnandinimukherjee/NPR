from basics import *
import glob
from ensemble_parameters import *

def check_NPR(ens):
    loc = path+ens+'/results/'
    if ens=='C1S' or ens=='M1S':
        ens = ens[:-1]

    cfgs = params[ens]['NPR_cfgs']
    masses = ['{:.4f}'.format(m) for m in params[ens]['masses']]
    moms = params[ens]['moms']
    twistvals = params[ens]['twistvals']

    exp_num_data = len(moms)*len(twistvals)
    for cfg in cfgs:
        for mass in masses:
            for mom, tw in itertools.product(moms, twistvals):
                mom_str = f'mom_p_{mom}_{mom}_0_0'
                tw_str = f'tw_{tw}_{tw}_0_0'
                filename = glob.glob(f'{cfg}/bilinears/*am_{mass}*{tw_str}*{mom_str}*{cfg}*')



