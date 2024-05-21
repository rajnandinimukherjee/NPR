from basics import *
import glob
import pandas as pd
from ensemble_parameters import *

def check_NPR(ens):
    print(f'Ensemble: {ens}')
    loc = path+ens+'/results'
    if ens=='C1S' or ens=='M1S':
        ens = ens[:-1]

    cfgs = params[ens]['NPR_cfgs']
    masses = ['{:.4f}'.format(m) for m in params[ens]['masses']]
    moms = params[ens]['moms']
    twistvals = ['{:.2f}'.format(tw) for tw in params[ens]['twistvals']]
    action = params[ens]['gauges'][0]

    exp_num_data = len(moms)*len(twistvals)*len(cfgs)*len(masses)
    count = 0
    df = pd.DataFrame(0, index=masses, columns=cfgs)
    for cfg in cfgs:
        for mass in masses:
            for mom, tw in itertools.product(moms, twistvals):
                mom_str1 = f'mom_p_{mom}_{mom}_0_0'
                mom_str2 = f'mom_p_{mom}_0_{mom}_0'
                tw_str = f'tw_{tw}_{tw}_0.00_0.00'
                files = glob.glob(f'{loc}/{cfg}/NPR/bilinears/*_{action}'+\
                        f'*am_{mass}*{tw_str}*{mom_str1}*_{action}*am_{mass}*{mom_str2}*{cfg}*')
                count += len(files)
                df[cfg][mass] += len(files)

    print(f'Found {count}/{exp_num_data} files')
    print(f'Expecting (mom)x(tw) = {len(moms)}x{len(twistvals)} = {len(moms)*len(twistvals)} files per config per mass point\n')
    print(df)
