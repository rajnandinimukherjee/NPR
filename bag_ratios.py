from bag_param_renorm import *
import pandas as pd

ratio_dict = {}
ratio_dict['Coarse'] = {key: {} for key in list(all_bag_data.keys())
                        if key[0] == 'C' and key in UKQCD_ens}
ratio_dict['Medium'] = {key: {} for key in list(all_bag_data.keys())
                        if key[0] == 'M' and key in UKQCD_ens}
ratio_dict['Fine'] = {key: {} for key in list(all_bag_data.keys())
                      if key == 'F1M'}

for lat in ratio_dict.keys():
    print('\n'+lat)
    for ens in ratio_dict[lat].keys():
        ratios = [stat(
            val=np.array(all_bag_data[ens]['ls']
                         [operators[i]]['gr-O-gr']['central']) /
            np.array(all_bag_data[ens]['ls']
                     [operators[0]]['gr-O-gr']['central']),
            btsp=np.array(all_bag_data[ens]['ls']
                          [operators[i]]['gr-O-gr']['Bootstraps'][:, 0]) /
            np.array(all_bag_data[ens]['ls']
                     [operators[0]]['gr-O-gr']['Bootstraps'][:, 0])
        )
            for i in range(1, len(operators))]

        ratios = stat(
            val=[rat.val for rat in ratios],
            err='fill',
            btsp=np.array([rat.btsp for rat in ratios]).T
        )
        SUSY_ratios = stat(
            val=NPR_to_SUSY[1:, 1:]@ratios.val,
            err='fill',
                btsp=np.array([NPR_to_SUSY[1:, 1:]@ratios.btsp[k, :]
                               for k in range(N_boot)])
        )
        ratio_dict[lat][ens] = {f'R{i+2}': err_disp(SUSY_ratios.val[i],
                                                    SUSY_ratios.err[i])
                                for i in range(len(operators)-1)}
    df = pd.DataFrame(data=ratio_dict[lat])
    print(df)
