from NPR_classes import *

sl_dict = {ens:bilinear_analysis(ens) for ens in UKQCD_ens}
filename = 'michael_sl_Z_V.h5'
file_h5 = h5py.File(filename,'a')

for ens, bl_obj in sl_dict.items():
    for a1, a2 in itertools.product(range(2),range(2)):
        print(ens, (a1,a2))
        bl_obj.NPR((bl_obj.sea_mass,bl_obj.sea_mass),action=(a1,a2),
                    massive=True,renorm='SMOM')
    bl_obj.merge_mixed()

for action in [(0,0), (0,1), (1,1)]:
    if str(action) in file_h5:
        del file_h5[str(action)]
    action_group = file_h5.create_group(str(action))
    for ens, bl_obj in sl_dict.items():
        mass = (bl_obj.sea_mass,bl_obj.sea_mass)
        ens_group = action_group.create_group(ens) 
        momenta = list(bl_obj.momenta[action][mass])
        ens_group.create_dataset('mu',data=momenta)
        Z_V = [bl_obj.avg_results[action][mass][m]['V']
               for m in range(len(momenta))]
        ens_group.create_dataset('Z_V',data=Z_V)
        Z_V_err = [bl_obj.avg_errs[action][mass][m]['V']
                   for m in range(len(momenta))]
        ens_group.create_dataset('Z_V_err',data=Z_V_err)
        

