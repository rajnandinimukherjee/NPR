from NPR_classes import *

sl_dict = {ens:bilinear_analysis(ens) for ens in ['C1','M1']}

for ens, bl_obj in sl_dict.items():
    r_actions = range(len(bl_obj.actions))
    for a1, a2 in itertools.product(r_actions,r_actions):
        for mass in bl_obj.all_masses:
            bl_obj.NPR((mass,bl_obj.sea_mass),action=(a1,a2),renorm='SMOM')
