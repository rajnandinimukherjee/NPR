from NPR_classes import *

#bl_list  = common_cf_files(path+'C1','bilinears','bi_')
#prop1_name, prop2_name = bl_list[0].split('__')
#prop1, prop2 = external('C1',filename=prop1_name), external('C1',filename=prop2_name)

C1 = bilinear_analysis('C1')
C1.NPR_all(massive=True)

