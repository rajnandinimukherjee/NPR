# Load standard modules
import numpy as np

########## Action parameters ##############
params = {'C0': {'XX': 48,
                 'TT': 96,
                 'Ls': 24,
                 'M5': 1.8,
                 'aml_sea': 0.000780,
                 'ams_sea': 0.03620,
                 'ams_phys': 0.03580,
                 'ams_phys_err': 0.00016,
                 'ainv': 1.7295,
                 'ainv_err': 0.0038,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/C0/ckpoint_lat",
                 'moms': [6, 7, 8, 9, 10, 11],
                 'twistvals': [0.00, 0.50,],
                 'baseactions': ["MDWF", "MDWF_sm"],
                 'scale': [2, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [24, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.0181, 0.2,   0.3,   0.4,   0.5],
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12]},
          'C1M': {'XX': 24,
                  'TT': 64,
                  'Ls': 24,
                  'M5': 1.8,
                  'aml_sea': 0.005,
                  'ams_sea': 0.03620,
                  'ams_phys': 0.03580,
                  'ams_phys_err': 0.00016,
                  'ainv': 1.7295,
                  'ainv_err': 0.0038,
                  'Gloc': "/mnt/lustre/tursafs1/home/dp207/dp207/shared/paboyle/C1M/ckpoint_C1M_lat",
                  'moms': [3, 4, 5],
                  'twistvals': [0.00, 0.25, 0.50, 0.75],
                  'baseactions': ["MDWF", "MDWF_sm"],
                  'scale': [2, 2],
                  'gauges': ["fgauge", "sfgauge"],
                  'Lss': [24, 12],
                  'M5s': [1.8, 1.0],
                  # 'masses' :[0.005, 0.2,   0.3,   0.4,   0.5],
                  # 'resids' :[1.e-8, 1.e-10,1.e-10,1.e-12,1.e-12,1.e-12]}
                  'masses': [0.005, 0.01, 0.0181, 0.0362, 0.05, 0.1, 0.15, 0.2,   0.3,   0.4,   0.5],
                  'NPR_cfgs': list(str(x) for x in range(600, 1880+1, 160)),
                  'resids': [1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-10, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12]},
          'C1': {'XX': 24,
                 'TT': 64,
                 'Ls': 16,
                 'M5': 1.8,
                 'aml_sea': 0.005,
                 'ams_sea': 0.04,
                 'ams_phys': 0.03224,
                 'ams_phys_err': 0.00018,
                 'ainv': 1.7848,
                 'ainv_err': 0.0050,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/C1/ckpoint_lat",
                 'moms': [3, 4, 5],
                 'twistvals': [0.00, 0.25, 0.50, 0.75],
                 'baseactions': ["SDWF", "MDWF_sm"],
                 'scale': [1, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [16, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.005, 0.01, 0.02, 0.04, 0.05, 0.1, 0.15, 0.2,   0.3,   0.4,   0.5],
                 'NPR_cfgs': list(str(x) for x in range(1000, 8000+1, 1000)),
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12]},
          'C2': {'XX': 24,
                 'TT': 64,
                 'Ls': 16,
                 'M5': 1.8,
                 'aml_sea': 0.01,
                 'ams_sea': 0.04,
                 'ams_phys': 0.03224,
                 'ams_phys_err': 0.00018,
                 'ainv': 1.7848,
                 'ainv_err': 0.0050,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/C2/ckpoint_lat",
                 'moms': [3, 4, 5],
                 'twistvals': [0.00, 0.25, 0.50, 0.75],
                 'baseactions': ["SDWF", "MDWF_sm"],
                 'scale': [1, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [16, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.01, 0.02, 0.2,   0.3,   0.4,   0.5],
                 'NPR_cfgs': list(str(x) for x in range(1500, 4000+1, 500)),
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12]},
          'M0': {'XX': 64,
                 'TT': 128,
                 'Ls': 12,
                 'M5': 1.8,
                 'aml_sea': 0.000678,
                 'ams_sea': 0.02661,
                 'ams_phys': 0.02539,
                 'ams_phys_err': 0.00017,
                 'ainv': 2.3586,
                 'ainv_err': 0.0070,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/M0/ckpoint_lat",
                 'moms': [6, 7, 8, 9, 10, 11],
                 'twistvals': [0.00, 0.50],
                 'baseactions': ["MDWF", "MDWF_sm"],
                 'scale': [2, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [12, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.0133, 0.15, 0.225, 0.3, 0.375,   0.45],
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12,]},
          'M1M': {'XX':32,
                  'TT':64,
                  'Ls': 12,
                  'M5':1.8,
                  'aml_sea':0.004,
                  'ams_sea':0.02661,
                  'ainv': 2.3586,
                  'ainv_err': 0.0070,
                  'Gloc': "/mnt/lustre/tursafs1/home/dp207/dp207/shared/paboyle/M1M/ckpoint_M1M_lat",
                  'moms': [3,4,5],
                  'twistvals': [0.00, 0.25, 0.50, 0.75],
                  'baseactions':["MDWF"  ,"MDWF_sm"],
                  'scale'      :[2       ,2        ],
                  'gauges'     :["fgauge","sfgauge"],
                  'Lss'        :[12      ,12       ],
                  'M5s'        :[1.8     ,1.0      ],
                  'masses' :[0.004, 0.008, 0.0133, 0.02661, 0.05, 0.1, 0.15, 0.225,   0.3,   0.375,   0.45],
                  'NPR_cfgs': list(str(x) for x in range(600, 1880+1, 160)),
                  'resids' :[1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-10, 1.e-10,1.e-10,1.e-12,1.e-12,1.e-12]},
          'M1': {'XX': 32,
                 'TT': 64,
                 'Ls': 16,
                 'M5': 1.8,
                 'aml_sea': 0.004,
                 'ams_sea': 0.03,
                 'ams_phys': 0.02477,
                 'ams_phys_err': 0.00018,
                 'ainv': 2.3833,
                 'ainv_err': 0.0086,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/M1/ckpoint_lat",
                 'moms': [3, 4, 5],
                 'twistvals': [0.00, 0.25, 0.50, 0.75],
                 'baseactions': ["SDWF", "MDWF_sm"],
                 'scale': [1, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [16, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.004, 0.008, 0.015, 0.03, 0.05, 0.01, 0.15,  0.225,   0.3, 0.375,   0.45],
                 'NPR_cfgs': list(str(x) for x in range(1250, 3250+1, 250)),
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12,]},
          'M2': {'XX': 32,
                 'TT': 64,
                 'Ls': 16,
                 'M5': 1.8,
                 'aml_sea': 0.006,
                 'ams_sea': 0.03,
                 'ams_phys': 0.02477,
                 'ams_phys_err': 0.00018,
                 'ainv': 2.3833,
                 'ainv_err': 0.0086,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/M2/ckpoint_lat",
                 'moms': [3, 4, 5],
                 'twistvals': [0.00, 0.25, 0.50, 0.75],
                 'baseactions': ["SDWF", "MDWF_sm"],
                 'scale': [1, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [16, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.006, 0.012, 0.015, 0.15,   0.225,   0.3, 0.375,   0.45],
                 'NPR_cfgs': list(str(x) for x in range(500, 1750+1, 250)),
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12,]},
          'M3': {'XX': 32,
                 'TT': 64,
                 'Ls': 16,
                 'M5': 1.8,
                 'aml_sea': 0.008,
                 'ams_sea': 0.03,
                 'ams_phys': 0.02477,
                 'ams_phys_err': 0.00018,
                 'ainv': 2.3833,
                 'ainv_err': 0.0086,
                 'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/M3/ckpoint_lat",
                 'moms': [3, 4, 5],
                 'twistvals': [0.00, 0.25, 0.50, 0.75],
                 'baseactions': ["SDWF", "MDWF_sm"],
                 'scale': [1, 2],
                 'gauges': ["fgauge", "sfgauge"],
                 'Lss': [16, 12],
                 'M5s': [1.8, 1.0],
                 'masses': [0.008, 0.015, 0.016,  0.15,   0.225,   0.3, 0.375,   0.45],
                 'NPR_cfgs': list(str(x) for x in range(500, 1750+1, 250)),
                 'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12,]},
          'F1M': {'XX': 48,
                  'TT': 96,
                  'Ls': 12,
                  'M5': 1.8,
                  'ams_sea': 0.02144,
                  'aml_sea': 0.002144,
                  'ams_phys': 0.02217,
                  'ams_phys_err': 0.00016,
                  'ainv': 2.708,
                  'ainv_err': 0.010,
                  'Gloc': "/tessfs1/work/dp008/dp008/shared/F1_mobius/F1_mobius/ckpoint_EODWF_lat",
                  'moms': [4, 5, 6, 7],
                  'twistvals': [0.00, 0.25, 0.50, 0.75],
                  'baseactions': ["MDWF", "MDWF_sm"],
                  'scale': [2, 2],
                  'gauges': ["fgauge", "sfgauge"],
                  'Lss': [12, 12],
                  'M5s': [1.8, 1.0],
                  'masses': [0.00214, 0.00428, 0.0107, 0.02144, 0.132, 0.198, 0.264, 0.330, 0.396, 0.45],
                  'NPR_cfgs': list(str(x) for x in range(250, 1500+1, 250)),
                  'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12, 1.e-12]},
          'F1S': {'XX': 48,
                  'TT': 96,
                  'Ls': 12,
                  'M5': 1.8,
                  'ams_sea': 0.02144,
                  'aml_sea': 0.002144,
                  'ainv': 2.7850,
                  'ainv_err': 0.010,
                  'Gloc': "/tessfs1/work/dp008/dp008/shared/dwf_2+1f/F1S/ckpoint_lat",
                  'moms': [4, 5, 6, 7],
                  'twistvals': [0.00, 0.25, 0.50, 0.75],
                  'baseactions': ["SDWF", "MDWF_sm"],
                  'scale': [1, 2],
                  'gauges': ["fgauge", "sfgauge"],
                  'Lss': [12, 12],
                  'M5s': [1.8, 1.0],
                  'masses': [0.002144, 0.004288, 0.01072, 0.02144, 0.132, 0.198, 0.264, 0.330, 0.396],
                  'NPR_cfgs': list(str(x) for x in range(4000, 7500+1, 500)),
                  'resids': [1.e-8, 1.e-10, 1.e-10, 1.e-12, 1.e-12, 1.e-12]},
          'KEKF1': {'XX': 64,
                    'TT': 128,
                    'Ls': 8,
                    'M5': 1.0,
                    'aml_sea': 0.0030,
                    'ams_sea': 0.015,
                    'ainv': 4.496,
                    'ainv_err': 0.080,
                    'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.47/ml0.0030/ms0.0150/X64_T128_Ls8_M51.0_Moebius/Conf",
                    'moms': [3, 4, 5],
                    'twistvals': [0.00, 0.25, 0.50, 0.75],
                    'baseactions': ["MDWF_sm"],
                    'scale': [2],
                    'gauges': ["sfbinary"],
                    'Lss': [8],
                    'M5s': [1.0],
                    'masses': [0.0030],
                    'resids': [1.e-8]},
          'KEKC2a': {'XX': 32,
                     'TT': 64,
                     'Ls': 12,
                     'M5': 1.0,
                     'aml_sea': 0.0070,
                     'ams_sea': 0.03,
                     'ainv': 2.453,
                     'ainv_err': 0.044,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.17/ml0.0070/ms0.030/X32_T64_Ls12_M51.0_Moebius/Conf",
                     'moms': [2, 3, 4, 5],
                     'twistvals': [0.00, 0.25, 0.50, 0.75],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [12],
                     'M5s': [1.0],
                     'masses': [0.0070],
                     'resids': [1.e-8]},
          'KEKC2b': {'XX': 32,
                     'TT': 64,
                     'Ls': 12,
                     'M5': 1.0,
                     'aml_sea': 0.0070,
                     'ams_sea': 0.04,
                     'ainv': 2.453,
                     'ainv_err': 0.044,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.17/ml0.0070/ms0.040/X32_T64_Ls12_M51.0_Moebius/Conf",
                     'moms': [2, 3, 4, 5],
                     'twistvals': [0.00, 0.25, 0.50, 0.75],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [12],
                     'M5s': [1.0],
                     'masses': [0.0070],
                     'resids': [1.e-8]},
          'KEKM1a': {'XX': 48,
                     'TT': 96,
                     'Ls': 8,
                     'M5': 1.0,
                     'aml_sea': 0.0042,
                     'ams_sea': 0.018,
                     'ainv': 3.610,
                     'ainv_err': 0.065,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.35/ml0.0042/ms0.018/X48_T96_Ls8_M51.0_Moebius/Conf",
                     'moms': [2, 3, 4, 5],
                     'twistvals': [0.00, 0.25, 0.50, 0.75],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [8],
                     'M5s': [1.0],
                     'masses': [0.0042],
                     'resids': [1.e-8]},
          'KEKM1b': {'XX': 48,
                     'TT': 96,
                     'Ls': 8,
                     'M5': 1.0,
                     'aml_sea': 0.0042,
                     'ams_sea': 0.025,
                     'ainv': 3.610,
                     'ainv_err': 0.065,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.35/ml0.0042/ms0.025/X48_T96_Ls8_M51.0_Moebius/Conf",
                     'moms': [2, 3, 4, 5],
                     'twistvals': [0.00, 0.25, 0.50, 0.75],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [8],
                     'M5s': [1.0],
                     'masses': [0.0042],
                     'resids': [1.e-8]},
          'KEKC1L': {'XX': 48,
                     'TT': 96,
                     'Ls': 12,
                     'M5': 1.0,
                     'aml_sea': 0.0035,
                     'ams_sea': 0.04,
                     'ainv': 2.453,
                     'ainv_err': 0.044,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/C1L/Conf",
                     'moms': [4, 5, 6, 7],
                     'twistvals': [0.00, 0.50],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [12],
                     'M5s': [1.0],
                     'masses': [0.0035],
                     'resids': [1.e-8]},
          'KEKC1S': {'XX': 32,
                     'TT': 64,
                     'Ls': 12,
                     'M5': 1.0,
                     'aml_sea': 0.0035,
                     'ams_sea': 0.04,
                     'ainv': 2.453,
                     'ainv_err': 0.044,
                     'Gloc': "/tessfs1/work/dp008/dp008/shared/UKQCD_KEK/configurations/KEK/b4.17/ml0.0035/ms0.040/X32_T64_Ls12_M51.0_Moebius/Conf",
                     'moms': [3, 4, 5, 6, 7, 8],
                     'twistvals': [0.00, 0.50],
                     'baseactions': ["MDWF_sm"],
                     'scale': [2],
                     'gauges': ["sfbinary"],
                     'Lss': [12],
                     'M5s': [1.0],
                     'masses': [0.0035],
                     'resids': [1.e-8]
                     }
          }


# momentum sources
kombinatorics = [[+1., +1., 0, 0],
                 [+1., 0, +1., 0],
                 [0, +1., +1., 0],
                 [-1., +1., 0, 0],
                 [-1., 0, +1., 0],
                 [0, -1., +1., 0],
                 [+1., -1., 0, 0],
                 [+1., 0, -1., 0],
                 [0, +1., -1., 0],
                 [-1., -1., 0, 0],
                 [-1., 0, -1., 0],
                 [0, -1., -1., 0]]
