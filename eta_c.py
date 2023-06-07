from basics import *

#=== measured eta_c masses in lattice units======
eta_c_data = {#'C0':{'central':{0.30:1.249409,
              #                 0.35:1.375320,
              #                 0.40:1.493579},
              #      'errors':{0.30:0.000056,
              #                0.35:0.000051,
              #                0.40:0.000048}},
              'C1':{'central':{0.30:1.24641,
                               0.35:1.37227,
                               0.40:1.49059},
                    'errors':{0.30:0.00020,
                              0.35:0.00019,
                              0.40:0.00017}},
              'M1':{'central':{0.22:0.96975,
                               0.28:1.13226,
                               0.34:1.28347,
                               0.40:1.42374},
                    'errors':{0.22:0.00018,
                              0.28:0.00015,
                              0.34:0.00016,
                              0.40:0.00015}},
             'F1S':{'central':{0.18:0.82322,
                               0.23:0.965045,
                               0.28:1.098129,
                               0.33:1.223360,
                               0.40:1.385711},
                    'errors':{0.18:0.00010,
                              0.23:0.000093,
                              0.28:0.000086,
                              0.33:0.000080,
                              0.40:0.000074}}
                    }

eta_PDG = 2983.9/1000
eta_PDG_err = 0.5/1000
eta_stars = [1.2,2.4,2.6,eta_PDG]

def interpolate_eta_c(ens,find_y,**kwargs):
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']
    f_central = interp1d(y*ainv,x,fill_value='extrapolate')
    pred_x = f_central(find_y)

    btsp = np.array([np.random.normal(y[i],yerr[i],100)
                    for i in range(len(y))])
    pred_x_k = np.zeros(100)
    for k in range(100):
        y_k = btsp[:,k]
        f_k = interp1d(y_k*ainv,x,fill_value='extrapolate')
        pred_x_k[k] = f_k(find_y) 
    pred_x_err = ((pred_x_k[:]-pred_x).dot(pred_x_k[:]-pred_x)/100)**0.5
    return [pred_x.item(), pred_x_err]

class etaCvalence:
    vpath = 'valence/'
    eta_C_gamma = ('Gamma5','Gamma5')
    eta_C_file = h5py.File('eta_C.h5','a')
    N_btsp = 200
    def __init__(self, ens, create=False):
        self.ens = ens
        self.NPR_masses = params[self.ens]['masses'][1:]
        self.N_mass = len(self.NPR_masses)
        self.NPR_mass_deg = [(f'c{m}',f'c{m}') for m in range(self.N_mass)]

        if ens not in self.eta_C_file or create:
            self.createH5()
        else:
            self.T, self.N_cf = self.eta_C_file[self.ens][str(
                                self.NPR_mass_deg[0])]['corr'].shape
            self.eta_C_corr = {k:np.zeros(shape=(self.N_cf,self.T),dtype=float)
                               for k in self.NPR_mass_deg}
            for key in self.eta_C_corr.keys():
                self.eta_C_corr[key][:,:] = np.array(self.eta_C_file[self.ens][str(
                                                   self.NPR_mass_deg[0])]['corr'])

    def massfit(self, t, key, **kwargs):
        corr_data = self.eta_C_corr[key]
        avg_data = np.mean(corr_data,axis=0)

        folded_data = 0.5*(corr_data + np.roll(corr_data[:,::-1],1,axis=0))
        folded_avg_data = 0.5*(avg_data + np.roll(avg_data[::-1],1))

        btsp_data = bootstrap(folded_data, K=self.N_boot)
        pdb.set_trace()
        #cov = COV(btsp_data, 'center'=folded_avg_data)

        def ansatz(params, t, **kwargs):
            return params[0]*np.exp(-params[1]*self.T)

    def createH5(self, **kwargs):
        self.datapath = path+self.vpath+self.ens
        self.cf_list = sorted(next(os.walk(self.datapath))[1])[:-4]
        self.N_cf = len(self.cf_list)

        self.meson_combos = []
        test_path = self.datapath+f'/{self.cf_list[0]}/mesons/'
        for foldername in next(os.walk(test_path))[1]:
            str_m1, str_m2 = foldername.rsplit('__')
            self.meson_combos.append((str_m1.rsplit('_R')[0], str_m2.rsplit('_R')[0]))
        self.N_mc = len(self.meson_combos)

        testfile_path = glob.glob(test_path+'c0*c0*')[0]+'/'
        self.T_src = len(os.listdir(testfile_path))

        testfile_path = glob.glob(testfile_path+'*c0*t00*')[0]
        self.testfile = h5py.File(testfile_path,'r')['meson']
        self.T = len(np.array(self.testfile['meson_0']['corr'])['re'])
        T_src_list = ['%02d'%t for t in np.arange(0,self.T,self.T/self.T_src)]

        self.gammas = []
        for mes_idx in range(16**2):
            gamma_src = self.testfile[f'meson_{mes_idx}'].attrs['gamma_src'][0].decode()
            gamma_snk = self.testfile[f'meson_{mes_idx}'].attrs['gamma_snk'][0].decode()
            self.gammas.append((gamma_src,gamma_snk))
        self.eta_C_idx = self.gammas.index(self.eta_C_gamma) 

        self.eta_C_corr = {k:np.zeros(shape=(self.N_cf,self.T),dtype=float)
                           for k in self.NPR_mass_deg}

        for key in self.eta_C_corr.keys():
            for c in range(self.N_cf):
                filepath = self.datapath+f'/{self.cf_list[c]}/mesons/'
                filepath = glob.glob(filepath+f'{key[0]}*{key[1]}*')[0]+'/'
                config_data = np.zeros(shape=(self.T_src, self.T))
                for t in range(self.T_src):
                    filename = glob.glob(filepath+f'*t{T_src_list[t]}*')[0]
                    datafile = h5py.File(filename,'r')['meson'][f'meson_{self.eta_C_idx}']
                    data = np.array(datafile['corr'])['re']
                    config_data[t,:] = np.roll(data, -int(self.T/self.T_src)*t)
                self.eta_C_corr[key][c,:] = np.mean(config_data,axis=0)

        if self.ens in self.eta_C_file:
            del self.eta_C_file[self.ens]
        ens_group = self.eta_C_file.create_group(self.ens)
        for key in self.eta_C_corr.keys():
            key_group = ens_group.create_group(str(key))
            key_group.create_dataset('corr',data=self.eta_C_corr[key])
            key_group.attrs['mass'] = self.NPR_masses[int(key[0][1])]
        print(f'Added correlator data to {self.ens} group in eta_C.h5 file')
