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
    N_boot = 200
    def __init__(self, ens, create=False):
        self.ens = ens
        self.NPR_masses = params[self.ens]['masses']
        self.N_mass = len(self.NPR_masses)
        self.mass_comb = {('l','l'):self.NPR_masses[0]}
        self.mass_comb.update({(f'c{m-1}',f'c{m-1}'):self.NPR_masses[m]
                                for m in range(1,self.N_mass)})
        self.data = {key:{} for key in self.mass_comb.keys()}

        if ens not in self.eta_C_file or create:
            self.createH5()
        else:
            self.N_cf, self.T = self.eta_C_file[self.ens][str(
                                list(self.mass_comb.keys())[0])]['corr'].shape
            self.data = {k:{obj:{'data':np.zeros(shape=(self.N_cf,self.T),
                                 dtype=float)} for obj in ['corr','PJ5q']}
                                 for k in self.mass_comb.keys()}
            for key in self.data.keys():
                for obj in ['corr','PJ5q']:
                    self.data[key][obj]['data'][:,:] = np.array(self.eta_C_file[self.ens][str(key)][obj])

        self.T_half = int(self.T/2)
        for key in self.data.keys():
            for obj in ['corr','PJ5q']:
                data = self.data[key][obj]['data']
                avg_data = np.mean(data,axis=0)
                err = np.array([st_dev(data[:,t], mean=avg_data[t])
                                for t in range(self.T)])

                folded_data = 0.5*(data[:,1:] + data[:,::-1][:,:-1])[:,:self.T_half]
                folded_avg_data = np.mean(folded_data,axis=0)
                folded_err = np.array([st_dev(folded_data[:,t]) for t in range(self.T_half)])
                
                btsp_data = bootstrap(folded_data, K=self.N_boot)
                cov = COV(btsp_data, center=folded_avg_data)

                self.data[key][obj].update({'avg':avg_data,
                                            'err':err,
                                            'folded':folded_data,
                                            'folded_avg':folded_avg_data,
                                            'folded_err':folded_err,
                                            'btsp':btsp_data,
                                            'COV':cov})

    def plot(self, key='all', **kwargs):
        keylist = [key] if key!='all' else self.mass_comb.keys()
        
        fig = plt.figure()
        x = np.arange(self.T_half)
        for k in keylist:
            y = m_eff(self.data[k]['corr']['folded_avg'],ansatz='cosh')
            btsp = np.array([m_eff(self.data[k]['corr']['btsp'][b,:],
                             ansatz='cosh') for b in range(self.N_boot)])
            e = np.array([st_dev(btsp[:,t], mean=y[t]) for t in range(len(y))])
            plt.errorbar(x[:-2], y, yerr=e, fmt='o',
                         capsize=4, label=str(self.mass_comb[k]))
        plt.title('C1 meff')
        plt.xlabel('t')
        plt.legend()

        filename = f'plots/{self.ens}_meff.pdf'
        pdf = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pdf, format='pdf')
        pdf.close()
        plt.close('all')
        os.system('open '+filename)

    def ansatz(self, params, t, **kwargs):
        return params[0]*np.exp(-params[1]*t)

    def massfit(self, key, t_info, **kwargs):
        (start, end, thin) = t_info 
        t = np.arange(start, end+thin, thin)

        def diff(params, fit='central', k=0, **kwargs):
            if fit=='central':
                y = self.data[key]['corr']['folded_avg'][t]
            else:
                y = self.data[key]['corr']['btsp'][k,t]
            return y - self.ansatz(params, t)

        cov = self.data[key]['COV'][start:end+thin:thin,
                                    start:end+thin:thin]
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)
        def LD(params, fit='central', k=0):
            return L.dot(diff(params, fit=fit, k=k))

        guess = [1e+2,0.1]
        res = least_squares(LD, guess, args=('central', 0))
        chi_sq = LD(res.x).dot(LD(res.x))
        dof = len(t)-len(guess)
        pdb.set_trace()
        pvalue = gammaincc(dof/2,chi_sq/2)
        print(pvalue)

        params = res.x

        params_btsp = np.zeros(shape=(self.N_boot,len(guess)))
        for k in range(self.N_boot):
            res_k = least_squares(LD, guess, args=('btsp',k))
            params_btsp[k,:] = res_k.x

        params_err = np.array([st_dev(params_btsp[:,i], mean=params[i])
                               for i in range(len(params))])


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

        self.data = {k:{obj:{'data':np.zeros(shape=(self.N_cf,self.T),dtype=float)}
                        for obj in ['corr','PJ5q']}
                        for k in self.mass_comb.keys()}

        for key in self.data.keys():
            for c in range(self.N_cf):
                for obj in ['corr', 'PJ5q']:
                    suffix = 'mesons/' if obj=='corr' else 'conserved/'
                    filepath = self.datapath+f'/{self.cf_list[c]}/'+suffix
                    if obj=='corr':
                        filepath = glob.glob(filepath+f'{key[0]}_R*{key[1]}_R*')[0]+'/'
                    config_data = np.zeros(shape=(self.T_src, self.T))
                    for t in range(self.T_src):
                        filename = glob.glob(filepath+f'*{key[0]}_R*t{T_src_list[t]}*')[0]
                        datafile = h5py.File(filename,'r')['meson'][f'meson_{self.eta_C_idx}']\
                                   if obj=='corr' else h5py.File(filename,'r')['wardIdentity']
                        data = np.array(datafile[obj])['re']
                        config_data[t,:] = np.roll(data, -int(self.T/self.T_src)*t)
                    self.data[key][obj]['data'][c,:] = np.mean(config_data,axis=0)


        if self.ens in self.eta_C_file:
            del self.eta_C_file[self.ens]
        ens_group = self.eta_C_file.create_group(self.ens)
        for key in self.data.keys():
            key_group = ens_group.create_group(str(key))
            for obj in ['corr','PJ5q']:
                key_group.create_dataset(obj,data=self.data[key][obj])
            key_group.attrs['mass'] = self.mass_comb[key]
        print(f'Added correlator data to {self.ens} group in eta_C.h5 file')
