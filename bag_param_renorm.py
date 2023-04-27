from NPR_classes import *

all_bag_data = h5py.File('kaon_bag_fits.h5','r')
bag_ensembles = list(all_bag_data.keys())

Z_A_dict = {'central':{'C0':0.711920,
                       'C1':0.717247,
                       'C2':0.717831,
                       'M0':0.743436,
                       'M1':0.744949,
                       'M2':0.745190,
                       'F1S':0.761125},
            'errors':{'C0':0.000024,
                      'C1':0.000067,
                      'C2':0.000053,
                      'M0':0.000016,
                      'M1':0.000039,
                      'M2':0.000040,
                      'F1S':0.000019}}

def load_info(key, ens, ops, meson='ls',**kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson=='ls':
        central_val = np.array([np.array(h5_data[
                      op][key]['central']).item() 
                      for op in ops])
        error = np.array([np.array(h5_data[
                op][key]['error']).item() 
                for op in ops])
        bootstraps = np.zeros(shape=(5,200))
        for i in range(5):
            op = ops[i]
            bootstraps[i,:] = np.array(h5_data[op][key][
                                'Bootstraps'])[:,0]

    elif meson=='ll':
        central_val = np.array(h5_data[key]['central']).item()
        error = np.array(h5_data[key]['error']).item()
        bootstraps = np.array(h5_data[key]['Bootstraps'])[:,0]

    return central_val, error, bootstraps

def into_arr(list_arr):
    return np.array([l for l in list_arr])

def load_Z_data(ens, **kwargs):
    mom, Z, Z_err = pickle.load(open(f'RISMOM/{ens}.p','rb'))
    for k1 in mom.keys():
        for k2 in mom[k1].keys():
            mom[k1][k2] = into_arr(mom[k1][k2])
            Z[k1][k2] = into_arr(Z[k1][k2])
            Z_err[k1][k2] = into_arr(Z_err[k1][k2])
    return mom, Z, Z_err

import itertools
from scipy.interpolate import interp1d
class Z_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP','TT']
    N_boot = N_boot
    def __init__(self, ensemble, action=(0,0), bag=False, **kwargs):
        
        self.ens = ensemble
        self.action = action
        self.bag = bag
        self.ainv = params[self.ens]['ainv']
        self.sea_m = "{:.4f}".format(params[self.ens]['masses'][0])
        self.masses = (self.sea_m, self.sea_m)
        if self.ens in bag_ensembles:
            self.mpi, self.mpi_err, self.mpi_btsp = load_info('m_0', self.ens,
                                                    self.operators, meson='ll')
        self.f_m_ren = f_pi_PDG/self.ainv
        np.random.seed(seed)
        self.f_m_ren_btsp = np.random.normal(self.f_m_ren, f_pi_PDG_err/self.ainv, self.N_boot)
    
        Z_data = load_Z_data(self.ens)
        self.Z_data = Z_data
        self.momenta = Z_data[0][self.action][self.masses]
        self.N_mom = self.momenta.shape[0]

        if self.action==(0,1) or self.action==(1,0):
            self.action==(0,1)
            print('Actions (0,1) and (1,0) have been averaged into group called "(0,1)"')

            Z1 =  Z_data[1][(0,1)][self.masses]
            Z1_err =  Z_data[2][(0,1)][self.masses]
            np.random.seed(seed)
            Z1_btsp = np.array([[[np.random.normal(Z1[m,i,j],
                      Z1_err[m,i,j], self.N_boot) for j in range(5)]
                      for i in range(5)] for m in range(self.N_mom)])
            Z2 =  Z_data[1][(1,0)][self.masses]
            Z2_err =  Z_data[2][(1,0)][self.masses]
            np.random.seed(seed)
            Z2_btsp = np.array([[[np.random.normal(Z2[m,i,j],
                      Z2_err[m,i,j], self.N_boot) for j in range(5)]
                      for i in range(5)] for m in range(self.N_mom)])
            self.Z = (Z1+Z2)/2.0
            self.Z_btsp = np.array([(Z1_btsp[m,]+Z2_btsp[m,])/2.0
                          for m in range(self.N_mom)])
            self.Z_err = np.array([[[st_dev(self.Z_btsp[m,i,j,:],self.Z[m,i,j])
                                     for j in range(5)] for i in range(5)]
                                     for m in range(self.N_mom)])
        else:
            self.Z = Z_data[1][self.action][self.masses]
            self.Z_err = Z_data[2][self.action][self.masses]
            np.random.seed(seed)
            self.Z_btsp = np.array([[[np.random.normal(self.Z[m,i,j],
                          self.Z_err[m,i,j], self.N_boot) for j in range(5)]
                          for i in range(5)] for m in range(self.N_mom)])
        if self.bag:
            bl = bilinear_analysis(self.ens, loadpath=f'pickles/{self.ens}_bl.p')
            Z_bl = bl.avg_results[self.action][self.masses]
            Z_bl_err = bl.avg_errs[self.action][self.masses]
            for m in range(self.N_mom):
                mult = Z_bl[m]['A']/Z_bl[m]['P']
                self.Z[m,:,:] = mask*self.Z[m,:,:]*(mult**2)
                self.Z[m,0,0] = self.Z[m,0,0]/mult

                np.random.seed(seed)
                bl_btsp = {c:np.random.normal(Z_bl[m][c],
                           Z_bl_err[m][c],self.N_boot) for c in Z_bl[m].keys()}
                for k in range(self.N_boot):
                    mult = bl_btsp['A'][k]/bl_btsp['P'][k]
                    self.Z_btsp[m,:,:,k] = mask*self.Z_btsp[m,:,:,k]*(mult**2)
                    self.Z_btsp[m,0,0,k] = self.Z_btsp[m,0,0,k]/mult
                self.Z_err[m,:,:]  = np.array([[st_dev(self.Z_btsp[m,i,j,:],
                                     self.Z[m,i,j]) for j in range(5)]
                                     for i in range(5)])

    def interpolate(self, mu, **kwargs):
        if mu in self.momenta:
            mu_idx = self.momenta.index(mu)
            matrix, errs, btsp = self.Z[mu_idx,], self.Z_err[mu_idx,], self.Z_btsp[mu_idx,]
        else:
            matrix, errs = np.zeros(shape=(5,5)), np.zeros(shape=(5,5))
            btsp = np.zeros(shape=(5,5,200))
            for i,j in itertools.product(range(5), range(5)):
                if mask[i,j]:
                    y = self.Z[:,i,j]
                    f = interp1d(self.momenta,y,fill_value='extrapolate')
                    matrix[i,j] = f(mu)

                    ys = self.Z_btsp[:,i,j,:]
                    store = []
                    for k in range(self.N_boot):
                        Y = ys[:,k]
                        f = interp1d(self.momenta,Y,fill_value='extrapolate')
                        store.append(f(mu))
                    errs[i,j] = st_dev(np.array(store), mean=matrix[i,j])
                np.random.seed(seed)
                btsp[i,j,:] = np.random.normal(matrix[i,j], errs[i,j],
                              self.N_boot)
        return matrix, errs, btsp
        
    def scale_evolve(self, mu1, mu2, **kwargs):
        
        Z_mu1, Z_mu1_err, Z_mu1_btsp = self.interpolate(mu1)
        Z_mu2, Z_mu2_err, Z_mu2_btsp = self.interpolate(mu2)
        sig = Z_mu1@np.linalg.inv(Z_mu2)

        sig_btsp = np.array([Z_mu1_btsp[:,:,b]@np.linalg.inv(Z_mu2_btsp[:,:,b])
                           for b in range(self.N_boot)])
        sig_err = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5),range(5)):
            if mask[i,j]:
                sig_err[i,j] = st_dev(sig_btsp[:,i,j], mean=sig[i,j])

        return sig, sig_err, sig_btsp

    def chiral_data(self, fit='central', i=range(5), **kwargs):
        a_sq = (1/self.ainv)**2
        if fit=='central':
            mpi_f_m_sq = (self.mpi/self.f_m_ren)**2
            return  [a_sq, mpi_f_m_sq]
        else:
            k = kwargs['k']
            mpi_f_m_sq = (self.mpi_btsp[k]/self.f_m_ren_btsp[:,k])**2
            return  [a_sq, mpi_f_m_sq]

    def output_h5(self, add_mu=[2.0,3.0], **kwargs):
        filename = 'Z_fq_bag.h5' if self.bag else 'Z_fq_A.h5'
        f = h5py.File(filename,'a')
        f_str = f'{self.action}/{self.ens}'

        momenta = self.momenta
        Z = self.Z
        Z_err = self.Z_err

        for mu in add_mu:
            momenta = np.append(momenta, mu)
            Z_mu, Z_mu_err, Z_mu_btsp = self.interpolate(mu)
            Z = np.append(Z,np.resize(Z_mu,(1,5,5)),axis=0)
            Z_err = np.append(Z_err,np.resize(Z_mu_err,(1,5,5)),axis=0)

        f.create_dataset(f_str+'/momenta',data=momenta)
        f.create_dataset(f_str+'/Z',data=Z)
        f.create_dataset(f_str+'/Z_err',data=Z_err)

class bag_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP','TT']
    #renorm_scale = 3
    #match_scale = 2
    N_boot = N_boot
    def __init__(self, ensemble, action=(0,0), **kwargs):
        
        self.ens = ensemble
        self.action = action
        self.ainv = params[self.ens]['ainv']
        bag_data = all_bag_data[self.ens]
        self.bag, self.bag_err, self.bag_btsp = load_info('bag', self.ens,
                                                self.operators, meson='ls')
        self.f_m_ren = f_pi_PDG/self.ainv
        np.random.seed(seed)
        self.f_m_ren_btsp = np.random.normal(self.f_m_ren, f_pi_PDG_err/self.ainv, self.N_boot)

        self.mpi, self.mpi_err, self.mpi_btsp = load_info('m_0', self.ens,
                                                self.operators, meson='ll')

        self.Z_info = Z_analysis(self.ens, bag=True)
        self.momenta = self.Z_info.momenta
        self.Z, self.Z_btsp = self.Z_info.Z, self.Z_info.Z_btsp
        self.N_mom = self.Z_info.N_mom
        self.bag_ren = np.array([self.Z[m,:,:]@self.bag
                                for m in range(self.N_mom)])
        self.bag_ren_btsp = np.array([[self.Z_btsp[m,:,:,k]@self.bag_btsp[:,k] 
                            for k in range(self.N_boot)] 
                            for m in range(self.N_mom)])
        self.bag_ren_err = np.array([[st_dev(self.bag_ren_btsp[m,:,i],
                                      self.bag_ren[m,i]) for i in range(5)]
                                      for m in range(self.N_mom)])

    def interpolate(self, mu):
        Z_mu, Z_mu_err, Z_mu_btsp = self.Z_info.interpolate(mu)
        central = Z_mu@self.bag
        btsp = np.array([Z_mu_btsp[:,:,k]@self.bag_btsp[:,k]
                        for k in range(self.N_boot)]) 
        err = np.array([st_dev(btsp[:,i],mean=central[i])
                       for i in range(5)])
        return central, err, btsp
    
    def chiral_data(self, fit='central', **kwargs):
        a_sq = (1/self.ainv)**2
        if fit=='central':
            mpi_f_m_sq = (self.mpi/self.f_m_ren)**2
            return  [a_sq, mpi_f_m_sq]
        else:
            k = kwargs['k']
            mpi_f_m_sq = (self.mpi_btsp[k]/self.f_m_ren_btsp[k])**2
            return  [a_sq, mpi_f_m_sq]

    def ansatz(self, params, **kwargs):
        a_sq, mpi_f_m_sq = self.chiral_data(**kwargs)
        func = params[0]*(1+params[1]*a_sq + params[2]*mpi_f_m_sq)
        if 'addnl_terms' in kwargs.keys():
            if kwargs['addnl_terms']=='a4':
                func += params[0]*(params[3]*(a_sq**2))
            elif kwargs['addnl_terms']=='m4':
                func += params[0]*(params[3]*(mpi_f_m_sq**2))
            elif kwargs['addnl_terms']=='log':
                lambda_f_m_sq = 1.0/(f_pi_PDG**2) 
                mult = params[3]/(16*np.pi**2)
                func += params[0]*(mult*mpi_f_m_sq*np.log(mpi_f_m_sq/lambda_f_m_sq))
        return func 


