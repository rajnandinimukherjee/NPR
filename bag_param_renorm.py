from NPR_structures import *

all_bag_data = h5py.File('kaon_bag_fits.h5','r')
bag_ensembles = list(all_bag_data.keys())

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

def st_dev(data, mean=None, **kwargs):
    '''standard deviation function - finds stdev around data mean or mean
    provided as input'''

    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return np.sqrt(((data-mean).dot(data-mean))/n)

import itertools
from scipy.interpolate import interp1d
class Z_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP','TT']
    action = (0,0)
    N_boot = 200
    def __init__(self, ensemble, **kwargs):
        
        self.ens = ensemble
        self.ainv = params[self.ens]['ainv']
        self.f_m, self.f_m_err, self.f_m_btsp = load_info('f_M', self.ens,
                                                self.operators, meson='ls')
        self.mpi, self.mpi_err, self.mpi_btsp = load_info('m_0', self.ens,
                                                self.operators, meson='ll')
    
        Z_data = pickle.load(open(f'RISMOM/{self.ens}.p','rb'))
        self.momenta = Z_data[0]
        self.Z, self.Z_err = Z_data[1][self.action], Z_data[2][self.action]
        self.Z_btsp = {m:np.array([[np.random.normal(Z[i,j],
                      self.Z_err[m][i,j], self.N_boot) for j in range(5)]
                      for i in range(5)]) for m, Z in self.Z.items()}

    def interpolate(self, mu, **kwargs):
        if mu in self.momenta:
            matrix, errs, btsp = self.Z[mu], self.Z_err[mu], self.Z_btsp[mu]
        else:
            matrix, errs = np.zeros(shape=(5,5)), np.zeros(shape=(5,5))
            btsp = np.zeros(shape=(5,5,200))
            for i,j in itertools.product(range(5), range(5)):
                if mask[i,j]:
                    y = [self.Z[m][i,j] for m in self.momenta]
                    f = interp1d(self.momenta,y,fill_value='extrapolate')
                    matrix[i,j] = f(mu)

                    ys = [list(self.Z_btsp[m][i,j,:]) for m in self.momenta]
                    store = []
                    for Y in ys:
                        f = interp1d(x,Y,fill_value='extrapolate')
                        store.append(f(point))
                    errors[i,j] = st_dev(np.array(store), mean=matrix[i,j])
                btsp[i,j,:] = np.random.normal(matrix[i,j], errs[i,j],
                              self.N_boot)
        return matrix, errs, btsp
        
    def scale_evolve(self, mu1, mu2, **kwargs):
        
        Z_mu1, Z_mu1_err, Z_mu1_btsp = self.interpolate(mu1)
        Z_mu2, Z_mu2_err, Z_mu2_btsp = self.interpolate(mu2)
        sig = Z_mu1@np.linalg.inv(Z_mu2)

        sigmas = np.array([Z_mu1_btsp[:,:,b]@np.linalg.inv(Z_mu2_btsp[:,:,b])
                           for b in range(self.N_boot)])
        sig_err = np.zeros(shape=(5,5))
        for i,j in itertools.product(range(5),range(5)):
            if mask[i,j]:
                sig_err[i,j] = st_dev(sigmas[:,i,j], mean=sig[i,j])

        return sig, sig_err

    def extrap_data(self, mu, fixed_mu=3, **kwargs):
        sig, sig_err = scale_evolve(fixed_mu,mu)
        a_sq = (1/self.ainv)**2
        mpi_f_m = np.diag([(self.mp/self.f_m[i,i])**2 for i in range(5)])
        return sig, sig_err, a_sq, mpi_f_m

class bag_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP','TT']
    #renorm_scale = 3
    #match_scale = 2
    action = (0,0)
    N_boot = 200
    def __init__(self, ensemble, **kwargs):
        
        self.ens = ensemble
        self.ainv = params[self.ens]['ainv']
        bag_data = all_bag_data[self.ens]
        self.bag, self.bag_err, self.bag_btsp = load_info('bag', self.ens,
                                                self.operators, meson='ls')
        self.f_m, self.f_m_err, self.f_m_btsp = load_info('f_M', self.ens,
                                                self.operators, meson='ls')
        self.mpi, self.mpi_err, self.mpi_btsp = load_info('m_0', self.ens,
                                                self.operators, meson='ll')

        self.Z_info = Z_analysis(self.ens)
        self.Z, self.Z_btsp = self.Z_info.Z, self.Z_info.Z_btsp
        self.bag_ren = {m:Z@self.bag for m, Z in self.Z.items()}
        bag_ren_btsp_diff = {m:np.array([Z_btsp[:,:,k]@self.bag_btsp[
                            :,k]-self.bag_ren[m] for k in range(self.N_boot)]) 
                            for m, Z_btsp in self.Z_btsp.items()}
        self.bag_ren_err = {m:np.array([(val[:,i].dot(
                           val[:,i])/self.N_boot)**0.5
                           for i in range(5)])
                           for m, val in bag_ren_btsp_diff.items()}

    def interpolate(self, mu):
        Z_mu, Z_mu_err, Z_mu_btsp = self.Z_info.interpolate(mu)
        central_bag_ren = Z_mu@self.bag
        btsp = np.array([Z_mu_btsp[:,:,k]@self.bag_btsp[
               :,:,k]-central_bag_ren for k in range(self.N_boot)]) 
        err = np.array([[(btsp[:,i,j].dot(btsp[:,i,j])/self.N_boot)**0.5
              for j in range(5)] for i in range(5)])
        mu_btsp = np.random.normal(central_bag_ren, err, self.N_boot)
        return central_bag_ren, err, mu_btsp
    
    def chiral_data(self, mu, **kwargs):
        a_sq = (1/self.ainv)**2
        mpi_f_m = np.diag([(self.mp/self.f_m[i,i])**2 for i in range(5)])
        return self.bag_ren, self.bag_ren_err, a_sq, mpi_f_m

