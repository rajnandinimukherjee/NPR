from NPR_structures import *

all_bag_data = h5py.File('kaon_bag_fits.h5','r')
ensembles = list(all_bag_data.keys())

def load_info(key, ens, meson='ls', **kwargs):
    h5_data = all_bag_data[ens][meson]
    if meson=='ls':
        central_val = np.diag([np.array(h5_data[
                      op][key]['central']).item() 
                      for op in list(h5_data.keys())])
        error = np.diag([np.array(h5_data[
                op][key]['error']).item() 
                for op in list(h5_data.keys())])
        bootstraps = np.zeros(shape=(5,5,200))
        for i in range(5):
            op = list(h5_data.keys())[i]
            bootstraps[i,i,:] = np.array(bag_data[op][key][
                                'Bootstraps'])[:,0]
    elif meson=='ll':
        central_val = np.diag([np.array(h5_data[
                      key]['central']).item()]*5)
        error = np.diag([np.array(h5_data[
                key]['error']).item()]*5)
        bootstraps = np.zeros(shape=(5,5,200))

    

class bag_analysis:
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP','TT']
    renorm_scale = 3
    match_scale = 2
    action = (0,0)
    N_boot = 200
    def __init__(self, ensemble, **kwargs):
        
        self.ens = ensemble
        bag_data = all_bag_data[self.ens]['ls']
        Z_data = pickle.load(open(f'RISMOM/{self.ens}.p','rb'))
        self.momenta = Z_data[0]
        
        self.bag = np.diag([np.array(bag_data[op]['bag'][
                   'central']).item() for op in self.operators])
        self.Z = Z_data[1][self.action]
        self.bag_ren = [Z@self.bag for Z in list(self.Z.values())]

        self.f_m = np.diag([np.array(bag_data[op]['f_M'][
                   'central']).item() for op in self.operators])

        self.bag_err = np.diag([np.array(bag_data[op]['bag'][
                       'error']).item() for op in self.operators])
        self.Z_err = Z_data[2][self.action][self.renorm_scale]


        self.bag_btsp = np.zeros(shape=(5,5,200))
        self.f_m_btsp = np.zeros(shape=(5,5,200))
        for i in range(5):
            op = self.operators[i]
            self.bag_btsp[i,i,:] = np.array(bag_data[op]['bag'][
                                   'Bootstraps'])[:,0]
            self.f_m_btsp[i,i,:] = np.array(bag_data[op]['f_M'][
                                   'Bootstraps'])[:,0]
        self.Z_btsp = np.array([[np.random.normal(self.Z[i,j],
                      self.Z_err[i,j], self.N_boot) for j in range(5)]
                      for i in range(5)])
        bag_ren_btsp_diff = np.array([self.Z_btsp[:,:,k]@self.bag_btsp[
                            :,:,k]-self.bag_ren for k in range(self.N_boot)]) 
        self.bag_ren_err = np.array([[(bag_ren_btsp_diff[:,i,j].dot(
                           bag_ren_btsp_diff[:,i,j])/self.N_boot)**0.5
                           for j in range(5)] for i in range(5)])


    
