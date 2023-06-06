from basics import *

class external:
    N_boot = N_boot
    obj = 'externalleg'
    prefix = 'external_'
    def __init__(self, ensemble, momentum=[-3,-3,0,0],
                 twist=[0.00, 0.00, 0.00, 0.00],
                 prop='fgauge_SDWF', Ls='16', M5='1.80', am='0.0100',
                 filename='', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        L = params[ensemble]['XX']
        cfgs = sorted(os.listdir(data)[1:])
        self.N_cf = len(cfgs) # number of configs)
        if filename != '':
            self.filename = filename
            self.info = decode_prop(self.filename)
            self.momentum = [int(i) for i in self.info['src_mom_p'].rsplit('_')]
            self.twist = [float(i) for i in self.info['tw'].rsplit('_')]
        else:
            self.momentum = momentum
            self.twist = twist
            self.info = {'prop':prop,
                         'Ls':Ls,
                         'M5':M5,
                         'am':am,
                         'tw':'_'.join(['%.2f'%i for i in self.twist]),
                         'src_mom_p':'_'.join([str(i) for i in self.momentum])}
            self.filename = encode_prop(self.info)

        coeff = 2*np.pi/L
        self.total_momentum = coeff*(np.array(self.momentum)+np.array(self.twist))
        self.pslash = np.sum([self.total_momentum[i]*Gamma[dirs[i]]
                             for i in range(len(dirs))],axis=0)
        self.momentum_norm = np.linalg.norm(self.total_momentum)
        self.momentum_squared = (a_inv*self.momentum_norm)**2
        
        self.propagator = np.empty(shape=(self.N_cf,12,12),dtype='complex128')
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.obj}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['ExternalLeg']['corr'][0,0,:]
            self.propagator[cf,:] = np.array(h5_data['re']+h5_data['im']*1j).swapaxes(1,
                                2).reshape((12,12))

        self.avg_propagator = np.mean(self.propagator,axis=0)
        self.daggered_propagator = self.avg_propagator.conj().T
        self.outgoing_avg_propagator = Gamma['5']@self.daggered_propagator@Gamma['5']
        self.inv_avg_propagator = np.linalg.inv(self.avg_propagator)
        self.inv_outgoing_avg_propagator = np.linalg.inv(self.outgoing_avg_propagator)
            
        self.btsp_propagator = bootstrap(self.propagator, K=self.N_boot)
        self.btsp_outgoing_propgator = np.array([Gamma['5']@(
                            self.btsp_propagator[k,].conj().T)@Gamma['5']
                            for k in range(self.N_boot)])
        self.btsp_inv_propagator = np.array([np.linalg.inv(
                            self.btsp_propagator[k]) for k in range(self.N_boot)])
        self.btsp_inv_outgoing_propagator = np.array([
                            np.linalg.inv(self.btsp_outgoing_propgator[k])
                            for k in range(self.N_boot)])

        self.Z_q_avg_qslash = np.trace(-1j*self.inv_avg_propagator@self.pslash
                                      ).real/(12*self.momentum_squared)
        self.Z_q_btsp_qslash = np.array([np.trace(-1j*(
                               self.btsp_inv_propagator[k,:,:]@self.pslash)).real/(
                               12*self.momentum_squared)
                               for k in range(self.N_boot)])

