from basics import *


class external:
    obj = 'externalleg'
    prefix = 'external_'

    def __init__(self, ensemble, momentum=[-3, -3, 0, 0],
                 twist=[0.00, 0.00, 0.00, 0.00],
                 prop='fgauge_SDWF', Ls='16', M5='1.80', am='0.0100',
                 filename='', **kwargs):

        data = path+ensemble
        L = params[ensemble]['XX']
        cfgs = sorted(os.listdir(data)[1:])
        self.N_cf = len(cfgs)  # number of configs)
        if filename != '':
            self.filename = filename
            self.info = decode_prop(self.filename)
            self.momentum = [int(i)
                             for i in self.info['src_mom_p'].rsplit('_')]
            self.twist = [float(i) for i in self.info['tw'].rsplit('_')]
        else:
            self.momentum = momentum
            self.twist = twist
            self.info = {'prop': prop,
                         'Ls': Ls,
                         'M5': M5,
                         'am': am,
                         'tw': '_'.join(['%.2f' % i for i in self.twist]),
                         'src_mom_p': '_'.join([str(i) for i in self.momentum])}
            self.filename = encode_prop(self.info)

        coeff = 2*np.pi/L
        self.total_momentum = coeff * \
            (np.array(self.momentum)+np.array(self.twist))
        self.pslash = np.sum([self.total_momentum[i]*Gamma[dirs[i]]
                             for i in range(len(dirs))], axis=0)
        self.momentum_norm = np.linalg.norm(self.total_momentum)
        self.momentum_squared = (self.momentum_norm)**2

        self.propagator = stat(val=np.empty(
            shape=(self.N_cf, 12, 12), dtype='complex128'))
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.obj}/\
                    {self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')['ExternalLeg']['corr'][0, 0, :]
            self.propagator.val[cf, :] = np.array(h5_data['re'] +
                                                  h5_data['im']*1j).swapaxes(
                1, 2).reshape((12, 12))

        self.propagator.avg = np.mean(self.propagator.val, axis=0)
        self.propagator.btsp = bootstrap(self.propagator.val, K=N_boot)
        self.propagator.dagger = self.propagator.avg.conj().T

        self.outgoing_propagator = stat(
            val=Gamma['5']@self.propagator.dagger@Gamma['5'],
            btsp=np.array([Gamma['5'] @
                           (self.propagator.btsp[k,].conj().T)@Gamma['5']
                           for k in range(N_boot)]))

        self.inv_propagator = stat(
            val=np.linalg.inv(self.propagator.avg),
            btsp=np.array([np.linalg.inv(self.propagator.btsp[k,])
                           for k in range(N_boot)]))

        self.inv_outgoing_propagator = stat(
            val=np.linalg.inv(self.outgoing_propagator.val),
            btsp=np.array([np.linalg.inv(self.btsp_outgoing_propgator[k])
                           for k in range(self.N_boot)]))

        self.Z_q_qslash = stat(
            val=np.trace(-1j*self.inv_propagator.val @
                         self.pslash).real/(12*self.momentum_squared),
            btsp=np.array([np.trace(-1j*(self.inv_propagator.btsp[k,] @
                                         self.pslash)).real /
                           (12*self.momentum_squared)
                           for k in range(N_boot)]))
