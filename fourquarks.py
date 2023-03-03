from basics import *
from bilinear import *

#=====fourquark projectors=======================================
fq_proj = {'SS':np.einsum('ab,cd->abcd',Gamma['I'],Gamma['I']),
        'PP':np.einsum('ab,cd->abcd',Gamma['5'],Gamma['5']),
        'VV':np.sum(np.array([np.einsum('ab,cd->abcd',Gamma[i],Gamma[i])
             for i in dirs]), axis=0),
        'AA':np.sum(np.array([np.einsum('ab,cd->abcd',
             Gamma[i]@Gamma['5'],Gamma[i]@Gamma['5'])
             for i in dirs]), axis=0),
        'TT':np.sum(np.array(sum([[np.einsum('ab,cd->abcd',
             Gamma[dirs[i]]@Gamma[dirs[j]],Gamma[dirs[i]]@Gamma[dirs[j]])
             for j in range(i+1,4)] for i in range(0,4-1)],[])), axis=0)}

fq_gamma_projector = {'VV+AA':fq_proj['VV']+fq_proj['AA'],
                  'VV-AA':fq_proj['VV']-fq_proj['AA'],
                  'SS-PP':fq_proj['SS']-fq_proj['PP'],
                  'SS+PP':fq_proj['SS']+fq_proj['PP'],
                  'TT':fq_proj['TT']}

fq_gamma_F = np.array([[np.einsum('abcd,badc',2*(fq_gamma_projector[k1]-fq_gamma_projector[k1].swapaxes(1,3)),
               fq_gamma_projector[k2]) for k2 in operators] for k1 in operators])


def fq_qslash_projector(q, p1, p2, **kwargs):
    qslash = np.sum([q[i]*gamma[dirs[i]] for i in range(len(dirs))], axis=0)
    q_sq = np.linalg.norm(qslash)**2
    p1_sq = np.linalg.norm(p1)**2
    p2_sq = np.linalg.norm(p2)**2
    p1dotp2 = p1.dot(p2)

    proj = {'VV':np.einsum('ab,cd->abcd',qslash,qslash)/q_sq,
            'AA':np.einsum('ab,cd->abcd',qslash@gamma['5'],qslash@gamma['5'])/q_sq}
    P_L = (np.identity(4)-gamma['5'])/2
    p_sig_P_p = np.sum([np.sum([p1[dirs.index(i)]*((commutator(i,j,
                          g=gamma)/2)@P_L)*p2[dirs.index(j)] for j in dirs],axis=0) for i in dirs],axis=0)
    proj['TT'] = np.einsum('ab,cd->abcd',p_sig_P_p, p_sig_P_p)/(p1_sq*p2_sq - p1dotp2**2)

    q_proj = {'VV+AA':np.einsum('abcd,ef,gh->abcdefgh',
                      proj['VV']+proj['AA'],np.identity(N_c), 
                      np.identity(N_c)),
              'VV-AA':np.einsum('abcd,ef,gh->abcdefgh',
                      proj['VV']-proj['AA'],np.identity(N_c),
                      np.identity(N_c))}
    q_proj['SS-PP'] = q_proj['VV-AA'].swapaxes(4,6)
    q_proj['TT'] = np.einsum('abcd,ef,gh->abcdefgh',proj['TT'],
                             np.identity(N_c),np.identity(N_c))
    q_proj['TT'] = np.moveaxis(np.moveaxis(np.moveaxis(q_proj['TT'],4,1),5,3),6,5)
    q_proj['SS+PP'] = q_proj['TT'].swapaxes(4,6)
    for k in q_proj.keys():
        print(q_proj[k].shape)
    return {k:np.q_proj[k].reshape((12,12,12,12)) for k in operators}


mask = np.full((len(operators),len(operators)),False)
for i in range(len(operators)):
    mask[i,i]=True
mask[1,2], mask[2,1] = True, True
mask[3,4], mask[4,3] = True, True 

class fourquark:
    def __init__(self, ensemble, prop1, prop2, scheme='gamma', **kwargs):

        data = path+ensemble
        a_inv = params[ensemble]['ainv'] 
        cfgs = os.listdir(data)[1:]
        cfgs.sort()
        N_cf = len(cfgs) # number of configs)
        L = params[ensemble]['XX']
        self.type = 'fourquarks'
        self.prefix = 'fourquarks_'
        self.filename = prop1.filename+'__'+prop2.filename
        fq_str = 'FourQuarkFullyConnected'
        self.scheme = 'gamma'

        self.N_cf = len(cfgs)
        self.fourquark = np.array([np.empty(shape=(self.N_cf,12,12,12,12),dtype='complex128')
                                  for i in range(N_fq)], dtype=object)
        for cf in range(self.N_cf):
            c = cfgs[cf]
            h5_path = f'{data}/{c}/NPR/{self.type}/{self.prefix}{self.filename}.{c}.h5'
            h5_data = h5py.File(h5_path, 'r')[fq_str]
            for i in range(N_fq):
                fourquark_i = h5_data[f'{fq_str}_{i}']['corr'][0,0,:]
                self.fourquark[i][cf,:] = np.array(fourquark_i['re']+fourquark_i['im']*1j).swapaxes(1,
                                          2).swapaxes(5,6).reshape((12,12,12,12))
        self.pOut = [int(x) for x in h5_data[f'{fq_str}_0']['info'].attrs['pOut'][0].decode().rsplit('.')[:-1]]
        self.pIn = [int(x) for x in h5_data[f'{fq_str}_0']['info'].attrs['pIn'][0].decode().rsplit('.')[:-1]]
        self.prop_in = prop1 if prop1.mom==self.pIn else prop2
        self.prop_out = prop1 if prop1.mom==self.pOut else prop2
        self.tot_mom = self.prop_out.tot_mom-self.prop_in.tot_mom
        self.mom_sq = self.prop_in.mom_sq
        self.q = self.mom_sq**0.5

        self.org_gammas = [[h5_data[f'{fq_str}_{i}']['info'].attrs['gammaA'][0].decode(),
                        h5_data[f'{fq_str}_{i}']['info'].attrs['gammaB'][0].decode()]
                        for i in range(N_fq)]
        self.gammas = [[[x for x in list(self.org_gammas[i][0]) if x in list(gamma.keys())],
                        [y for y in list(self.org_gammas[i][1]) if y in list(gamma.keys())]]
                        for i in range(N_fq)]

        self.doublets = {'SS':self.fourquark[self.gammas.index([['I'],['I']])],
                         'PP':self.fourquark[self.gammas.index([['5'],['5']])],
                         'VV':np.sum(np.array([self.fourquark[self.gammas.index([[i],[i]])]
                              for i in dirs]), axis=0),
                         'AA':np.sum(np.array([self.fourquark[self.gammas.index([[i,'5'],[i,'5']])]
                              for i in dirs]), axis=0),
                         'TT':np.sum(np.array(sum([[self.fourquark[self.gammas.index([[dirs[i],
                              dirs[j]],[dirs[i],dirs[j]]])] for j in range(i+1,4)] 
                              for i in range(0,4-1)], [])), axis=0)}
        self.operator = {'VV+AA':self.doublets['VV']+self.doublets['AA'],
                         'VV-AA':self.doublets['VV']-self.doublets['AA'],
                         'SS-PP':self.doublets['SS']-self.doublets['PP'],
                         'SS+PP':self.doublets['SS']+self.doublets['PP'],
                         'TT':self.doublets['TT']}
        
        self.amputated = {}
        in_, out_ = self.prop_in.inv, self.prop_out.inv_out
        for k in operators:
            op = 2*(self.operator[k]-self.operator[k].swapaxes(2,4))
            op_avg = np.array(np.mean(op, axis=0),dtype='complex128')
            self.amputated[k] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                          out_,in_,out_,in_,op_avg) 

        if self.scheme=='gamma':
            self.projector = fq_gamma_projector
            self.F = fq_gamma_F

        self.projected = np.array([[np.einsum('abcd,badc',self.projector[k1],
                          self.amputated[k2]) for k2 in operators]
                          for k1 in operators]) 
        self.bl = bilinear(ensemble, self.prop_in, self.prop_out, cfgs=cfgs)
        self.Z_V = ((self.F/(self.bl.F['V']**2))@np.linalg.inv(self.projected.T/self.bl.projected['V']**2)).real 
        self.Z_V = np.multiply(self.Z_V,mask)
        self.Z_A = ((self.F/self.bl.F['A']**2)@np.linalg.inv(self.projected/self.bl.projected['A']**2)).real
        self.Z_A = np.multiply(self.Z_A,mask)

    def errs(self):
        self.prop_in.errs()
        self.prop_out.errs()
        self.bl.errs()

        self.samples = {k:bootstrap(self.operator[k]) for k in operators}

        self.samples_Z_V = np.zeros(shape=(N_boot,len(operators),len(operators)))
        for k in tqdm(range(N_boot), leave=False):
        #for k in range(N_boot):
            in_, out_ = self.prop_in.samples_inv[k], self.prop_out.samples_out[k]
            samples_amp = {}
            for key in self.samples.keys():
                samples_op = 2*(self.samples[key][k,]-self.samples[key][k,].swapaxes(1,3))
                samples_amp[key] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                        out_,in_,out_,in_,samples_op)
            samples_proj = np.array([[np.einsum('abcd,badc',self.projector[k1],
                                              samples_amp[k2]) for k2 in operators]
                                              for k1 in operators])
            self.samples_Z_V[k,:,:] = (self.F@np.linalg.inv(samples_proj.T/(self.bl.samp_proj['V'][k]**2))).real
        self.Z_V_errs = np.zeros(shape=(len(operators),len(operators)))
        for i in range(len(operators)):
            for j in range(len(operators)):
                diff = self.samples_Z_V[:,i,j]-self.Z_V[i,j]
                self.Z_V_errs[i,j] = (diff.dot(diff)/N_boot)**0.5
