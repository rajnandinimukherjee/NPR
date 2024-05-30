from basics import *
from fourquark import *
from valence import *


class bilinear_analysis:
    keys = ['S', 'P', 'V', 'A', 'T', 'm']
    N_boot = N_boot

    def __init__(self, ensemble, sea_mass_idx=0,
            scheme='gamma', cfgs=None, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.cfgs = info['NPR_cfgs'] if cfgs==None else cfgs
        self.ainv = params[ensemble]['ainv']
        self.asq = (1/self.ainv)**2
        self.sea_mass = '{:.4f}'.format(info['masses'][sea_mass_idx])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(len(info['masses'])) if k!=sea_mass_idx]

        gauge_count = len(info['baseactions'])
        self.actions = [info['gauges'][i]+'_'+info['baseactions'][i] for
                        i in range(gauge_count)]
        self.all_masses = [self.sea_mass]+self.non_sea_masses
        self.scheme = scheme

        self.momenta, self.Z = {}, {}

    def NPR(self, masses, action=(0, 0), scheme=1,
            massive=False, mres=None, Z_A_input=None, **kwargs):
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]

        self.momenta[action][masses] = {}
        self.Z[action][masses] = {}

        self.data = path+self.ens
        self.data += 'S/results' if self.ens[-1] not in ['M','S'] else '/results'
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.bl_list = common_cf_files(
                self.data, self.cfgs, 'bilinears', prefix='bi_')

            results = {}

            desc = '_'.join([self.ens, m1, m2, str(action)])
            for b in tqdm(range(len(self.bl_list)), leave=False, desc=desc):
                prop1_name, prop2_name = self.bl_list[b].split('__')
                prop1_info, prop2_info = decode_prop(
                    prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am'] ==
                              m1 and prop2_info['am'] == m2)
                condition2 = (prop1_info['prop'] ==
                              a1 and prop2_info['prop'] == a2)

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name, cfgs=self.cfgs)
                    prop2 = external(self.ens, filename=prop2_name, cfgs=self.cfgs)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared == prop2.momentum_squared
                    condition4 = prop1.momentum_squared == scheme * \
                        np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        bl = bilinear(self.ens, prop1, prop2,
                                      cfgs=self.cfgs, mres=mres,
                                      Z_A_input=Z_A_input,
                                      **kwargs)
                        mom = bl.q
                        if mom not in results.keys():
                            bl.NPR(massive=massive, **kwargs)
                            results[mom] = bl.Z

            self.momenta[action][(m1, m2)] = sorted(results.keys())
            self.Z[action][(m1, m2)] = np.array(
                [results[mom] for mom in self.momenta[action][(m1, m2)]])

    def save_NPR(self, filename_add='', **kwargs):
        for action in self.Z.keys():
            for masses in self.Z[action].keys():
                if len(self.Z[action][masses])==0:
                    continue
                else:
                    a1, a2 = action
                    filename = f'NPR/action{a1}_action{a2}/'
                    filename += '__'.join(['NPR', self.ens,
                        params[self.ens]['baseactions'][a1],
                        params[self.ens]['baseactions'][a2]])
                    filename += filename_add
                    filename += '.h5'
                    f = h5py.File(filename, 'a')

                    if str(masses)+'/bilinear' in f.keys():
                        del f[f'{str(masses)}/bilinear']
                    m_grp = f.create_group(f'{str(masses)}/bilinear')
                    m_grp.attrs['cfgs'] = self.cfgs

                    mom = m_grp.create_dataset(
                        'ap', data=self.momenta[action][masses])
                    for current in self.Z[action][masses][0].keys():
                        c_grp = m_grp.create_group(current)
                        Zs = stat(
                            val=[self.Z[action][masses][mom][current].val
                                 for mom in range(len(self.Z[action][masses]))],
                            err=[self.Z[action][masses][mom][current].err
                                 for mom in range(len(self.Z[action][masses]))],
                            btsp=np.array([self.Z[action][masses][mom][current].btsp
                                           for mom in range(len(self.Z[action][masses]))]).swapaxes(0, 1))

                        central = c_grp.create_dataset('central', data=Zs.val)
                        err = c_grp.create_dataset('errors', data=Zs.err)
                        btsp = c_grp.create_dataset('bootstrap', data=Zs.btsp)

        print(f'Saved {self.ens} bilinear NPR results.')

    def NPR_all(self, massive=False, save=True, renorm='mSMOM', **kwargs):
        if massive:
            self.valence = valence(self.ens)
            self.valence.compute_amres(load=False)
            self.valence.compute_Z_A(load=False)
            self.amres_dict = self.valence.amres
            self.Z_A_dict = self.valence.Z_A

            self.momenta[(0, 0)] = {}
            self.Z[(0, 0)] = {}
            self.NPR((self.sea_mass, self.sea_mass),
                     massive=massive, renorm=renorm,
                     mres=self.amres_dict[self.sea_mass],
                     Z_A_input=self.Z_A_dict[self.sea_mass])
            for mass in self.non_sea_masses:
                self.NPR((mass, mass), massive=massive,
                        renorm=renorm, mres=self.amres_dict[mass],
                        Z_A_input=self.Z_A_dict[mass])
            filename_add = f'_{renorm}'
        else:
            N_a = len(self.actions)
            for a1, a2 in itertools.product(range(N_a), range(N_a)):
                self.momenta[(a1, a2)] = {}
                self.Z[(a1, a2)] = {}
                self.NPR((self.sea_mass, self.sea_mass), action=(a1, a2))
            if N_a == 2:
                self.merge_mixed()
            filename_add = '_{self.scheme}'

        if save:
            self.save_NPR(filename_add=filename_add)

    def merge_mixed(self, **kwargs):
        mass_combinations = self.momenta[(0, 1)].keys()

        for masses in list(mass_combinations):
            momenta = self.momenta[(0, 1)][masses]
            res1 = self.Z[(0, 1)][masses]
            res2 = self.Z[(1, 0)][masses]
            self.Z[(0, 1)][masses] = [
                {c: (res1[m][c]+res2[m][c])/2.0
                 for c in res1[m].keys()}
                for m in range(len(momenta))]

        self.Z.pop((1, 0))
        self.momenta.pop((1, 0))


class fourquark_analysis:
    N_boot = N_boot

    def __init__(self, ensemble, sea_mass_idx=0, scheme='gamma', 
                 cfgs=None, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.cfgs = info['NPR_cfgs'] if cfgs==None else cfgs
        self.sea_mass = '{:.4f}'.format(info['masses'][sea_mass_idx])
        self.scheme = scheme

        gauge_count = len(info['baseactions'])
        self.actions = [info['gauges'][i]+'_'+info['baseactions'][i] for
                        i in range(gauge_count)]
        self.momenta, self.Z = {}, {}

    def NPR(self, masses, action=(0, 0), scheme=1, **kwargs):
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]

        self.momenta[action] = {masses: {}}
        self.Z[action] = {masses: {}}

        self.data = path+self.ens+'/results/'
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.fq_list = common_cf_files(
                    self.data, self.cfgs, 'fourquarks', prefix='fourquarks_')

            results = {}

            desc = '_'.join([self.ens, m1, m2, str(action)])
            for f in tqdm(range(len(self.fq_list)), leave=False, desc=desc):
                prop1_name, prop2_name = self.fq_list[f].split('__')
                prop1_info, prop2_info = decode_prop(
                    prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am'] ==
                              m1 and prop2_info['am'] == m2)
                condition2 = (prop1_info['prop'] ==
                              a1 and prop2_info['prop'] == a2)

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name, cfgs=self.cfgs)
                    prop2 = external(self.ens, filename=prop2_name, cfgs=self.cfgs)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared == prop2.momentum_squared
                    condition4 = prop1.momentum_squared == scheme * \
                        np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        fq = fourquark(self.ens, prop1, prop2, cfgs=self.cfgs)
                        if fq.q not in results.keys():
                            fq.NPR()
                            results[fq.q] = fq.Z

                self.momenta[action][masses] = sorted(results.keys())
                self.Z[action][masses] = np.array(
                    [results[mom] for mom in self.momenta[action][masses]])

    def save_NPR(self, filename_add='', **kwargs):
        for action in self.Z.keys():
            for masses in self.Z[action].keys():
                if len(self.Z[action][masses])==0:
                    continue
                else:
                    a1, a2 = action
                    filename = f'NPR/action{a1}_action{a2}/'
                    filename += '__'.join(['NPR', self.ens,
                        params[self.ens]['baseactions'][a1],
                        params[self.ens]['baseactions'][a2],
                        self.scheme])
                    filename += filename_add
                    filename += '.h5'
                    f = h5py.File(filename, 'a')

                    if str(masses)+'/fourquark' in f.keys():
                        del f[f'{str(masses)}/fourquark']
                    m_grp = f.create_group(f'{str(masses)}/fourquark')
                    m_grp.attrs['cfgs'] = self.cfgs

                    mom = m_grp.create_dataset(
                        'ap', data=self.momenta[action][masses])
                    Zs = stat(
                        val=[self.Z[action][masses][mom].val
                             for mom in range(len(self.Z[action][masses]))],
                        err=[self.Z[action][masses][mom].err
                             for mom in range(len(self.Z[action][masses]))],
                        btsp=np.array([self.Z[action][masses][mom].btsp
                                       for mom in range(len(self.Z[action][masses]))]).swapaxes(0, 1))

                    central = m_grp.create_dataset('central', data=Zs.val)
                    err = m_grp.create_dataset('errors', data=Zs.err)
                    btsp = m_grp.create_dataset('bootstrap', data=Zs.btsp)

        print(f'Saved {self.ens} fourquark NPR results.')

    def NPR_all(self, save=True, **kwargs):
        N_a = len(self.actions)
        for a1, a2 in itertools.product(range(N_a), range(N_a)):
            self.NPR((self.sea_mass, self.sea_mass), action=(a1, a2))
        if N_a == 2:
            self.merge_mixed()
        if save:
            self.save_NPR()

    def merge_mixed(self, **kwargs):
        mass_combinations = self.momenta[(0, 1)].keys()

        for masses in list(mass_combinations):
            momenta = self.momenta[(0, 1)][masses]
            res1 = self.Z[(0, 1)][masses]
            res2 = self.Z[(1, 0)][masses]
            self.Z[(0, 1)][masses] = [(res1[m]+res2[m])/2.0
                                      for m in range(len(momenta))]

        self.Z.pop((1, 0))
        self.momenta.pop((1, 0))
