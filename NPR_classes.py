from basics import *
from fourquark import *


class bilinear_analysis:
    keys = ['S', 'P', 'V', 'A', 'T', 'm']
    N_boot = N_boot

    def __init__(self, ensemble, mres=False, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.ainv = params[ensemble]['ainv']
        self.asq = (1/self.ainv)**2
        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1, len(info['masses']))]

        gauge_count = len(info['baseactions'])
        self.actions = [info['gauges'][i]+'_'+info['baseactions'][i] for
                        i in range(gauge_count)]
        self.all_masses = [self.sea_mass]+self.non_sea_masses
        self.mres = mres

        self.momenta, self.Z = {}, {}

    def NPR(self, masses, action=(0, 0), scheme=1, massive=False, **kwargs):
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]

        self.momenta[action] = {masses: {}}
        self.Z[action] = {masses: {}}

        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.bl_list = common_cf_files(
                self.data, 'bilinears', prefix='bi_')

            results = {}

            for b in tqdm(range(len(self.bl_list)), leave=False):
                prop1_name, prop2_name = self.bl_list[b].split('__')
                prop1_info, prop2_info = decode_prop(
                    prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am'] ==
                              m1 and prop2_info['am'] == m2)
                condition2 = (prop1_info['prop'] ==
                              a1 and prop2_info['prop'] == a2)

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name)
                    prop2 = external(self.ens, filename=prop2_name)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared == prop2.momentum_squared
                    condition4 = prop1.momentum_squared == scheme * \
                        np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        bl = bilinear(self.ens, prop1, prop2, mres=self.mres)
                        mom = bl.q*self.ainv
                        if mom not in results.keys():
                            bl.NPR(massive=massive, **kwargs)
                            results[mom] = bl.Z

            self.momenta[action][(m1, m2)] = sorted(results.keys())
            self.Z[action][(m1, m2)] = np.array(
                [results[mom] for mom in self.momenta[action][(m1, m2)]])

    def save_NPR(self, filename, **kwargs):
        f = h5py.File(filename, 'a')
        for action in self.Z.keys():
            for masses in self.Z[action].keys():
                if str(action)+'/'+self.ens+'/'+str(masses) in f.keys():
                    del f[str(action)][self.ens][str(masses)]
                m_grp = f.create_group(
                    str(action)+'/'+self.ens+'/'+str(masses))
                mom = m_grp.create_dataset(
                    'momenta', data=self.momenta[action][masses])
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

        print('Saved bilinear NPR results to '+filename)

    def NPR_all(self, massive=False, save=True, renorm='mSMOM', **kwargs):
        if massive:
            self.NPR((self.sea_mass, self.sea_mass),
                     massive=massive, renorm=renorm)
            for mass in self.non_sea_masses:
                self.NPR((mass, mass), massive=massive, renorm=renorm)
            filename = 'bilinear_Z_qslash.h5'
        else:
            N_a = len(self.actions)
            for a1, a2 in itertools.product(range(N_a), range(N_a)):
                self.NPR((self.sea_mass, self.sea_mass), action=(a1, a2))
            if N_a == 2:
                self.merge_mixed()
            filename = 'bilinear_Z_gamma.h5'

        if save:
            self.save_NPR(filename)

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
    N_boot = 200

    def __init__(self, ensemble, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.sea_mass = '{:.4f}'.format(info['masses'][0])

        self.actions = [info['gauges'][0]+'_'+info['baseactions'][0],
                        info['gauges'][-1]+'_'+info['baseactions'][-1]]

        self.momenta, self.Z = {}, {}

    def NPR(self, masses, action=(0, 0), scheme=1, **kwargs):
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]

        self.momenta[action] = {masses: {}}
        self.Z[action] = {masses: {}}

        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.fq_list = common_cf_files(self.data,
                                           'fourquarks', prefix='fourquarks_')

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
                    prop1 = external(self.ens, filename=prop1_name)
                    prop2 = external(self.ens, filename=prop2_name)
                    mom_diff = prop1.total_momentum-prop2.total_momentum

                    condition3 = prop1.momentum_squared == prop2.momentum_squared
                    condition4 = prop1.momentum_squared == scheme * \
                        np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        fq = fourquark(self.ens, prop1, prop2)
                        if fq.q not in results.keys():
                            fq.NPR()
                            results[fq.q] = fq.Z

                self.momenta[action][masses] = sorted(results.keys())
                self.Z[action][masses] = np.array(
                    [results[mom] for mom in self.momenta[action][masses]])

    def save_NPR(self, filename='fourquarks_Z.h5', **kwargs):
        f = h5py.File(filename, 'a')
        for action in self.Z.keys():
            for masses in self.Z[action].keys():
                if str(action)+'/'+self.ens+'/'+str(masses) in f.keys():
                    del f[str(action)][self.ens][str(masses)]
                m_grp = f.create_group(
                    str(action)+'/'+self.ens+'/'+str(masses))
                mom = m_grp.create_dataset(
                    'momenta', data=self.momenta[action][masses])
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

    def NPR_all(self, save=True, **kwargs):
        N_a = len(self.actions)
        for a1, a2 in itertools.product(range(N_a), range(N_a)):
            self.NPR((self.sea_mass, self.sea_mass), action=(a1, a2))
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
