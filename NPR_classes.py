from NPR_structures import *
from matplotlib.ticker import FormatStrFormatter
import itertools
import pickle
import os

phys_ens = ['C0', 'M0']

class bilinear_analysis:
    def __init__(self, ensemble, loadpath=None, **kwargs):

        self.ens = ensemble
        info = params[self.ens]
        self.sea_mass = '{:.4f}'.format(info['masses'][0])
        self.non_sea_masses = ['{:.4f}'.format(info['masses'][k])
                               for k in range(1,len(info['masses']))]

        self.actions = [info['gauges'][0]+'_'+info['baseactions'][0],
                   info['gauges'][1]+'_'+info['baseactions'][1]]

        if loadpath==None:
            self.momenta, self.avg_results, self.avg_errs = {}, {}, {}
            for data in [self.momenta, self.avg_results, self.avg_errs]:
                for a1, a2 in itertools.product([0,1],[0,1]):
                    data[(a1,a2)] = {}

        else:
            print('Loading NPR bilinear data from '+loadpath)
            self.momenta, self.avg_results, self.avg_errs = pickle.load(
                                                            open(loadpath, 'rb')) 
            self.all_masses = list(self.momenta[(0,0)].keys()) 

    
    def NPR(self, masses, action=(0,0), scheme=1, **kwargs): 
        m1, m2 = masses
        a1, a2 = action
        a1, a2 = self.actions[a1], self.actions[a2]
        
        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.bl_list = common_cf_files(self.data, 'bilinears', prefix='bi_')

            results = {c:{} for c in currents}
            errs = {c:{} for c in currents}

            for b in tqdm(range(len(self.bl_list)), leave=False):
                prop1_name, prop2_name = self.bl_list[b].split('__')
                prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

                condition1 = (prop1_info['am']==m1 and prop2_info['am']==m2)
                condition2 = (prop1_info['prop']==a1 and prop2_info['prop']==a2) 

                if (condition1 and condition2):
                    prop1 = external(self.ens, filename=prop1_name)
                    prop2 = external(self.ens, filename=prop2_name)
                    mom_diff = prop1.tot_mom-prop2.tot_mom

                    condition3 = prop1.mom_sq==prop2.mom_sq
                    condition4 = prop1.mom_sq==scheme*np.linalg.norm(mom_diff)**2

                    if (condition3 and condition4):
                        bl = bilinear(self.ens, prop1, prop2)
                        bl.errs()
                        for c in currents:
                            if bl.q not in results[c].keys():
                                results[c][bl.q] = [(bl.projected[c]/bl.F[c]).real]
                                errs[c][bl.q] = [(bl.proj_err[c].real)]

                self.momenta[action][(m1,m2)] = sorted(results['S'].keys())
                self.avg_results[action][(m1,m2)] = {k:np.array([np.mean(v[mom])
                                               for mom in self.momenta[action][(m1,m2)]])
                                               for k,v in results.items()}
                self.avg_errs[action][(m1,m2)] = {k:np.array([np.mean(v[mom])
                                            for mom in self.momenta[action][(m1,m2)]])
                                            for k,v in errs.items()}
    
    def save_NPR(self, addl_txt='', **kwargs):
        filename = 'pickles/'+self.ens+'_bl.p'
        pickle.dump([self.momenta, self.avg_results, self.avg_errs],
                    open(filename,'wb'))
        print('Saved bilinear NPR results to '+filename)

    def NPR_all(self, massive=False, save=True, **kwargs):
        for a1,a2 in itertools.product([0,1],[0,1]):
            self.NPR((self.sea_mass, self.sea_mass), action=(a1,a2))
        if massive:
            for mass in self.non_sea_masses:
                self.NPR((self.sea_mass, mass))
                self.NPR((mass, self.sea_mass))
                self.NPR((mass, mass))

        if save:
            self.save_NPR()

    def plot_masswise(self, action=(0,0), save=False, **kwargs):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        if 'mass_combination' in kwargs.keys():
            if kwargs['mass_combination']=='nondeg':
                plot_masses = []
                for m in [self.sea_mass]+self.non_sea_masses:
                    plot_masses.append((m, self.sea_mass))
            elif kwargs['mass_combination']=='deg':
                plot_masses = []
                for m in [self.sea_mass]+self.non_sea_masses:
                    plot_masses.append((m, m))
            else:
                plot_masses = kwargs['mass_combination']
        else:
            plot_masses = self.momenta[action].keys()
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for m in plot_masses:
                label = '('+m[0]+','+m[1]+')'
                mom = self.momenta[action][m]
                res = self.avg_results[action][m][c]
                err = self.avg_errs[action][m][c]
                val_col.scatter(mom, res, label=label)
                err_col.scatter(mom, err, label=label)
            val_col.title.set_text(c)
            handles, labels = err_col.get_legend_handles_labels()
        ax[1,2].set_xlabel('$|q|/GeV$')
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(self.ens+' comparison of masses')
        fig.tight_layout()

        if save:
            plt.savefig('plots/'+self.ens+'_mass_comp_bl.pdf')
            print('Plot saved to plots/'+self.ens+'_mass_comp_bl.pdf')

    def plot_actionwise(self, mass, save=False, **kwargs):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        action_combinations = self.momenta.keys()
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for action in action_combinations:
                mom = self.momenta[action][(m1,m2)]
                res = self.avg_results[action][(m1,m2)][c]
                err = self.avg_errs[action][(m1,m2)][c]
                val_col.scatter(mom, res, label=action)
                err_col.scatter(mom, err, label=action)
            val_col.title.set_text(c)
            handles, labels = err_col.get_legend_handles_labels()
        ax[1,2].set_xlabel('$|q|/GeV$')
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(self.ens+' comparison of actions')
        fig.tight_layout()

        if save:
            plt.savefig('plots/'+self.ens+'_action_comp_bl.pdf')
            print('Plot saved to plots/'+self.ens+'_action_comp_bl.pdf')

