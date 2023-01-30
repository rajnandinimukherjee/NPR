from NPR_structures import *
from matplotlib.ticker import FormatStrFormatter
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
        else:
            print('Loading NPR bilinear data from '+loadpath)
            self.momenta, self.avg_results, self.avg_errs = pickle.load(
                                                            open(loadpath, 'rb')) 
            any_key = list(self.momenta.keys())[0]
            self.all_masses = list(self.momenta[any_key].keys()) 

    
    def NPR(self, action=(0,0), scheme=1, massive=True, save=False, **kwargs): 
        a1, a2 = action
        self.avg_results[action] = {}
        self.avg_errs[action] = {}
        self.momenta[action] = {}

        self.all_masses = [self.sea_mass] + self.non_sea_masses*massive

        self.data = path+self.ens
        if not os.path.isdir(self.data):
            print('NPR data for this ensemble could not be found on this machine')
        else:
            self.bl_list = common_cf_files(self.data, 'bilinears', prefix='bi_')
            for m in self.all_masses:
                results = {c:{} for c in currents}
                errs = {c:{} for c in currents}

                for b in tqdm(range(len(self.bl_list)), leave=False):
                    prop1_name, prop2_name = self.bl_list[b].split('__')
                    prop1_info, prop2_info = decode_prop(prop1_name), decode_prop(prop2_name)

                    condition1 = (prop1_info['am']==m and prop2_info['am']==m)
                    condition2 = (prop1_info['prop']==self.actions[a1] and 
                                  prop2_info['prop']==self.actions[a2])

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

                self.momenta[action][m] = sorted(results['S'].keys())
                self.avg_results[action][m] = {k:np.array([np.mean(v[mom])
                                               for mom in self.momenta[action][m]])
                                               for k,v in results.items()}
                self.avg_errs[action][m] = {k:np.array([np.mean(v[mom])
                                            for mom in self.momenta[action][m]])
                                            for k,v in errs.items()}
    
    def save_NPR(self, addl_txt='', **kwargs):
        filename = 'pickles/'+self.ens+'_bl.p'
        pickle.dump([self.momenta, self.avg_results, self.avg_errs],
                    open(filename,'wb'))
        print('Saved bilinear NPR results to '+filename)


    def plot_masswise(self, action=(0,0), save=False, **kwargs):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for m in self.all_masses:
                mom = self.momenta[action][m]
                res = self.avg_results[action][m][c]
                err = self.avg_errs[action][m][c]
                val_col.scatter(mom, res, label=m)
                err_col.scatter(mom, err, label=m)
            val_col.title.set_text(c)
            handles, labels = err_col.get_legend_handles_labels()
        ax[1,2].set_xlabel('$|q|/GeV$')
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(self.ens+' comparison of masses')
        fig.tight_layout()

        if save:
            plt.savefig('plots/'+self.ens+'_mass_comp_bl.pdf')
            print('Plot saved to plots/'+self.ens+'_mass_comp_bl.pdf')

    def plot_actionwise(self, mass=None, save=False, **kwargs):
        if (mass==None) or mass not in self.all_masses:
            mass = self.sea_mass
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))
        action_combinations = self.momenta.keys()
        for i in range(5):
            val_col, err_col = ax[:,i]
            err_col.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            c = currents[i]
            for action in action_combinations:
                mom = self.momenta[action][mass]
                res = self.avg_results[action][mass][c]
                err = self.avg_errs[action][mass][c]
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

