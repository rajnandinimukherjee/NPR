from bag_param_renorm import *

extrap_ensembles = ['C0','C1','C2','M0','M1','M2','M3']

Z_dict = {ens:Z_analysis(ens) for ens in bag_ensembles}
bag_dict = {ens:bag_analysis(ens) for ens in bag_ensembles}


cmap = plt.cm.tab20b
norm = mc.BoundaryNorm(np.linspace(0, 1, len(extrap_ensembles)),cmap.N)
import pdb
def plot_dep(mu, x_key='mpisq', y_key='sigma', xmax=1.7, **kwargs):
    ref = {e:{'mpisq':(Z_dict[e].mpi*Z_dict[e].ainv)**2,
           'sigma':Z_dict[e].scale_evolve(mu,3)[0],
           'bag':np.diag(bag_dict[e].bag_ren[mu]),
           'asq':(1/Z_dict[e].ainv)**2}
           for e in extrap_ensembles}
    ref_err = {e:{'sigma':Z_dict[e].scale_evolve(mu,3)[1],
               'bag':np.diag(bag_dict[e].bag_ren_err[mu])}
               for e in extrap_ensembles}

    #pdb.set_trace()
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlim([0,xmax])
    for e in extrap_ensembles:
        x, y = ref[e][x_key], ref[e][y_key][0,0]
        y_err = ref_err[e][y_key][0,0]
        ax.errorbar(x,y,yerr=y_err,color=cmap(norm(extrap_ensembles.index(
                   e)/len(extrap_ensembles))), label=e, fmt='o')
    
    ax.set_xlabel(x_key)
    ax.legend(bbox_to_anchor=(1.02, 1.03))

    fig, ax = plt.subplots(2,2,sharex=True)
    ax[0,0].set_xlim([0,xmax])
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+1, j+1
        for e in extrap_ensembles:
            x, y = ref[e][x_key], ref[e][y_key][k,l]
            y_err = ref_err[e][y_key][k,l]
            ax[i,j].errorbar(x,y,yerr=y_err, color=cmap(norm(extrap_ensembles.index(
                       e)/len(extrap_ensembles))), label=e, fmt='o')
            if i==1:
                ax[i,j].set_xlabel(x_key)
    handles, labels = ax[0,0].get_legend_handles_labels()

    fig, ax = plt.subplots(2,2,sharex=True)
    ax[0,0].set_xlim([0,xmax])
    for i,j in itertools.product(range(2), range(2)):
        k, l = i+3, j+3
        for e in extrap_ensembles:
            x, y = ref[e][x_key], ref[e][y_key][k,l]
            y_err = ref_err[e][y_key][k,l]
            ax[i,j].errorbar(x,y,yerr=y_err, color=cmap(norm(extrap_ensembles.index(
                       e)/len(extrap_ensembles))), label=e, fmt='o')
            if i==1:
                ax[i,j].set_xlabel(x_key)
    handles, labels = ax[0,0].get_legend_handles_labels()

    filename = f'{y_key}_{x_key}_dep.pdf'
    pp = PdfPages('plots/'+filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    os.system("open plots/"+filename)

def chiral_continuum_ansatz(params, a_sq, mpi_f_m_sq, **kwargs):
    func = params[0]*(1+params[1]*a_sq + params[2]*mpi_f_m_sq)
    if 'addnl_terms' in kwargs.keys():
        if kwargs['addnl_terms']=='a4':
            func += params[0]*(params[3]*(a_sq**2))
        elif kwargs['addnl_terms']=='m4':
            func += params[0]*(params[3]*(mpi_f_m_sq**2))
        elif kwargs['addnl_terms']=='log':
            lambda_f_m_sq = (1.0/f_pi_PDG)**2
            mult = params[3]/(16*np.pi**2)
            func += params[0]*(mult*mpi_f_m_sq*np.log(mpi_f_m_sq/lambda_f_m_sq))
    return func 


class bag_fits:
    N_boot = 200
    operators = ['VVpAA', 'VVmAA', 'SSmPP', 'SSpPP', 'TT']
    def __init__(self, ens_list):
        self.ens_list = ens_list
        self.bag_dict = {e:bag_analysis(e) for e in self.ens_list}
        self.colors = {list(self.bag_dict.keys())[k]:list(
                       mc.TABLEAU_COLORS.keys())[k]
                       for k in range(len(self.ens_list))}
        
    def fit_operator(self, operator, mu, 
                     guess=[1e-1,1e-2,1e-3], **kwargs):
        op_idx = bag_analysis.operators.index(operator)
        dof = len(self.ens_list)-len(guess)

        def pred(params, **akwargs):
            p = [self.bag_dict[e].ansatz(params, **kwargs)
                 for e in self.ens_list]
            return p

        def diff(bag, params, **akwargs):
            return np.array(bag) - np.array(pred(params, **kwargs))

        #==central fit======
        '''something majorly wrong here, need to interpolate to mu=2.0,
        not use the bag_ren at index 2 - fix it!!!!'''

        bags_central = np.array([self.bag_dict[e].interpolate(mu)[0][op_idx]
                                for e in self.ens_list])
        COV = np.diag([self.bag_dict[e].interpolate(mu)[1][op_idx]**2
                       for e in self.ens_list])
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params, **akwargs):
            return L.dot(diff(bags_central, params, fit='central',
                        operator=operator))

        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        pvalue = gammaincc(dof/2,chi_sq/2)
        
        res_btsp = np.zeros(shape=(self.N_boot, len(res.x)))
        bag_btsp = np.array([self.bag_dict[e].interpolate(mu)[2][:,op_idx]
                            for e in self.ens_list])
        for k in range(self.N_boot):

            def LD_btsp(params):
                return L.dot(diff(bag_btsp[:,k], params, fit='btsp',
                            operator=operator, k=k))
            res_k = least_squares(LD_btsp, guess,
                                     ftol=1e-10, gtol=1e-10)
            res_btsp[k,:] = res_k.x

        res_err = [((res_btsp[:,i]-res.x[i]).dot(res_btsp[:,
                  i]-res.x[i])/self.N_boot)**0.5 for i in range(len(res.x))]

        return res.x, chi_sq/dof, pvalue, res_err, res_btsp 

    def plot_fits(self, mu, **kwargs):
        fig, ax = plt.subplots(len(self.operators),2,figsize=(15,30))
        for i in range(len(self.operators)): 
            ax[i,0].title.set_text(self.operators[i])
            ax[i,1].title.set_text(self.operators[i])
            op = self.operators[i]
            params, chi_sq_dof, pvalue, err, btsp = self.fit_operator(op, mu, **kwargs)

            x_asq = np.linspace(0,(1/1.7)**2,50)
            x_mpi_sq = np.linspace(0,0.2, 50)

            for e in self.ens_list:
                mpi_sq = self.bag_dict[e].mpi**2
                f_m_sq = self.bag_dict[e].f_m_ren**2
                mpi_f_m_sq = mpi_sq/f_m_sq
                y_asq = np.array([chiral_continuum_ansatz(params,x,
                                 mpi_f_m_sq,**kwargs) for x in x_asq])
                y_asq_var_diff = np.array([[chiral_continuum_ansatz(btsp[k,:],x_asq[a],
                                 mpi_f_m_sq,**kwargs)-y_asq[a] for k in range(self.N_boot)]
                                 for a in range(len(x_asq))])
                y_asq_err = np.array([(y_asq_var_diff[a,:].dot(y_asq_var_diff[a,
                                     :])/self.N_boot)**0.5
                                     for a in range(len(x_asq))])
                ax[i,0].plot(x_asq, y_asq, color=self.colors[e])
                ax[i,0].fill_between(x_asq, y_asq+y_asq_err, y_asq-y_asq_err,
                                     color=self.colors[e], alpha=0.2)

                a_sq = self.bag_dict[e].ainv**(-2)
                y_mpi = np.array([chiral_continuum_ansatz(params,a_sq,x*a_sq/f_m_sq,**kwargs)
                                  for x in x_mpi_sq])
                y_mpi_var_diff = np.array([[chiral_continuum_ansatz(btsp[k,:],
                                            a_sq,x_mpi_sq[p]*a_sq/f_m_sq,**kwargs)-y_mpi[p]
                                            for k in range(self.N_boot)]
                                            for p in range(len(x_mpi_sq))])
                y_mpi_err = np.array([(y_mpi_var_diff[p,:].dot(y_mpi_var_diff[p,
                                     :])/self.N_boot)**0.5
                                     for p in range(len(x_mpi_sq))])
                ax[i,1].plot(x_mpi_sq, y_mpi, color=self.colors[e])
                ax[i,1].fill_between(x_mpi_sq, y_mpi+y_mpi_err, y_mpi-y_mpi_err,
                                     color=self.colors[e], alpha=0.2)

                a_sq = self.bag_dict[e].ainv**(-2)
                mpi_sq = (self.bag_dict[e].mpi)**2
                bag = self.bag_dict[e].interpolate(mu)[0][i]
                bag_err = self.bag_dict[e].interpolate(mu)[1][i]
                ax[i,0].errorbar([a_sq],[bag],yerr=[bag_err],fmt='o',
                                 label=e,capsize=4,color=self.colors[e],
                                 mfc='None')
                ax[i,1].errorbar([mpi_sq/a_sq],[bag],yerr=[bag_err],fmt='o',
                                 label=e,capsize=4,color=self.colors[e],
                                 mfc='None')

            
            y_phys = chiral_continuum_ansatz(params,0,(mpi_PDG/f_pi_PDG)**2)
            y_phys_var_diff = np.array([chiral_continuum_ansatz(btsp[k,:],0,
                             (mpi_PDG/f_pi_PDG)**2)-y_phys
                             for k in range(self.N_boot)])
            y_phys_err = (y_phys_var_diff[:].dot(y_phys_var_diff[:])/self.N_boot)**0.5
            ax[i,0].errorbar([0],[y_phys],yerr=[y_phys_err],color='k',
                             fmt='o',capsize=4,label='phys')
            ax[i,1].errorbar([mpi_PDG**2],[y_phys],yerr=[y_phys_err],
                             color='k',fmt='o',capsize=4,label='phys')
            ax[i,1].axvline(mpi_PDG**2,color='k',linestyle='dashed')
            ax[i,0].legend()
            ax[i,0].set_xlabel(r'$a^2$ (GeV${}^{-2}$)')
            ax[i,1].legend()
            ax[i,1].set_xlabel(r'$m_{\pi}^2$ (GeV${}^2$)')

            ax[i,0].text(0.4,0.05,r'$\chi^2$/DOF:'+'{:.3f}'.format(chi_sq_dof),
                         transform=ax[i,0].transAxes)
            ax[i,1].text(0.4,0.05,r'$p$-value:'+'{:.3f}'.format(pvalue),
                         transform=ax[i,1].transAxes)
        plt.suptitle(r'$K\overline{K}$ renormalised bag parameter ($\mu=2$ GeV) fits',y=0.90)
            
            
        filename = f'bag_fits.pdf'
        pp = PdfPages('plots/'+filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        os.system("open plots/"+filename)



        


        
































