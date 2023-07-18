from NPR_classes import *
from basics import *
from eta_c import *

ens_list = list(eta_c_data.keys())

argument = ''#int(sys.argv[1])
for ens in ens_list:
    if ens in valence_ens:
        e = etaCvalence(ens, mres=False)
        e.toDict(keys=list(e.mass_comb.keys()))

mu_chosen = 2.0
filename = f'plots/combined_massive_Z{argument}.pdf'
pdf = PdfPages(filename)
fig, ax = plt.subplots(nrows=3,ncols=len(ens_list),
                       sharex='col',sharey='row',
                       squeeze=True,figsize=(12,15))
plt.subplots_adjust(hspace=0,wspace=0)


#====plotting eta_c data from f_D paper=========================================
for ens in ens_list:
    ens_idx = ens_list.index(ens)
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']

    ax[0,ens_idx].errorbar(x, y*ainv, yerr=yerr*ainv,
                           fmt='o', capsize=4, mfc='None',
                           color=color_list[ens_idx])   
    ax[0,ens_idx].scatter(x[:-1], y[:-1]*ainv, 
                          marker='x',lw=1,color='k',
                          label='used in extrap')   
    ymin0, ymax0 = ax[0,0].get_ylim()
    ymin, ymax = ax[0,ens_idx].get_ylim()
    xmin, xmax = ax[0,ens_idx].get_xlim()
    if ymax0<ymax:
        ymax0 = ymax

    for eta in eta_stars:
        eta_idx = eta_stars.index(eta)
        m_q_star, m_q_star_err = interpolate_eta_c(ens,eta)

        ax[0,ens_idx].axvspan(m_q_star-m_q_star_err,
                            m_q_star+m_q_star_err,
                            color='k', alpha=0.1)
        label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'
        ax[0,ens_idx].hlines(eta,-0.05,m_q_star,label=label,
                             color=color_list[eta_idx+3])
        ax[0,ens_idx].vlines(m_q_star,0,eta,linestyle='dashed',
                           color=color_list[eta_idx+3])
    ax[0,ens_idx].set_title(ens)
    ax[0,ens_idx].legend(loc='lower right')
    ax[0,ens_idx].set_xlim([-0.05,xmax])

ax[0,0].set_ylabel(r'$M_{\eta_C}$ (GeV)')
for ax_i in ax[0,:]:
    ax_i.set_ylim([0,ymax0])

plt.text(1.03,2.3,'F1S using data from $f_D$ paper',
         transform=plt.gca().transAxes,color='r',rotation=90)
                

#====plotting Z_m(mu_chosen) extrapolation at m_q_stars=========================
def Z_m_ansatz(params, am, key='m', **kwargs):
    if key=='m':
        return params[0] + params[1]*am + params[2]/am
    else:
        return params[0]*am + (am**2)*params[1] + params[2]


ens_dict = {ens:{'mSMOM':bilinear_analysis(ens,
                 loadpath=f'pickles/{ens}_bl_massive_mSMOM.p'),
                 'SMOM':bilinear_analysis(ens,
                 loadpath=f'pickles/{ens}_bl_massive_SMOM.p')}
            for ens in ens_list}

#====fit choices=======
ens_dict['C1']['fit_idx'] = [0,1,2,3]
ens_dict['M1']['fit_idx'] = [0,1,2,3,4]
ens_dict['F1S']['fit_idx'] = [0,1,2,3,4]

for key in ['m','mam_q']:
    key_idx = ['m','mam_q'].index(key)+1
    ax[key_idx,0].set_ylabel('$Z_'+key+'(\mu=${:.3f} GeV)'.format(
                         mu_chosen))
    if key=='m':
        plt.text(1.03, 1.2,r'$y=\alpha/x + \beta + \gamma x$',
                transform=plt.gca().transAxes,color='gray',rotation=90)
    else:
        plt.text(1.03, 0.2,r'$y=\alpha + \beta x + \gamma x^2$',
                transform=plt.gca().transAxes,color='gray',rotation=90)

    for ens in ['M1','C1','F1S']:
        ens_idx = ens_list.index(ens)
        ainv = ens_dict[ens]['mSMOM'].ainv
        eta_star_dict = {eta:interpolate_eta_c(ens,eta)
                         for eta in eta_stars}
        x, y, e = ens_dict[ens]['mSMOM'].massive_Z_plots(key=key,
                                          mu=mu_chosen, passinfo=True)

        x_, y_, e_ = ens_dict[ens]['SMOM'].massive_Z_plots(key=key,
                                          mu=mu_chosen, passinfo=True)

        ax[key_idx,ens_idx].errorbar(x,y,yerr=e,fmt='o',capsize=4,
                             label='mSMOM',
                             color=color_list[ens_idx],
                             zorder=1)
        adjusted_x = (x[1]+x[0])/2.0
        ax[key_idx,ens_idx].errorbar(adjusted_x,y_[[0]],yerr=e_[[0]],
                             fmt='o',capsize=4,
                             label='SMOM',
                             color='k',
                             zorder=0)
        if ens in valence_ens:
            ax[2,ens_idx].set_xlabel(r'$a_{'+ens+r'}m_{eff}$')
        else:
            ax[2,ens_idx].set_xlabel(r'$a_{'+ens+r'}m_q$')

        fit_idx = ens_dict[ens]['fit_idx']
        ax[key_idx,ens_idx].scatter(x[fit_idx],y[fit_idx],
                            color='k',marker='x',lw=1,
                            label='included in fit',
                            zorder=2)
        def diff(params):
            return y[fit_idx] - Z_m_ansatz(params, x[fit_idx], key=key)
        
        cov = np.diag(e[fit_idx]**2)
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)

        def LD(params, **akwargs):
            return L.dot(diff(params))
        
        guess = [1,1,1]
        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        dof = len(fit_idx)-len(guess)
        pvalue = gammaincc(dof/2,chi_sq/2)

        xmin, xmax = ax[key_idx,ens_idx].get_xlim()
        x_grain = np.linspace(x[0],xmax,500)
        ax[key_idx,ens_idx].plot(x_grain,Z_m_ansatz(res.x,x_grain,key=key),#**0.5),
                    label='fit $p$-value:{:.2f}'.format(pvalue),
                    color='tab:gray',zorder=0)

        m_q_stars = np.array([v[0] for k,v in eta_star_dict.items()])
        Z_ms = Z_m_ansatz(res.x,m_q_stars,key=key)
        np.random.seed(seed)
        m_q_btsp = np.array([np.random.normal(v[0],v[1],N_boot)
                           for k,v in eta_star_dict.items()])
        Z_m_btsp = np.array([Z_m_ansatz(res.x,m_q_btsp[:,k],key=key)
                               for k in range(N_boot)])
        Z_m_errs = [st_dev(Z_m_btsp[:,i],mean=Z_ms[i])
                for i in range(len(Z_ms))]

        ymin, ymax = ax[key_idx,1].get_ylim()
        for eta in eta_stars:
            eta_idx = eta_stars.index(eta)
            val, err = eta_star_dict[eta]
            eta_label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'
            ax[key_idx,ens_idx].vlines(val**1,0,ymax,linestyle='dashed',
                       color=color_list[eta_idx+3])
            ax[key_idx,ens_idx].axvspan((val-err)**1,(val+err)**1,
                                color=color_list[eta_idx+3],
                                alpha=0.1,zorder=3)
            ax[key_idx,ens_idx].errorbar(val**1,Z_ms[eta_idx],
                                 yerr=Z_m_errs[eta_idx],
                                 color=color_list[eta_idx+3],
                                 fmt='o',capsize=4,zorder=4)
        ax[key_idx,ens_idx].legend()
        ax[key_idx,ens_idx].set_ylim([0,ymax])

        ens_dict[ens][f'Z_{key}'] = Z_ms
        ens_dict[ens][f'Z_{key}_err'] = Z_m_errs
        
        if key=='m':
            #===m_C renormalisation=====================
            m_C, m_C_err = interpolate_eta_c(ens,eta_PDG)
            ens_dict[ens]['m_C_ren'] = Z_ms*m_C*ainv
            np.random.seed(seed)
            m_C_btsp = np.random.normal(m_C, m_C_err, N_boot)
            m_C_ren_btsp = np.array([Z_m_btsp[k,:]*m_C_btsp[k]*ainv
                                     for k in range(N_boot)])
            ens_dict[ens]['m_C_ren_err'] = [st_dev(m_C_ren_btsp[:,i],Z_ms[i]*m_C*ainv)
                            for i in range(len(Z_ms))]
            ens_dict[ens]['m_C_ren_chiral'] = y[0]*m_C*ainv
            np.random.seed(seed)
            Z_m_chiral_btsp = np.random.normal(y[0], e[0], N_boot)
            m_C_chiral_btsp = np.array([Z_m_chiral_btsp[k]*m_C_btsp[k]*ainv
                                        for k in range(N_boot)])
            ens_dict[ens]['m_C_ren_chiral_err'] = st_dev(m_C_chiral_btsp,
                                                  mean=y[0]*m_C*ainv)
        if key=='mam_q':
            #===m_q renormalisation=====================
            ens_dict[ens]['m_q_ren'] = Z_ms*ainv 
            m_q_ren_btsp = Z_m_btsp*ainv
            ens_dict[ens]['m_q_ren_err'] = np.array([st_dev(m_q_ren_btsp[:,i],
                                       mean=(Z_ms*ainv)[i])
                                       for i in range(len(Z_ms))])
            ens_dict[ens]['m_q_ren_chiral'] = y[0]*ainv
            m_q_chiral_btsp = Z_m_chiral_btsp*ainv
            ens_dict[ens]['m_q_ren_chiral_err'] = st_dev(m_q_chiral_btsp,
                                                  mean=y[0]*ainv)

fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pdf, format='pdf')
pdf.close()
plt.close('all')
os.system('open '+filename)


#====plotting renormalised charm quark mass and m_q for m-bar===================
#====continuum extrapolation for renormalised m=====
def continuum_ansatz(params, a_sq, **kwargs):
    return params[0]*(1 + params[1]*a_sq + params[2]*(a_sq**2))

cont_dict = {'m_C':{'val':[], 'err':[], 'btsp':[]},
             'm_q':{'val':[], 'err':[], 'btsp':[]}}

for key in ['m_C','m_q']:
    plt.figure()
    for ens in ens_list:
        ens_idx = ens_list.index(ens)
        y = ens_dict[ens][f'{key}_ren']
        e = ens_dict[ens][f'{key}_ren_err']
        x = [ens_dict[ens]['mSMOM'].asq]*len(y)
        plt.errorbar(x,y,yerr=e,fmt='o',capsize=4,
                     color=color_list[ens_idx],label=ens)
        #===massless scheme renorm====================
        #plt.errorbar([ens_dict[ens]['bl_obj'].asq],
        #             [ens_dict[ens][f'{key}_ren_chiral']],
        #             fmt='D',capsize=4,color=color_list[ens_idx])
                     

    x = np.array([v['mSMOM'].asq for k,v in ens_dict.items()])
    for eta in eta_stars:
        eta_idx = eta_stars.index(eta)
        y = np.array([v[f'{key}_ren'][eta_idx] for k,v in ens_dict.items()])
        e = np.array([v[f'{key}_ren_err'][eta_idx] for k,v in ens_dict.items()])

        #===central fit====
        def diff(y, params):
            return y - continuum_ansatz(params,x) 

        COV = np.diag(e**2)
        L_inv = np.linalg.cholesky(COV)
        L = np.linalg.inv(L_inv)

        def LD(params):
            return L.dot(diff(y, params))
        
        guess = [1,-1,-1]
        res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
        chi_sq = LD(res.x).dot(LD(res.x))
        dof = len(x)-len(guess)
        pvalue = gammaincc(dof/2,chi_sq/2)

        a0_pred = continuum_ansatz(res.x,0)
        cont_dict[key]['val'].append(a0_pred)

        #===bootstrap=====
        np.random.seed(seed)
        y_btsp = np.array([np.random.normal(y[i],e[i],N_boot)
                           for i in range(len(y))])
        a0_pred_btsp = np.zeros(N_boot)
        res_btsp = np.zeros(shape=(N_boot,len(guess)))
        for k in range(N_boot):
            def LD_btsp(params):
                return L.dot(diff(y_btsp[:,k], params))
            res_k = least_squares(LD_btsp, guess, ftol=1e-10, gtol=1e-10)
            a0_pred_btsp[k] = continuum_ansatz(res_k.x,0) 
            res_btsp[k,:] = res_k.x
        cont_dict[key]['btsp'].append(a0_pred_btsp)

        a0_pred_err = st_dev(a0_pred_btsp,mean=a0_pred)
        cont_dict[key]['err'].append(a0_pred_err)
        eta_label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'
        plt.errorbar([0],[a0_pred],yerr=[a0_pred_err],
                     capsize=4,color=color_list[eta_idx+3],
                     label=eta_label,#+' ($p$:{:.2f})'.format(pvalue),
                     fmt='o')

        x_grain = np.linspace(0,1.1*max(x),50) 
        y_grain = continuum_ansatz(res.x,x_grain)
        plt.plot(x_grain,y_grain,color=color_list[eta_idx+3],
                 linestyle='dashed')
        y_grain_btsp = np.array([continuum_ansatz(res_btsp[k,:],
                                 x_grain) for k in range(N_boot)])
        y_grain_err = np.array([st_dev(y_grain_btsp[:,i],y_grain[i])
                                for i in range(len(y_grain))])
        plt.fill_between(x_grain,y_grain+y_grain_err,
                         y_grain-y_grain_err,alpha=0.3,
                         color=color_list[eta_idx+3])
        text_mc = r'$m_C^{ren}(a^2)=m_C^{ren}(0)(1+\alpha a^2+\beta a^4)$'
        text_mq = r'$m_q^{*ren}(a^2)=m_q^{*ren}(0)(1+\alpha a^2+\beta a^4)$'
        plt.text(0.02,0.02,text_mc if key=='m_C' else text_mq,
                    transform=plt.gca().transAxes)
        
    plt.legend()
    plt.xlabel(r'$a^2$ (GeV${}^{2})$')
    ylabel_mc = r'$m_C^{ren}(\mu='+str(mu_chosen)+'$ GeV$)=Z_mm_C$'
    ylabel_mq = r'$\overline{m}(\mu='+str(mu_chosen)+'$ GeV$)=m_q^{*ren}=Z_mm_q^*$'
    plt.ylabel(ylabel_mc if key=='m_C' else ylabel_mq)
    title_mc = 'Renormalised charm quark mass'
    title_mq = 'Renormalised quark mass'
    plt.title(title_mc if key=='m_C' else title_mq)

#===m_c_ren vs mbar=====================
plt.figure()
for eta_idx in range(len(eta_stars)):
    x = cont_dict['m_q']['val'][eta_idx]
    xerr = cont_dict['m_q']['err'][eta_idx]
    y = cont_dict['m_C']['val'][eta_idx]
    yerr = cont_dict['m_C']['err'][eta_idx]
    eta = eta_stars[eta_idx]
    eta_label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'

    plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', markerfacecolor='none',
                 color=color_list[eta_idx+3], label=eta_label)

plt.legend()
plt.xlabel(r'$\overline{m}$ (GeV)')
plt.ylabel(r'$m_C^{ren}$ (GeV)')
plt.title(r'$m_C^{ren}(\overline{m},\mu='+str(mu_chosen)+'$ GeV$)$')

filename = f'plots/m_c_ren{argument}.pdf'
pq = PdfPages(filename) 
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pq, format='pdf')
pq.close()
plt.close('all')

os.system("open "+filename)

#===make table==========================

