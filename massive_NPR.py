from NPR_classes import *
from eta_c import *

ens_list = list(eta_c_data.keys())
mu_chosen = 2.0

#====plotting eta_c data from f_D paper=========================================
fig, ax = plt.subplots(1,len(ens_list),sharey=True)
for ens in ens_list:
    ens_idx = ens_list.index(ens)
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']

    ax[ens_idx].errorbar(x, y*ainv, yerr=yerr*ainv,
                fmt='o', capsize=4, mfc='None')   
    xmin, xmax = ax[ens_idx].get_xlim()
    ymin0, ymax0 = ax[0].get_ylim()
    ymin, ymax = ax[ens_idx].get_ylim()
    if ymin0>ymin:
        ymin0 = ymin
    if ymax0<ymax:
        ymax0 = ymax

    for eta in eta_stars:
        m_q_star, m_q_star_err = interpolate_eta_c(ens,eta)

        ax[ens_idx].axvspan(m_q_star-m_q_star_err,
                            m_q_star+m_q_star_err,
                            color='k', alpha=0.1)
        label = 'PDG' if eta==eta_PDG else str(eta)
        ax[ens_idx].hlines(eta,xmin,m_q_star,label=label,color='k')
        ax[ens_idx].vlines(m_q_star,ymin,eta,linestyle='dashed',color='k')
    ax[ens_idx].legend()
    ax[ens_idx].set_xlabel(r'$a_{'+ens+r'}m_q$')
    ax[ens_idx].set_title(ens)

ax[0].set_ylabel(r'$M_{\eta_C}$ (GeV)')
for ax_i in ax:
    ax_i.set_ylim([ymin0,ymax0])

plt.text(1.03,0.3,'Using data from $f_D$ paper',
         transform=plt.gca().transAxes,color='r',rotation=90)
                
pp = PdfPages('plots/eta_c.pdf')
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')
os.system("open plots/eta_c.pdf")

#====plotting Z_m(mu_chosen) extrapolation at m_q_stars=========================
def Z_m_ansatz(params, am, **kwargs):
    return params[0] + (am**2)*params[1]+ params[2]*np.log(am)*am

fig, ax = plt.subplots(1,len(ens_list),figsize=(15,6))
plt.suptitle('$Z_m(\mu=${:.3f} GeV)'.format(
             mu_chosen)+'\n$ Z_m(am_q)=k_1+k_2\,am_q\,\log(am_q)+k_3(am_q)^2$')
filename = 'plots/combined_massive_Z.pdf'
pdf = PdfPages(filename)

ens_dict = {}
for ens in ens_list:
    ens_idx = ens_list.index(ens)
    ens_dict[ens] = {'bl_obj':bilinear_analysis(ens, 
                     loadpath=f'pickles/{ens}_bl.p')}
    ainv = ens_dict[ens]['bl_obj'].ainv
    eta_star_dict = {eta:interpolate_eta_c(ens,eta)
                     for eta in eta_stars}
    x, y, e = ens_dict[ens]['bl_obj'].massive_Z_plots(
                                      mu=mu_chosen, passinfo=True)

    ax[ens_idx].set_title(ens)
    ax[ens_idx].errorbar(x**1,y,yerr=e,fmt='o',capsize=4,
                         label='simulated $am_q$',
                         color=color_list[ens_idx])
    ax[ens_idx].set_xlabel(r'$am_q$')#'$(am_q)^2$')

    def diff(params):
        return y - Z_m_ansatz(params, x)
    
    COV = np.diag(e**2)
    L_inv = np.linalg.cholesky(COV)
    L = np.linalg.inv(L_inv)

    def LD(params, **akwargs):
        return L.dot(diff(params))
    
    guess = [1,-1,1]
    res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
    chi_sq = LD(res.x).dot(LD(res.x))
    dof = len(x)-len(guess)
    pvalue = gammaincc(dof/2,chi_sq/2)

    xmin, xmax = ax[ens_idx].get_xlim()
    x_grain = np.linspace(xmin,xmax,50)
    ax[ens_idx].plot(x_grain,Z_m_ansatz(res.x,x_grain**1),#**0.5),
                label='fit $p$-value:{:.2f}'.format(pvalue),
                color='tab:gray')

    m_q_stars = np.array([v[0] for k,v in eta_star_dict.items()])
    Z_ms = Z_m_ansatz(res.x,m_q_stars)
    m_q_btsp = np.array([np.random.normal(v[0],v[1],N_boot)
                       for k,v in eta_star_dict.items()])
    Z_m_btsp = np.array([Z_m_ansatz(res.x,m_q_btsp[:,k])
                           for k in range(N_boot)])
    Z_m_errs = [st_dev(Z_m_btsp[:,i],mean=Z_ms[i])
            for i in range(len(Z_ms))]

    ymin, ymax = ax[ens_idx].get_ylim()
    for eta in eta_stars:
        eta_idx = eta_stars.index(eta)
        val, err = eta_star_dict[eta]
        eta_label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'
        ax[ens_idx].vlines(val**1,ymin,ymax,linestyle='dashed',
                   color=color_list[eta_idx+3],
                   label='$am_q^*(M_{\eta_c}^*=$'+eta_label+'$)$')
        ax[ens_idx].axvspan((val-err)**1,(val+err)**1,
                            color=color_list[eta_idx+3],alpha=0.1)
        ax[ens_idx].errorbar(val**1,Z_ms[eta_idx],yerr=Z_m_errs[eta_idx],
                     color=color_list[eta_idx+3],fmt='o',capsize=4)
    ax[ens_idx].legend()

    ens_dict[ens]['Z_m'] = Z_ms
    ens_dict[ens]['Z_m_err'] = Z_m_errs
    
    #===m_C renormalisation=====================
    m_C, m_C_err = interpolate_eta_c(ens,eta_PDG)
    ens_dict[ens]['m_C_ren'] = Z_ms*m_C*ainv
    m_C_btsp = np.random.normal(m_C, m_C_err, N_boot)
    m_C_ren_btsp = np.array([Z_m_btsp[k,:]*m_C_btsp[k]*ainv
                             for k in range(N_boot)])
    ens_dict[ens]['m_C_ren_err'] = [st_dev(m_C_ren_btsp[:,i],Z_ms[i]*m_C*ainv)
                    for i in range(len(Z_ms))]

    #===m_q renormalisation=====================
    ens_dict[ens]['m_q_ren'] = Z_ms*m_q_stars*ainv 
    m_q_ren_btsp = np.array([Z_m_btsp[k,:]*m_q_btsp[:,k]*ainv
                             for k in range(N_boot)])
    ens_dict[ens]['m_q_ren_err'] = np.array([st_dev(m_q_ren_btsp[:,i],
                               mean=(Z_ms*m_q_stars*ainv)[i])
                               for i in range(len(m_q_stars))])

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

for key in ['m_C','m_q']:
    plt.figure()
    for ens in ens_list:
        ens_idx = ens_list.index(ens)
        y = ens_dict[ens][f'{key}_ren']
        e = ens_dict[ens][f'{key}_ren_err']
        x = [ens_dict[ens]['bl_obj'].asq]*len(y)
        plt.errorbar(x,y,yerr=e,fmt='o',capsize=4,
                     color=color_list[ens_idx],label=ens)

    x = np.array([v['bl_obj'].asq for k,v in ens_dict.items()])
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

        #===bootstrap=====
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

        a0_pred_err = st_dev(a0_pred_btsp,mean=a0_pred)
        eta_label = 'PDG' if eta==eta_PDG else str(eta)+' GeV'
        plt.errorbar([0],[a0_pred],yerr=[a0_pred_err],
                     capsize=4,color=color_list[eta_idx+3],
                     label=eta_label,#+' ($p$:{:.2f})'.format(pvalue),
                     marker=marker_list[eta_idx])

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
    ylabel_mc = r'$m_C^{ren}=Z_mm_C$'
    ylabel_mq = r'$\overline{m}=m_q^{*ren}=Z_mm_q^*$'
    plt.ylabel(ylabel_mc if key=='m_C' else ylabel_mq)
    title_mc = 'Renormalised charm quark mass'
    title_mq = 'Renormalised quark mass'
    plt.title(title_mc if key=='m_C' else title_mq)

pq = PdfPages('plots/m_c_ren.pdf') 
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pq, format='pdf')
pq.close()
plt.close('all')

os.system("open plots/m_c_ren.pdf")
    