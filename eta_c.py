from basics import *

#=== measured eta_c masses in lattice units======
eta_c_data = {'C0':{'central':{0.30:1.249409,
                               0.35:1.375320,
                               0.40:1.493579},
                    'errors':{0.30:0.000056,
                              0.35:0.000051,
                              0.40:0.000048}},
              'C1':{'central':{0.30:1.24641,
                               0.35:1.37227,
                               0.40:1.49059},
                    'errors':{0.30:0.00020,
                              0.35:0.00019,
                              0.40:0.00017}},
              'M1':{'central':{0.22:0.96975,
                               0.28:1.13226,
                               0.34:1.28347,
                               0.40:1.42374},
                    'errors':{0.22:0.00018,
                              0.28:0.00015,
                              0.34:0.00016,
                              0.40:0.00015}}}

eta_stars = [2.4,2.6]
ens_list = ['C0','C1','M1']

def interpolate_eta_c(ens,find_y,**kwargs):
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']
    f_central = interp1d(y*ainv,x,fill_value='extrapolate')
    pred_x = f_central(find_y)

    btsp = np.array([np.random.normal(y[i],yerr[i],100)
                    for i in range(len(y))])
    pred_x_k = np.zeros(100)
    for k in range(100):
        y_k = btsp[:,k]
        f_k = interp1d(y_k*ainv,x,fill_value='extrapolate')
        pred_x_k[k] = f_k(find_y) 
    pred_x_err = ((pred_x_k[:]-pred_x).dot(pred_x_k[:]-pred_x)/100)**0.5
    return pred_x, pred_x_err

fig, ax = plt.subplots(1,len(ens_list),sharey=True)
for ens in ens_list:
    ens_idx = ens_list.index(ens)
    x = np.array(list(eta_c_data[ens]['central'].keys()))
    y = np.array([eta_c_data[ens]['central'][x_q] for x_q in x])
    yerr = np.array([eta_c_data[ens]['errors'][x_q] for x_q in x])
    ainv = params[ens]['ainv']

    ax[ens_idx].errorbar(x, y*ainv, yerr=yerr*ainv, fmt='o', capsize=4, mfc='None')   
    xmin, xmax = ax[ens_idx].get_xlim()
    ymin0, ymax0 = ax[0].get_ylim()
    ymin, ymax = ax[ens_idx].get_ylim()
    if ymin0>ymin:
        ymin0 = ymin
    if ymax0<ymax:
        ymax0 = ymax

    for eta in eta_stars:
        m_q_star, m_q_star_err = interpolate_eta_c(ens,eta)

        ax[ens_idx].axvspan(m_q_star-m_q_star_err, m_q_star+m_q_star_err,
                      color='k', alpha=0.1)

        ax[ens_idx].hlines(eta,xmin,m_q_star,label=str(eta),color='k')
        ax[ens_idx].vlines(m_q_star,ymin,eta,linestyle='dashed',color='k')
    ax[ens_idx].legend()
    ax[ens_idx].set_xlim([xmin,xmax])
    ax[ens_idx].set_xlabel(r'$a_{'+ens+r'}m_q$')
    ax[ens_idx].set_ylabel(r'$M_{\eta_C}$ (GeV)')
    ax[ens_idx].set_title(ens)

for ax_i in ax:
    ax_i.set_ylim([ymin0,ymax0])
                
pp = PdfPages('plots/eta_c.pdf')
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')
os.system("open plots/eta_c.pdf")
