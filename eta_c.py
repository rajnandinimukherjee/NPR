import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

eta_c_data = {'C0':{'central':{0.30:2.1609,
                               0.35:2.3786,
                               0.40:2.5831},
                    'errors':{0.30:0.0047,
                              0.35:0.0051,
                              0.40:0.0056}},
              'C1':{'central':{0.30:2.2246,
                               0.35:2.4492,
                               0.40:2.6604},
                    'errors':{0.30:0.0062,
                              0.35:0.0068,
                              0.40:0.0074}},
              'M1':{'central':{0.22:2.3112,
                               0.28:2.6985,
                               0.34:3.059,
                               0.40:3.393},
                    'errors':{0.22:0.0083,
                              0.28:0.0097,
                              0.34:0.011,
                              0.40:0.012}}}

eta_stars = [2.4,2.6]
ens_list = ['C1','M1']

fig, ax = plt.subplots(1,len(ens_list),sharey=True)

def interpolate(x,y,yerr,find_y,**kwargs):
    f_central = interp1d(y,x,fill_value='extrapolate')
    pred_x = f_central(find_y)

    btsp = np.array([np.random.normal(y[i],yerr[i],100)
                    for i in range(len(y))])
    pred_x_k = np.zeros(100)
    for k in range(100):
        y_k = btsp[:,k]
        f_k = interp1d(y_k,x,fill_value='extrapolate')
        pred_x_k[k] = f_k(find_y) 
    pred_x_err = ((pred_x_k[:]-pred_x).dot(pred_x_k[:]-pred_x)/100)**0.5
    return pred_x, pred_x_err

#C1
x = list(eta_c_data['C1']['central'].keys())
y = [eta_c_data['C1']['central'][x_q] for x_q in x]
yerr = [eta_c_data['C1']['errors'][x_q] for x_q in x]

ax[0].errorbar(x, y, yerr=yerr, fmt='o', capsize=4, mfc='None')   
xmin, xmax = ax[0].get_xlim()
ymin_1, ymax_1 = ax[0].get_ylim()
for eta in eta_stars:
    m_q_star, m_q_star_err = interpolate(x,y,yerr,eta)

    ax[0].axvspan(m_q_star-m_q_star_err, m_q_star+m_q_star_err,
                  color='k', alpha=0.1)

    ax[0].hlines(eta,xmin,m_q_star,label=str(eta),color='k')
    ax[0].vlines(m_q_star,ymin_1,eta,linestyle='dashed',color='k')
ax[0].legend()
ax[0].set_xlim([xmin,xmax])
ax[0].set_xlabel(r'$a_{C1}m_q$')
ax[0].set_ylabel(r'$M_{\eta_C}$ (GeV)')
ax[0].set_title('C1')
                
#M1
x = list(eta_c_data['M1']['central'].keys())
y = [eta_c_data['M1']['central'][x_q] for x_q in x]
yerr = [eta_c_data['M1']['errors'][x_q] for x_q in x]
ax[1].errorbar(x, y, yerr=yerr, fmt='o', capsize=4, mfc='None')   
xmin, xmax = ax[1].get_xlim()
ymin_2, ymax_2 = ax[1].get_ylim()
for eta in eta_stars:
    m_q_star, m_q_star_err = interpolate(x,y,yerr,eta)

    ax[1].axvspan(m_q_star-m_q_star_err, m_q_star+m_q_star_err,
                  color='k', alpha=0.1)

    ax[1].hlines(eta,xmin,m_q_star,label=str(eta),color='k')
    ax[1].vlines(m_q_star,ymin_1,eta,linestyle='dashed',color='k')
ax[1].set_xlim([xmin,xmax])
ax[1].set_xlabel(r'$a_{M1}m_q$')
ax[1].set_title('M1')

ax[0].set_ylim([ymin_1,ymax_2])

from matplotlib.backends.backend_pdf import PdfPages
import os

pp = PdfPages('plots/eta_c.pdf')
fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pp, format='pdf')
pp.close()
plt.close('all')
os.system("open plots/eta_c.pdf")
