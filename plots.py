from NPR_classes import *
from basics import *
from eta_c import *

ens_list = list(eta_c_data.keys())
mu_chosen = 2.0
filename = '/Users/rajnandinimukherjee/Desktop/LatPlots.pdf'
pdf = PdfPages(filename)

#===plot 1: M1 M_eta_C measurements===================
x = np.array(list(eta_c_data['M1']['central'].keys()))
y = np.array([eta_c_data['M1']['central'][x_q] for x_q in x])
yerr = np.array([eta_c_data['M1']['errors'][x_q] for x_q in x])


fig_nums = plt.get_fignums()
figs = [plt.figure(n) for n in fig_nums]
for fig in figs:
    fig.savefig(pdf, format='pdf')
pdf.close()
plt.close('all')
os.system('open '+filename)
