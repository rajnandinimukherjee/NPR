from NPR_structures import *
from bag_param_renorm import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mc

extrap_ensembles = bag_ensembles

Z_dict = {ens:Z_analysis(ens) for ens in extrap_ensembles}
bag_dict = {ens:bag_analysis(ens) for ens in extrap_ensembles}


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

def ansatz(a_sq, mpi_f_m, **kwargs):
    return signal

