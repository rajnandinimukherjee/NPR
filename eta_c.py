from basics import *

#=== measured eta_c masses in lattice units======
eta_c_data = {#'C0':{'central':{0.30:1.249409,
              #                 0.35:1.375320,
              #                 0.40:1.493579},
              #      'errors':{0.30:0.000056,
              #                0.35:0.000051,
              #                0.40:0.000048}},
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
                              0.40:0.00015}},
             'F1S':{'central':{0.18:0.82322,
                               0.23:0.965045,
                               0.28:1.098129,
                               0.33:1.223360,
                               0.40:1.385711},
                    'errors':{0.18:0.00010,
                              0.23:0.000093,
                              0.28:0.000086,
                              0.33:0.000080,
                              0.40:0.000074}}
                    }

eta_PDG = 2983.9/1000
eta_PDG_err = 0.5/1000
eta_stars = [2.4,2.6,eta_PDG]

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
    return [pred_x.item(), pred_x_err]

