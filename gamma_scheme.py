from basics import *

#=====bilinear projectors==============================
bl_proj = {'S':[Gamma['I']],
           'P':[Gamma['5']],
           'V':[Gamma[i] for i in dirs],
           'A':[Gamma[i]@Gamma['5'] for i in dirs],
           'T':sum([[Gamma[dirs[i]]@Gamma[dirs[j]]
               for j in range(i+1,4)] for i in range(0,4-1)],[])}
bl_gamma_F = {k:np.trace(np.sum([mtx@mtx for mtx in bl_proj[k]],axis=0))
              for k in currents}











#=====fourquark projectors=======================================
fq_proj = {'SS':np.einsum('ab,cd->abcd',Gamma['I'],Gamma['I']),
        'PP':np.einsum('ab,cd->abcd',Gamma['5'],Gamma['5']),
        'VV':np.sum(np.array([np.einsum('ab,cd->abcd',Gamma[i],Gamma[i])
             for i in dirs]), axis=0),
        'AA':np.sum(np.array([np.einsum('ab,cd->abcd',
             Gamma[i]@Gamma['5'],Gamma[i]@Gamma['5'])
             for i in dirs]), axis=0),
        'TT':np.sum(np.array(sum([[np.einsum('ab,cd->abcd',
             Gamma[dirs[i]]@Gamma[dirs[j]],Gamma[dirs[i]]@Gamma[dirs[j]])
             for j in range(i+1,4)] for i in range(0,4-1)],[])), axis=0)}

fq_gamma_projector = {'VV+AA':fq_proj['VV']+fq_proj['AA'],
                  'VV-AA':fq_proj['VV']-fq_proj['AA'],
                  'SS-PP':fq_proj['SS']-fq_proj['PP'],
                  'SS+PP':fq_proj['SS']+fq_proj['PP'],
                  'TT':fq_proj['TT']}

fq_gamma_F = np.array([[np.einsum('abcd,badc',2*(fq_gamma_projector[k1]-fq_gamma_projector[k1].swapaxes(1,3)),
               fq_gamma_projector[k2]) for k2 in operators] for k1 in operators])

