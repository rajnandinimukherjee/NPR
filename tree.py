from NPR_structures import *
currents = ['S','P','V','A','T']
from numpy.linalg import norm
from tqdm import tqdm

direc = '/Users/rajnandinimukherjee/PhD/NPR/tree'

prop_in_file = h5py.File(f'{direc}/ExternalLeg_3300_wilson.1.h5','r')
prop_in_data = prop_in_file['ExternalLeg']['corr'][0,0,:]
prop_in = (prop_in_data['re']+prop_in_data['im']*1j).swapaxes(1,2).reshape((12,12))
prop_in_inv = np.linalg.inv(prop_in)

prop_out_file = h5py.File(f'{direc}/ExternalLeg_0330_wilson.1.h5','r')
prop_out_data = prop_out_file['ExternalLeg']['corr'][0,0,:]
prop_out = (prop_out_data['re']+prop_out_data['im']*1j).swapaxes(1,2).reshape((12,12))
prop_out_inv_out = np.linalg.inv(Gamma['5']@(prop_out.conj().T)@Gamma['5']) 

fqstr = 'FourQuarkFullyConnected'
tree_fq_file = h5py.File(f'{direc}/Fourquark_SMOM_wilson.1.h5','r')[fqstr]
tree_fq_data = [tree_fq_file[f'{fqstr}_{i}']['corr'][0,0,:]
                for i in range(32)]
tree_fq = [(tree_fq_data[i]['re']+tree_fq_data[i]['im']*1j).swapaxes(1,
           2).swapaxes(5,6).reshape((12,12,12,12)) for i in range(32)]

p_in = np.array([int(x) for x in tree_fq_file[f'{fqstr}_0']['info'].attrs['pIn'][0].decode().rsplit(' ')])
p_out = np.array([int(x) for x in tree_fq_file[f'{fqstr}_0']['info'].attrs['pOut'][0].decode().rsplit(' ')])
q = p_out-p_in

p_in, p_out, q = (2*np.pi*a/L)*p_in, (2*np.pi*a/L)*p_out, (2*np.pi*a/L)*q

org_gammas = [[tree_fq_file[f'{fqstr}_{i}']['info'].attrs['gammaA'][0].decode(),
               tree_fq_file[f'{fqstr}_{i}']['info'].attrs['gammaB'][0].decode()]
               for i in range(32)]
gammas = [[[x for x in list(org_gammas[i][0]) if x in list(gamma.keys())],
           [y for y in list(org_gammas[i][1]) if y in list(gamma.keys())]]
           for i in range(32)]


doublets = {'SS':tree_fq[gammas.index([['I'],['I']])],
            'PP':tree_fq[gammas.index([['5'],['5']])],
            'VV':np.sum(np.array([tree_fq[gammas.index([[i],[i]])]
                  for i in dirs]), axis=0),
            'AA':np.sum(np.array([tree_fq[gammas.index([[i,'5'],[i,'5']])]
                  for i in dirs]), axis=0),
            'TT':np.sum(np.array(sum([[tree_fq[gammas.index([[dirs[i],
                  dirs[j]],[dirs[i],dirs[j]]])] for j in range(i+1,4)] 
                  for i in range(0,4-1)], [])), axis=0)}
operator = {'VV+AA':doublets['VV']+doublets['AA'],
            'VV-AA':doublets['VV']-doublets['AA'],
            'SS-PP':doublets['SS']-doublets['PP'],
            'SS+PP':doublets['SS']+doublets['PP'],
            'TT':doublets['TT']}

amputated = {}
in_, out_ = prop_in_inv, prop_out_inv_out
for k in operator.keys():
    op = 2*(operator[k]-operator[k].swapaxes(1,3))
    op_avg = np.array(op,dtype='complex128')
    amputated[k] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                  out_,in_,out_,in_,op_avg) 

projector = fq_gamma_projector  
projected = np.array([[np.einsum('abcd,badc',projector[k1],
                  amputated[k2]) for k2 in operators]
                  for k1 in operators])
matrix = (fq_gamma_F@np.linalg.inv(projected)).real
