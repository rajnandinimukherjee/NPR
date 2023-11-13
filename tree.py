from tqdm import tqdm
from numpy.linalg import norm
from NPR_classes import *


def pole_mass(m_q, tot_mom, M5, Ls):
    p_hat = 2*np.sin(tot_mom/2)
    p_hat_sq = np.linalg.norm(p_hat)**2
    p_dash = np.sin(tot_mom)
    p_dash_sq = np.linalg.norm(p_dash)**2

    W = 1 - M5 + p_hat_sq/2
    alpha = np.arccosh((1+W**2+p_dash_sq)/(2*np.abs(W)))
    Z = np.abs(W)*np.exp(alpha)
    delta = (W/Z)**Ls
    R = 1-(W**2)/Z + (delta**2)*(Z-(W**2)/Z)/(1-delta**2)

    E = np.arcsinh(m_q*R + delta*(Z-(W**2)/Z)/(1-delta**2))
    return E


currents = ['S', 'P', 'V', 'A', 'T']

direc = '/Users/rajnandinimukherjee/PhD/NPR/analysis/DWF_tree/npr_m0point005'
m_in = 0.005
L = 8
M5 = 1.8
Ls = 16
N_bl = 16

prop_in_file = h5py.File(f'{direc}/ExternalLeg_3300_wilson.1.h5', 'r')
prop_in_data = prop_in_file['ExternalLeg']['corr'][0, 0, :]
prop_in = (prop_in_data['re']+prop_in_data['im']
           * 1j).swapaxes(1, 2).reshape((12, 12))
prop_in_inv = np.linalg.inv(prop_in)

prop_out_file = h5py.File(f'{direc}/ExternalLeg_0330_wilson.1.h5', 'r')
prop_out_data = prop_out_file['ExternalLeg']['corr'][0, 0, :]
prop_out = (prop_out_data['re']+prop_out_data['im']
            * 1j).swapaxes(1, 2).reshape((12, 12))
prop_out_inv_out = np.linalg.inv(Gamma['5']@(prop_out.conj().T)@Gamma['5'])
in_, out_ = prop_in_inv, prop_out_inv_out


blstr = 'Bilinear'
tree_bl_file = h5py.File(f'{direc}/Bilinear_SMOM_wilson.1.h5', 'r')[blstr]
tree_bl_data = [tree_bl_file[f'{blstr}_{i}']['corr'][0, 0, :]
                for i in range(N_bl)]
tree_bl = [(tree_bl_data[i]['re']+tree_bl_data[i]['im']*1j).swapaxes(1,
           2).reshape((12, 12)) for i in range(N_bl)]

p_in = np.array([int(x) for x in tree_bl_file[f'{blstr}_0'][
    'info'].attrs['pIn'][0].decode().rsplit(' ')])
p_out = np.array([int(x) for x in tree_bl_file[f'{blstr}_0'][
    'info'].attrs['pOut'][0].decode().rsplit(' ')])
q = p_out-p_in

p_in, p_out, q = (2*np.pi/L)*p_in, (2*np.pi/L)*p_out, (2*np.pi/L)*q
cos_term = np.sum([(1-np.cos(p_in[i])) for i in range(len(dirs))])

p_in, p_out, q = np.sin(p_in), np.sin(p_out), np.sin(q)
pslash = np.sum([p_in[i]*Gamma[dirs[i]]
                 for i in range(len(dirs))], axis=0)
p_sq = np.linalg.norm(p_in)**2
Z_q = np.trace(-1j*in_@pslash
               ).real/(12*p_sq)

org_gammas = [tree_bl_file[f'{blstr}_{i}']['info'].attrs['gamma'][0].decode()
              for i in range(N_bl)]
gammas = [[x for x in list(org_gammas[i]) if x in list(gamma.keys())]
          for i in range(N_bl)]

amputated = np.array([out_@tree_bl[b]@in_ for b in range(N_bl)])
bl_operators = {'S': [amputated[gammas.index(['I'])]],
                'P': [amputated[gammas.index(['5'])]],
                'V': [amputated[gammas.index([i])]
                      for i in dirs],
                'A': [amputated[gammas.index([i, '5'])]
                      for i in dirs],
                'T': sum([[amputated[gammas.index([dirs[i],
                                                  dirs[j]])] for j in range(i+1, 4)]
                         for i in range(0, 4-1)], [])}

projected = {c: np.trace(np.sum([bl_gamma_proj[c][i]@bl_operators[c][i]
                                for i in range(len(bl_operators[c]))], axis=0))
             for c in bilinear.currents}

gamma_Z = {c: (bl_gamma_F[c]/projected[c]).real for c in bilinear.currents}

qslash = np.sum([q[i]*Gamma[dirs[i]]
                 for i in range(len(dirs))], axis=0)
q_sq = np.linalg.norm(q)**2
Z_P = Z_q*gamma_Z['P']
Z_T = Z_q*gamma_Z['T']
Z_V = Z_q/(np.trace(np.sum([q[i]*bl_operators['V'][i]
                           for i in range(len(dirs))],
                           axis=0)@qslash).real/(12*q_sq))

m_q = pole_mass(m_in, q, M5, Ls)
A1 = np.trace(np.sum([q[i]*bl_operators['A'][i]
                     for i in range(len(dirs))], axis=0)@Gamma['5'])
A2 = np.trace(np.sum([q[i]*bl_operators['A'][i]
                     for i in range(len(dirs))], axis=0)@Gamma['5']@qslash)
P = np.trace(bl_operators['P'][0]@Gamma['5']@qslash)
S = np.trace(in_)

Z_A = (144*q_sq*(Z_q**2)-2*Z_P*S*P)/(12*Z_q*A2 + 1j*Z_P*A1*P)
Z_m = (S+(Z_A*A1*1j)/2)/(12*m_q*Z_q)

s_term = np.trace(bl_operators['S'][0])
mass_term = 4*m_q*Z_m*Z_P*P
Z_S = (12*q_sq*Z_q-mass_term)/(q_sq*s_term)
qslash_Z = {'S': (Z_S).real,
            'P': (Z_P).real,
            'V': (Z_V).real,
            'A': (Z_A).real,
            'T': (Z_T).real,
            'm': (Z_m).real,
            'q': Z_q.real}
qslash_Z_Zq = {'S': (Z_S/Z_q).real,
               'P': (Z_P/Z_q).real,
               'V': (Z_V/Z_q).real,
               'A': (Z_A/Z_q).real,
               'T': (Z_T/Z_q).real,
               'm': (Z_m*Z_q).real}


def fq_SMOM():

    fqstr = 'FourQuarkFullyConnected'
    tree_fq_file = h5py.File(f'{direc}/Fourquark_SMOM_wilson.1.h5', 'r')[fqstr]
    tree_fq_data = [tree_fq_file[f'{fqstr}_{i}']['corr'][0, 0, :]
                    for i in range(32)]
    tree_fq = [(tree_fq_data[i]['re']+tree_fq_data[i]['im']*1j).swapaxes(1,
               2).swapaxes(5, 6).reshape((12, 12, 12, 12)) for i in range(32)]

    p_in = np.array(
        [int(x) for x in tree_fq_file[f'{fqstr}_0']['info'].attrs['pIn'][0].decode().rsplit(' ')])
    p_out = np.array(
        [int(x) for x in tree_fq_file[f'{fqstr}_0']['info'].attrs['pOut'][0].decode().rsplit(' ')])
    q = p_out-p_in

    p_in, p_out, q = (2*np.pi/L)*p_in, (2*np.pi/L)*p_out, (2*np.pi/L)*q

    org_gammas = [[tree_fq_file[f'{fqstr}_{i}']['info'].attrs['gammaA'][0].decode(),
                   tree_fq_file[f'{fqstr}_{i}']['info'].attrs['gammaB'][0].decode()]
                  for i in range(32)]
    gammas = [[[x for x in list(org_gammas[i][0]) if x in list(gamma.keys())],
               [y for y in list(org_gammas[i][1]) if y in list(gamma.keys())]]
              for i in range(32)]

    doublets = {'SS': tree_fq[gammas.index([['I'], ['I']])],
                'PP': tree_fq[gammas.index([['5'], ['5']])],
                'VV': np.sum(np.array([tree_fq[gammas.index([[i], [i]])]
                                      for i in dirs]), axis=0),
                'AA': np.sum(np.array([tree_fq[gammas.index([[i, '5'], [i, '5']])]
                                      for i in dirs]), axis=0),
                'TT': np.sum(np.array(sum([[tree_fq[gammas.index([[dirs[i],
                                                                   dirs[j]], [dirs[i], dirs[j]]])] for j in range(i+1, 4)]
                                           for i in range(0, 4-1)], [])), axis=0)}
    operator = {'VV+AA': doublets['VV']+doublets['AA'],
                'VV-AA': doublets['VV']-doublets['AA'],
                'SS-PP': doublets['SS']-doublets['PP'],
                'SS+PP': doublets['SS']+doublets['PP'],
                'TT': doublets['TT']}

    amputated = {}
    for k in operator.keys():
        op = 2*(operator[k]-operator[k].swapaxes(1, 3))
        op_avg = np.array(op, dtype='complex128')
        amputated[k] = np.einsum('ea,bf,gc,dh,abcd->efgh',
                                 out_, in_, out_, in_, op_avg)

    projector = fq_gamma_projector
    projected = np.array([[np.einsum('abcd,badc', projector[k1],
                                     amputated[k2]) for k2 in operators]
                          for k1 in operators])
    matrix = (fq_gamma_F@np.linalg.inv(projected)).real
    return matrix
