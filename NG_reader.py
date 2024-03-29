from fourquark import *
import glob

fq_qslash_F = np.array([
    [64*N_c*(N_c+1), 0,0,0,0],
    [0,64*N_c**2,64*N_c,0,0],
    [0,-32*N_c,-32*N_c**2,0,0],
    [0,0,0,8*N_c**2,8*N_c],
    [0,0,0,8*N_c*(N_c+2),8*N_c*(2*N_c+1)]], dtype=complex)


data_ensembles = [
        #'C0', 'C1M',
        'C1', 'C2',
        #'M0', 'M1M',
        'M1', 'M2', 'M3']
        #'F1M']
sea_mass_dict = {
        'C0':0,
        'C1':1,
        'C1M':3,
        'C2':1,
        'M0':0,
        'M1':1,
        'M1M':3,
        'M2':1,
        'M3':1,
        'F1M':1}

am_dict = {ens:"{:.4f}".format(params[ens]['masses'][sea_mass_dict[ens]])
        for ens in data_ensembles}

NG_path = path+'NG_data'

# STEP 1: read in Z_4q

fq_scheme = 'qq'

Z_4q_files = {ens: glob.glob(f'{NG_path}/{fq_scheme}/Z_4q_{fq_scheme}_{ens}_am_{am_dict[ens]}*{N_boot}.h5')
        for ens in data_ensembles}
Z_ij_Z_A_2 = {ens:{} for ens in data_ensembles}

for ens in data_ensembles:
    ens_dict = {}
    for f in Z_4q_files[ens]:
        filename = f.rsplit('/')[-1]
        info_list = filename.rsplit('_')
        tw1 = np.array([float(t) for t in info_list[7:11]])
        mom1 = np.array([float(m) for m in info_list[14:18]])
        tw2 = np.array([float(t) for t in info_list[19:23]])
        mom2 = np.array([float(m) for m in info_list[26:30]])

        coeff = 2*np.pi/params[ens]['XX']
        p1 = coeff*(mom1+tw1)
        p2 = coeff*(mom2+tw2)
        q = np.linalg.norm(p1-p2)

        data = h5py.File(f,'r')[f'Z_4q_{fq_scheme}']
        central = np.array(data['central'][:]).reshape(5,5)
        bootstrap = np.moveaxis(np.array(data['bootstraps'][:]).reshape(5,5,N_boot),-1,0)

        if q not in ens_dict:
            ens_dict[q] = stat(
                    val=central.T,
                    err='fill',
                    btsp=np.array([bootstrap[k,:,:].T for k in range(N_boot)]))

    momenta = sorted(ens_dict.keys())
    Z_ij_Z_A_2[ens]['ap'] = momenta
    Z_ij_Z_A_2[ens]['Z'] = [ens_dict[m] for m in momenta]

# STEP 2: read in As

A_scheme = 'q'

Z_AV_files = {ens: glob.glob(f'{NG_path}/{A_scheme}/Lambda_bil_VA_{A_scheme}_{ens}_am_{am_dict[ens]}*{N_boot}.h5')
        for ens in data_ensembles} 
Z_SP_files = {ens: glob.glob(f'{NG_path}/g/Lambda_bil_SVTAP_g_{ens}_am_{am_dict[ens]}*{N_boot}.h5')
        for ens in data_ensembles} 
Z_bl = {ens:{} for ens in data_ensembles}

for ens in data_ensembles:
    ens_dict = {}
    for f in Z_AV_files[ens]:
        filename = f.rsplit('/')[-1]
        info_list = filename.rsplit('_')
        tw1 = np.array([float(t) for t in info_list[8:12]])
        mom1 = np.array([float(m) for m in info_list[15:19]])
        tw2 = np.array([float(t) for t in info_list[20:24]])
        mom2 = np.array([float(m) for m in info_list[27:31]])

        coeff = 2*np.pi/params[ens]['XX']
        p1 = coeff*(mom1+tw1)
        p2 = coeff*(mom2+tw2)
        q = np.linalg.norm(p1-p2)

        data = h5py.File(f,'r')[f'Lambda_bil_VA_{A_scheme}']
        VA = stat(
                val=np.array(data['central'][:]),
                err='fill',
                btsp=np.array(data['bootstraps'][:]).reshape(2,1000).T)
        V, A = VA[0]**(-1), VA[1]**(-1)
        if q not in ens_dict:
            ens_dict[q] = {'V':V, 'A':A}

    for f in Z_SP_files[ens]:
        filename = f.rsplit('/')[-1]
        info_list = filename.rsplit('_')
        tw1 = np.array([float(t) for t in info_list[8:12]])
        mom1 = np.array([float(m) for m in info_list[15:19]])
        tw2 = np.array([float(t) for t in info_list[20:24]])
        mom2 = np.array([float(m) for m in info_list[27:31]])

        coeff = 2*np.pi/params[ens]['XX']
        p1 = coeff*(mom1+tw1)
        p2 = coeff*(mom2+tw2)
        q = np.linalg.norm(p1-p2)

        data = h5py.File(f,'r')[f'Lambda_bil_SVTAP_g']
        SVTAP = stat(
                val=np.array(data['central'][:]),
                err='fill',
                btsp=np.array(data['bootstraps'][:]).reshape(5,1000).T)
        S, P = SVTAP[0]**(-1), SVTAP[4]**(-1)
        if 'S' not in ens_dict[q]:
            ens_dict[q].update({'S':S, 'P':P})

    momenta = sorted(ens_dict.keys())
    Z_bl[ens]['ap'] = momenta
    Z_bl[ens]['Z'] = {c:[ens_dict[m][c] for m in momenta]
            for c in ['S','P','V','A']}

action = (0,0)
a1, a2 = action
for ens in data_ensembles:
    filename = f'NPR/action{a1}_action{a2}/'
    filename += '__'.join(['NPR', ens, params[ens]['baseactions'][a1],
                           params[ens]['baseactions'][a2], 'qslash'])
    filename += '.h5'
    file = h5py.File(filename, 'a')
    grp_name = f'{str((am_dict[ens], am_dict[ens]))}/bilinear'
    if grp_name in file.keys():
        del file[grp_name]
    grp = file.create_group(grp_name)
    ap = grp.create_dataset('ap', data=Z_bl[ens]['ap'])
    for c in ['S','P','V','A']:
        c_grp = grp.create_group(c)
        Z = join_stats(Z_bl[ens]['Z'][c])
        central = c_grp.create_dataset('central', data=Z.val)
        errors = c_grp.create_dataset('errors', data=Z.err)
        btsp = c_grp.create_dataset('bootstrap', data=Z.btsp)

    grp_name = f'{str((am_dict[ens], am_dict[ens]))}/fourquark'
    if grp_name in file.keys():
        del file[grp_name]
    grp = file.create_group(grp_name)
    ap = grp.create_dataset('ap', data=Z_ij_Z_A_2[ens]['ap'])

    Z_ij_A = join_stats(Z_ij_Z_A_2[ens]['Z'])
    Z_A_2 = join_stats(Z_bl[ens]['Z']['A'])**2
    Z = stat(
            val=np.array([Z_ij_A.val[m,]*Z_A_2.val[m]
                for m in range(len(Z_A_2.val))]),
            err='fill',
            btsp=np.array([[Z_ij_A.btsp[k,m]*Z_A_2.btsp[k,m]
                for m in range(len(Z_A_2.val))] for k in range(N_boot)]))
    
    central = grp.create_dataset('central', data=Z.val)
    errors = grp.create_dataset('errors', data=Z.err)
    btsp = grp.create_dataset('bootstrap', data=Z.btsp)

