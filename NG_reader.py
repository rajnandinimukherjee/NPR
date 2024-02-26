from fourquark import *
import glob


data_ensembles = [
        'C0','C1','C1M','C2',
        'M0','M1', 'M1M', 'M2', 'M3',
        'F1M']
am_dict = {ens:"{:.4f}".format(params[ens]['masses'][0])
        for ens in data_ensembles}

NG_path = '/home/rm/external/NPR/NG_data'

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
                    val=central,
                    err='fill',
                    btsp=bootstrap)

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

bl_qslash_filename = 'bilinear_Z_qslash.h5'
bl_file = h5py.File(bl_qslash_filename, 'a')
for ens in data_ensembles:
    grp_name = f'{str((0,0))}/{ens}/{str((am_dict[ens], am_dict[ens]))}'
    if grp_name in bl_file.keys():
        del bl_file[grp_name]
    grp = bl_file.create_group(grp_name)
    ap = grp.create_dataset('ap', data=Z_bl[ens]['ap'])
    for c in ['S','P','V','A']:
        c_grp = grp.create_group(c)
        Z = join_stats(Z_bl[ens]['Z'][c])
        central = c_grp.create_dataset('central', data=Z.val)
        errors = c_grp.create_dataset('errors', data=Z.err)
        btsp = c_grp.create_dataset('bootstrap', data=Z.btsp)

# STEP 3: Remove division by Z_A on Z_ij_Z_A_2 and save to file

fq_qslash_filename = 'fourquark_Z_qslash.h5'
fq_file = h5py.File(fq_qslash_filename, 'a')
for ens in data_ensembles:
    grp_name = f'{str((0,0))}/{ens}/{str((am_dict[ens], am_dict[ens]))}'
    if grp_name in fq_file.keys():
        del fq_file[grp_name]
    grp = fq_file.create_group(grp_name)
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
