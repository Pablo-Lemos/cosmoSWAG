import os
import numpy as np
import einops
import gc
import torch
import torch.utils.data as data_utils

def read_data(n = 2000):
    print('Start reading data')

    gc.enable()
    os.getcwd()
    os.chdir('/projects/QUIJOTE/Density_fields_3D')

    all_files = np.sort((np.sort((os.listdir()))[0:n]).astype('int32'))
    f_params = '/home/pl7508/cosmoSWAG/cosmoswag/data/galaxy_sims/latin_hypercube_params.txt'
    Om, Ob, h, ns, s8 = np.loadtxt(f_params, unpack=True)
    params_reformatted = np.array([Om, Ob, h, ns, s8])

    all_data = np.array(range(0, len(all_files)))
    all_data = np.array([[i] for i in all_data])
    storing_data = np.zeros((n, 128, 128, 128))

    # Only look at z = 0, store all density data
    for i in range(len(storing_data)):
        if i%50 == 0 and i>0: 
            print('Read', i, 'sims.')
        os.chdir('/projects/QUIJOTE/Density_fields_3D')
        files_placeholder = str(all_files[i])
        data_files = np.sort(os.listdir(files_placeholder))
        curr_direc = ('/projects/QUIJOTE/Density_fields_3D/' + files_placeholder)
        os.chdir(curr_direc)
        redshift_zero = data_files[1].copy()
        redshift_point_five = data_files[0].copy()
        data_files[0] = redshift_zero
        data_files[1] = redshift_point_five
        z_0, z_5, z_1c, z_2, z_3 = data_files
        density_data_z0 = np.load(z_0)
        storing_data[i, :, :] = density_data_z0

    # reformat data to put into pytorch object

    # training set
    indices = np.arange(0, int(n*0.9), 1)
    train_inp = storing_data[indices, :, :]
    train_out = params_reformatted[:, indices]

    train_out_reshaped = einops.rearrange(train_out, 'param num -> num () param')
    train_inp_reshaped = einops.rearrange(train_inp, 'number x y z -> number () x y z ')

    train_out_torch = torch.tensor(train_out_reshaped).float()
    train_inp_torch = torch.tensor(train_inp_reshaped).float()

    # validation set

    indices_val = np.arange(int(n*0.9), n, 1)
    val_inp = storing_data[indices_val, :, :]
    val_out = params_reformatted[:, indices_val]

    val_out_reshaped = einops.rearrange(val_out, 'param num -> num () param')
    val_inp_reshaped = einops.rearrange(val_inp, 'number x y z -> number () x y z ')

    val_out_torch = torch.tensor(val_out_reshaped).float()
    val_inp_torch = torch.tensor(val_inp_reshaped).float()

    dataset_val = data_utils.TensorDataset(val_inp_torch, val_out_torch)

    # reshape original data
    params_reshaped = einops.rearrange(params_reformatted, 'param num -> num param')
    stored_data_reshaped = einops.rearrange(storing_data, 'number x y z -> number () x y z ')

    # load data into pytorch object
    inps = torch.tensor(
        stored_data_reshaped).float()  # this will be the simulation data, 2000 must always be the first dimension
    tgts = torch.tensor(params_reshaped[:n]).float()  # these will be parameters, 2000 must be the first dimension
    #dataset = data_utils.TensorDataset(inps, tgts)

    print('Finished reading data')
    # Also return a zero for the errors
    return train_inp_torch, train_out_torch, val_inp_torch, val_out_torch, 0
