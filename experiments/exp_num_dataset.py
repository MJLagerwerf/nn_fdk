#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:30:45 2019

@author: lagerwer
"""


import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
import h5py
import time
import pylab
import os
import gc
import scipy.io as sp

from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'num_datasets'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))
#url=mongo_url, db_name='sacred'))
# %%
def load_dataset_adapt_voxels_mult(data_path, idData, nVox, num_dat):
    # Load the dataset
    Ds = np.load(data_path + 'Dataset' + str(idData) + '.npy')
    
    DS_out = np.zeros((num_dat, int(nVox), np.size(Ds, 1)))
    # Make an array with all the possible voxels
    idVox = np.arange(np.shape(Ds)[0])
    # Make an array with all the possible voxels
    np.random.shuffle(idVox)
    for i in range(num_dat):
        DS_out[i, :, :] = Ds[i * int(nVox):(i + 1) * int(nVox), :]
    
    return DS_out


def make_custom_data(data_path, nTests, nTrain, nTD, nVal, nVD):
    # Walnut 21 is our test data, we cannot use that one
    curData = nn.number_of_datasets(data_path, 'Dataset') - 1
    # The total number of datasets we need
    nDtotal = 2 * nTests * nTD
    
    if nDtotal % curData == 0:
        num_dat = nDtotal // curData
    else:
        num_dat = nDtotal // curData + 1
    full_path = data_path + nn.make_full_path(nTrain, nTD, nVal, nVD)
    voxTD = nTrain // nTD
    voxVD = nVal // nVD

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    DD = curData // 2
    for i in range(DD):
        TDs = load_dataset_adapt_voxels_mult(data_path, i, voxTD, num_dat)
        for j in range(num_dat):
            sp.savemat(full_path + 'TD' + str(i + j * DD),
                       {'TD': TDs[j, :, :]})
            
    for i in range(DD, 2 * DD):
        VDs = load_dataset_adapt_voxels_mult(data_path, i, voxVD, num_dat)
        for j in range(num_dat):
            sp.savemat(full_path + 'VD' + str((i- DD) + j * DD),
                       {'VD': VDs[j, :, :]})
# %%
            
            
@ex.config
def cfg():
    it_i = 0
    it_j = 0
    pix = 256
    # Specific phantom
    phantom = 'Fourshape_test'
    # Number of angles
    angles = 32
    # Source radius
    src_rad = 10
    # Noise specifics
    I0 = 2 ** 10
    noise = ['Poisson', I0]
    
    # Load data?
    f_load_path = None
    g_load_path = None
    # Should we retrain the networks?
    retrain = True
    # Total number of voxels used for training
    nVox = 5e5
    nD = [1, 2, 5, 10, 20, 50, 100]
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = nD
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = nD
    nNodes = 4
    nTests = 10

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'num_dat'
    filts = ['Hann']

# %%
@ex.capture
def create_datasets(pix, phantom, angles, src_rad, noise, nTD, nVD, Exp_bin,
                    bin_param, nTests):
    print('Making the IID datasets')
    make_custom_data(CT_obj.NNFDK.data_path, nTests, nTrain, nTD, nVal, nVD)
    print('Finished')


        
@ex.capture
def CT(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, f_load_path, g_load_path):
    
    voxels = [pix, pix, pix]
    det_rad = 0
    if g_load_path is not None:
        if f_load_path is not None:
            data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                                   det_rad, load_data_g=g_load_path,
                                   load_data_f=f_load_path)
        else:
            data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                               det_rad, load_data_g=g_load_path)
            
    else:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                                   det_rad)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
#    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param):
    data_path, full_path = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param)
    return data_path, full_path

@ex.capture
def save_and_add_artifact(path, arr):
    np.save(path, arr)
    ex.add_artifact(path)

@ex.capture
def save_network(case, full_path, NW_path):
    NW_full = h5py.File(full_path + NW_path, 'r')
    NW = h5py.File(case.WV_path + NW_path, 'w')

    NW_full.copy(str(case.NNFDK.network[-1]['nNW']), NW, name='NW')
    NW_full.close()
    NW.close()
    ex.add_artifact(case.WV_path + NW_path)
    
@ex.capture
def save_table(case, WV_path):
    case.table()
    latex_table = open(WV_path + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(WV_path + '_latex_table.txt')

@ex.capture
def log_variables(results, Q, RT):
    Q = np.append(Q, results.Q, axis=0)
    RT = np.append(RT, results.rec_time)
    return Q, RT
    
# %%
@ex.automain
def main(retrain, nNodes, nTests, nD, filts, specifics):
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    create_datasets()
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    save_and_add_artifact(WV_path + '_g.npy', case.g)


#    for i in range(len(filts)):
#        case.FDK.do(filts[i])
#    Q, RT = log_variables(case.FDK.results, Q, RT)
#
#    save_and_add_artifact(WV_path + '_FDKHN_rec.npy',
#            case.FDK.results.rec_axis[-1])
#    
#    
#    print('Finished FDKs')
    TT = np.zeros(nTests)
    for i in range(nTests):
        DS_list = [[], []]
        for j in range(nD):
            DS_list[0] += [j + i * nD]
            DS_list[1] += [j + i * nD]
        case.NNFDK.train(nNodes, name='_' + str(i), retrain=retrain,
                         preprocess=False, d_fls=DS_list)
        
        TT[i] = case.NNFDK.train_time
        save_network(case, full_path, 'network_' + str(nNodes) + '_' + str(i) +
                     '.hdf5')
        
        case.NNFDK.do()
        save_and_add_artifact(WV_path + '_NNFDK'+  str(nNodes) + '_' + str(i) + 
                               '_rec.npy', case.NNFDK.results.rec_axis[-1])


    save_and_add_artifact(WV_path + '_TT.npy', TT)
    Q, RT = log_variables(case.NNFDK.results, Q, RT)
    
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    print('Finished NNFDKs')
    save_table(case, WV_path)

    
    case = None
    gc.collect()
    return Q

