#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:16:56 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import pylab
import os
import time
import h5py

# %%
path = 'python_data/results/'
ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 1

    load_path = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/walnut_21/'
    
    
    dsets = ['noisy', 'good', 'good', 'good']
    dset = dsets[it_i]
    pd = 'processed_data/'
    sc = 1
    ang_freqs = [1, 8, 16, 32]
    ang_freq = ang_freqs[it_i]
    pix = 768 // sc

    # Load data?
    f_load_path = None
    g_load_path = None
    
    # Noise specifics

    # Should we retrain the networks?
    retrain = True
    # Total number of voxels used for training
    nVox = 1e6
    nD = 8
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = nD
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = nD
    
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    if it_i == 0:
        specifics = 'noisy'
    elif it_i in [1, 2, 3]:
        specifics = 'good_ang_freq' + str(ang_freq)
    
    filts = ['Ram-Lak', 'Hann']


# %%  
@ex.capture
def CT(load_path, dset, sc, ang_freq, Exp_bin, bin_param, nTrain, nTD, nVal,
       nVD):
    dataset = ddf.load_and_preprocess_real_data(load_path, dset, sc)
    meta = ddf.load_meta(load_path + dset + '/', sc)
    pix_size = meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']
    
    data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,
                 zoom=False)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
#    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param, dset=dset)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(dset, ang_freq, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param):
    data_path, full_path = nn.make_map_path_RD(dset, ang_freq, nTrain, nTD,
                                               nVal, nVD)
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
def main(it_i, retrain, filts, specifics):
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    save_and_add_artifact(WV_path + '_g.npy', case.g)


    for i in range(len(filts)):
        case.FDK.do(filts[i])
    Q, RT = log_variables(case.FDK.results, Q, RT)

    save_and_add_artifact(WV_path + '_FDKHN_rec.npy',
            case.FDK.results.rec_axis[-1])
    
    print('Finished FDKs')
    TT = np.zeros(5)
    for i in range(5):
        if i == 0:
            case.NNFDK.train(2 ** i, retrain=retrain)
        else:
            case.NNFDK.train(2 ** i, retrain=retrain, preprocess=False)
    
        TT[i] = case.NNFDK.train_time
        save_network(case, full_path, 'network_' + str(2 ** i) + '.hdf5')
        
        case.NNFDK.do()
        save_and_add_artifact(WV_path + '_NNFDK'+  str(2 ** i) + 
                               '_rec.npy', case.NNFDK.results.rec_axis[-1])

    save_and_add_artifact(WV_path + '_TT.npy', TT)
        
    Q, RT = log_variables(case.NNFDK.results, Q, RT)
    
    if it_i == 0:
        niter = [20, 50, 100]
    elif it_i in [1, 2, 3]:
        niter = [50, 100, 200]
    case.SIRT_NN.do(niter)
    for ni in range(len(niter)):
        save_and_add_artifact(WV_path + '_SIRT' + str(niter[ni]) + '_rec.npy',
                              case.SIRT_NN.results.rec_axis[ni])


    Q, RT = log_variables(case.SIRT_NN.results, Q, RT)
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    print('Finished NNFDKs')
    save_table(case, WV_path)

    
    case = None
    gc.collect()
    return Q
