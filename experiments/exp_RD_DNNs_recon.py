#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:16:56 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
#ddf.import_astra_GPU()
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
    it_i = 0
    bpath = '/bigstore/lagerwer/data/FleXray/'
    load_path = f'{bpath}walnuts_10MAY/walnut_21/'
    
    dsets = ['noisy', 'good']
    dset = dsets[it_i]
    pd = 'processed_data/'
    sc = 1
    ang_freqs = [1, 16]
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
    nD = 10
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = 1
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = 0
    
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    if it_i == 0:
        specifics = 'noisy'
    elif it_i in [1]:
        specifics = 'good_ang_freq' + str(ang_freq)
    
    filts = ['Hann']


# %%  
@ex.capture
def CT(load_path, dset, sc, ang_freq, Exp_bin, bin_param, nTrain, nTD, nVal,
       nVD, bpath):
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
                                   Exp_op, bin_param, dset=dset,
                                   base_path=bpath)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(dset, ang_freq, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, bpath):
    data_path, full_path = nn.make_map_path_RD(dset, ang_freq, nTrain, nTD,
                                               nVal, nVD, base_path=bpath)
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
def main(filts, specifics, nVD, nTD):
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    # %%

    print('Finished setting up')
    case.MSD = nn.MSD_class(case, case.NNFDK.data_path)
    case.rec_methods += [case.MSD]
    case.Unet = nn.Unet_class(case, case.NNFDK.data_path)
    case.rec_methods += [case.Unet]
    
    l_tr, l_v = nn.Preprocess_datasets.random_lists(nTD, nVD)
    if nVD == 0:
        list_tr = [0]
        list_v = None
    elif nVD == 1:
        list_tr = [0]
        list_v = [1]
    else:
        list_tr = [i for i in range(10)]
        list_v = [i + 10 for i in range(5)]
    
    # %% Do MSD
    
    case.MSD.add2sp_list(list_tr, list_v)
    print('added lists')
    case.MSD.do()
    print('MSD rec time:', case.MSD.results.rec_time[0])
    case.table()
    save_and_add_artifact(WV_path + '_MSD_rec.npy',
                          case.MSD.results.rec_axis[-1])
    Q, RT = log_variables(case.MSD.results, Q, RT)
    # %% Do Unet
    case.Unet.add2sp_list(list_tr, list_v)
    case.Unet.do()
    print('Unet rec time:', case.Unet.results.rec_time[0])
    save_and_add_artifact(WV_path + '_MSD_rec.npy',
                          case.Unet.results.rec_axis[-1])
    Q, RT = log_variables(case.Unet.results, Q, RT)
    
    # %%
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    print('Finished NNFDKs')
    save_table(case, WV_path)

    
    case = None
    return Q