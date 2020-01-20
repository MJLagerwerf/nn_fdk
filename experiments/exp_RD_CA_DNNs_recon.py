#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 09:20:30 2020

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
import load_and_preprocess_CA as cap
# %%
path = 'python_data/results/'
ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 1
    bp = '/bigstore/lagerwer/data/FleXray/' 
    path = f'{bp}Walnuts/Walnut21/Projections/'
    dset = f'tubeV{2}'
    pd = 'processed_data/'
    sc = 2
    ang_freq = 4
    vox = 1024 // sc

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
    nTD = nD
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = nD
    
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'CA_RD'
    
    filts = ['Hann']
    MSD = True
    Unet = True

# %%  
@ex.capture
def CT(path, dset, sc, ang_freq, Exp_bin, bin_param, nTrain, nTD, nVal,
       nVD, vox, bp):
    dataset, vecs = cap.load_and_preprocess(path, dset, ang_freq=ang_freq)
    meta = ddf.load_meta(f'{path}{dset}/', 1)
    pix_size = sc * meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']
    
    data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,
                             vox=vox, vecs=vecs)

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
                                   base_path=bp)
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
def main(filts, specifics, nVD, nTD, MSD, Unet):
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
    if MSD:
        case.MSD.add2sp_list(list_tr, list_v)
        print('added lists')
        case.MSD.do()
        print('MSD rec time:', case.MSD.results.rec_time[0])
        case.table()
        save_and_add_artifact(WV_path + '_MSD_rec.npy',
                              case.MSD.results.rec_axis[-1])
        Q, RT = log_variables(case.MSD.results, Q, RT)
        save_table(case, WV_path)
    # %% Do Unet
    if Unet:
        case.Unet = nn.Unet_class(case, case.NNFDK.data_path)
        case.rec_methods += [case.Unet]

        case.Unet.add2sp_list(list_tr, list_v)
        case.Unet.do()
        print('Unet rec time:', case.Unet.results.rec_time[0])
        save_and_add_artifact(WV_path + '_Unet_rec.npy',
                              case.Unet.results.rec_axis[-1])
        Q, RT = log_variables(case.Unet.results, Q, RT)
    
    # %%
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    print('Finished NNFDKs')
    save_table(case, WV_path)

    
    case = None
    return Q
