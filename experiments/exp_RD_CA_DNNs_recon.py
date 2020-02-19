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
import sys
sys.path.append('../nn_fdk/')
import astra
astra.set_gpu_index([0, 1, 2, 3])
import load_and_preprocess_CA as cap
# %%
path = 'python_data/results/'
ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 1
    it_j = 21
    bp = '/bigstore/lagerwer/data/FleXray/' 
    path = f'{bp}Walnuts/Walnut{it_j}/Projections/'
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


# %%  
@ex.capture
def CT(path, dset, sc, ang_freq, vox):
    dataset, vecs = cap.load_and_preprocess(path, dset, ang_freq=ang_freq)
    meta = ddf.load_meta(f'{path}{dset}/', 1)
    pix_size = sc * meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']
    
    data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,
                             vox=vox, vecs=vecs)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    return CT_obj

@ex.capture
def NNFDK_obj(CT_obj, dset, Exp_bin, bin_param, nTrain, nTD, nVal, nVD, bp):    
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)


    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param, dset=dset,
                                   base_path=bp)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(dset, ang_freq, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, bp):
    data_path, full_path = nn.make_map_path_RD(dset, ang_freq, nTrain, nTD,
                                               nVal, nVD, base_path=bp)
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
def main(filts, dset, specifics, nVD, nTD):
    scens = [0, 1, 2]
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    # %%
    # Do FDK recons
    case.FDK.do('Hann')
    save_and_add_artifact(case.WV_path + 'DNN__FDKHN_rec.npy', 
                          case.FDK.results.rec_axis[-1])
    
    print('FDK rec time:', case.FDK.results.rec_time[0])
    Q, RT = log_variables(case.FDK.results, Q, RT)
    for S in scens:
        if S == 0:
            nTD, nVD = 1, 0
        elif S == 1:
            nTD, nVD = 1, 1
        elif S == 2: 
            nTD, nVD = 10, 5
            
        specifics = f'DNN_'
        WV_path = case.WV_path + specifics 
        case = NNFDK_obj(CT_obj=case, nTD=nTD, nVD=nVD)
        case.NNFDK.train(4, retrain=False)
        case.NNFDK.do()
        save_and_add_artifact(WV_path + '_NNFDK4_rec.npy',
                          case.NNFDK.results.rec_axis[-1])
        Q, RT = log_variables(case.NNFDK.results, Q, RT)
        case.table()
        print('NNFDK rec time:', case.NNFDK.results.rec_time[-1])
    
    # %% Set up DNNs
    list_tr = [[0], [0], [i for i in range(10)]]
    list_v = [None, [1], [i + 10 for i in range(5)]]

    
    # %% Do MSD
    import MSD_functions as msd

    for S in scens:
        if S == 0:
            nTD, nVD = 1, 0
        elif S == 1:
            nTD, nVD = 1, 1
        elif S == 2: 
            nTD, nVD = 10, 5
        case.MSD = msd.MSD_class(case, data_path)
        case.rec_methods += [case.MSD]
        
        case.MSD.add2sp_list(list_tr[S], list_v[S])
        print('added lists')
        case.MSD.do()
        print('MSD rec time:', case.MSD.results.rec_time[-1])
        
        save_and_add_artifact(WV_path + '_MSD_rec.npy',
                              case.MSD.results.rec_axis[-1])
        Q, RT = log_variables(case.MSD.results, Q, RT)

    case.table()
    # %% Do Unet
    import Unet_functions as unet
    
    for S in scens:
        if S == 0:
            nTD, nVD = 1, 0
        elif S == 1:
            nTD, nVD = 1, 1
        elif S == 2: 
            nTD, nVD = 10, 5
        case.Unet = unet.Unet_class(case, data_path)
        case.rec_methods += [case.Unet]
        case.Unet.add2sp_list(list_tr[S], list_v[S])
        case.Unet.do()
        print('Unet rec time:', case.Unet.results.rec_time[-1])
        save_and_add_artifact(WV_path + '_MSD_rec.npy',
                              case.Unet.results.rec_axis[-1])
    
        Q, RT = log_variables(case.Unet.results, Q, RT)
    case.table()
    # %%

    niter = [50, 100, 200]

    case.SIRT_NN.do(niter)
    for ni in range(len(niter)):
        save_and_add_artifact(WV_path + '_SIRT' + str(niter[ni]) + '_rec.npy',
                              case.SIRT_NN.results.rec_axis[ni])

    Q, RT = log_variables(case.SIRT_NN.results, Q, RT)
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    save_table(case, WV_path)

    case = None

    return Q