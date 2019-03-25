#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:52:56 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
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
    it_j = 0
    pix = 1024
    # Specific phantom
    phantom = 'Fourshape_test'
    # Number of angles
    ang = [8, 16, 32, 64, 128]
    angles = ang[it_i]
    
    # Load data?
    f_load_path = None
    g_load_path = None
    
    # Source radius
    src_rad = 10
    # Noise specifics
    noise = None #['Poisson', I0[it_i]]
    
    # Should we reshuffle the datapoints from the training sets?
    shuffle = False
    # Should we retrain the networks?
    retrain = True
    # Total number of voxels used for training
    nVox = 1e6
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = 100
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = 100

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'ang' + str(ang[it_i])


# %%
@ex.capture
def make_map_path(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param):
    data_path, full_path = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param)
    return data_path, full_path

@ex.capture
def Create_data(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, shuffle):
    # Create training and validation data
    if shuffle:
        nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                  nTrain, nTD, nVal, nVD, Exp_bin, bin_param,
                                  shuffle_TD_VD=True)
    else:
        nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                  nTrain, nTD, nVal, nVD, Exp_bin, bin_param)

@ex.capture
def CT(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, f_load_path, g_load_path):
    voxels = [pix, pix, pix]
    det_rad = 0
    if f_load_path is not None:
        data_obj = ddf.phantom(voxels, phantom, load_data=f_load_path)
    else:
        data_obj = ddf.phantom(voxels, phantom)
    if g_load_path is not None:
        CT_obj = ddf.CCB_CT(data_obj, angles, src_rad, det_rad, noise,
                            load_data=g_load_path)
    else:
        CT_obj = ddf.CCB_CT(data_obj, angles, src_rad, det_rad, noise)
    CT_obj.init_algo()
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.automain
def main(specifics, retrain):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists('NNFDK_results'):
        os.makedirs('NNFDK_results')

    # Create the training and validation data
    t1 = time.time()
    Create_data()
    t2 = time.time()
    print(t2 - t1, 'seconds to create data')
    # Create a test dataset
    case = CT()
    t3 = time.time()
    print(t3 - t2, 'seconds to initialize CT object')

    Q = np.zeros((10, 3))
    T_rec = np.zeros(10)
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    np.save(case.WV_path + specifics + '_g.npy', case.g)
    ex.add_artifact(case.WV_path + specifics + '_g.npy')

    filts = ['Ram-Lak', 'Hann']
    for f in filts:
        case.FDK.do(f)
    Q[:2, :] = case.FDK.results.Q
    T_rec[:2] = case.FDK.results.rec_time

    np.save(case.WV_path + specifics + '_FDKHN_rec.npy',
            case.FDK.results.rec_axis[-1])
    ex.add_artifact(case.WV_path + specifics + '_FDKHN_rec.npy')
    print('Finished FDKs')

    T_tr = []
    for i in range(5):
        t = time.time()
        if retrain:
            case.NNFDK.train(2 ** i, retrain=True)
        else:
            case.NNFDK.train(2 ** i)
        T_tr += [time.time() - t]
        NW_full = h5py.File(full_path + 'network_' + str(2 ** i) + '.hdf5', 'r')
        NW = h5py.File(case.WV_path + 'network_' + str(2 ** i) + '.hdf5', 'w')

        NW_full.copy(str(case.NNFDK.network[-1]['nNW']), NW, name='NW')
        NW_full.close()
        NW.close()
        ex.add_artifact(case.WV_path + 'network_' + str(2 ** i) + '.hdf5')
        case.NNFDK.do()
        np.save(case.WV_path + specifics + '_NNFDK' + str(2 ** i) + '_rec.npy',
                case.NNFDK.results.rec_axis[-1])
        ex.add_artifact(case.WV_path + specifics + '_NNFDK' + str(2 ** i) + \
                        '_rec.npy')
    print('Finished NNFDKs')
    Q[2:7, :] = case.NNFDK.results.Q
    T_rec[2:7] = case.NNFDK.results.rec_time

    # Do a SIRT reconstruction and save the results
    niter = [50, 100, 200]
    case.SIRT_NN.do(niter)
    for ni in range(len(niter)):
        np.save(case.WV_path + specifics + '_SIRT' + str(niter[ni]) + '_rec.npy',
                case.SIRT_NN.results.rec_axis[ni])
        ex.add_artifact(case.WV_path + specifics + '_SIRT' + str(niter[ni]) + \
                        '_rec.npy')

    print('Finished SIRT')
    Q[7:10, :] = case.SIRT_NN.results.Q
    T_rec[7:10] = case.SIRT_NN.results.rec_time

    case.table()
    latex_table = open(case.WV_path + specifics + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(case.WV_path + specifics + '_latex_table.txt')

    T_tr = np.asarray(T_tr)
    np.save(case.WV_path + specifics + 'T_tr.npy', T_tr)
    ex.add_artifact(case.WV_path + specifics + 'T_tr.npy')

    np.save(case.WV_path + specifics + 'T_rec.npy', T_rec)
    ex.add_artifact(case.WV_path + specifics + 'T_rec.npy')

    np.save(case.WV_path + specifics + 'Q.npy', Q)
    ex.add_artifact(case.WV_path + specifics + 'Q.npy')
    case = None
    gc.collect()
    return Q
