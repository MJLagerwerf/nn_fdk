#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:52:56 2019

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
    pix = 256
    # Specific phantom
    phantom = 'Fourshape'
    # Number of angles
    ang = [8, 16, 32, 64, 128, 192, 256, 320, 384]
    angles = ang[it_i]

    # Source radius
    src_rad = 10
    # Noise specifics
    noise = None #['Poisson', I0[it_i]]

    # Should we retrain the networks?
    retrain = True
    # Total number of voxels used for training
    nVox = 1e5
    nD = 1
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = nD
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = nD

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'ang' + str(ang[it_i])
    filts = ['Hann']
    bpath = '/bigstore/lagerwer/data/NNFDK/'
    # bpath = '/export/scratch2/lagerwer/NNFDK/'

# %%
@ex.capture
def create_datasets(pix, phantom, angles, src_rad, noise, nTD, nVD, Exp_bin,
                    bin_param, bpath):
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                 Exp_bin, bin_param, nTD + nVD,
                                 base_path=bpath)

        
@ex.capture
def CT(pix, phantom, angles, src_rad, noise):
    voxels = [pix, pix, pix]
    det_rad = 0
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    return CT_obj

@ex.capture
def NNFDK_obj(CT_obj, phantom, pix, angles, src_rad, noise, nTrain, nTD, nVal,
              nVD, Exp_bin, bin_param, bpath):
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                    Exp_bin, bin_param, nTD + nVD,
                                    base_path=bpath)
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param, base_path=bpath)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, bpath):
    data_path, full_path = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param,
                                             base_path=bpath)
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
def main(retrain, filts, specifics):
   # %%
    scens = [0, 1]
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    data_path = [[], [], []]
    case = CT()
    print('CT object is set up')
    save_and_add_artifact(case.WV_path + f'{specifics}_g.npy', case.g)
    # Do FDK recons
    case.FDK.do('Hann')
    save_and_add_artifact(case.WV_path + specifics + 'FDKHN_rec.npy', 
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
            

        WV_path = case.WV_path + specifics 
        case = NNFDK_obj(CT_obj=case, nTD=nTD, nVD=nVD)
        case.NNFDK.train(4, retrain=False)
        case.NNFDK.do()
        save_and_add_artifact(WV_path + f'S{S}_NNFDK4_rec.npy',
                          case.NNFDK.results.rec_axis[-1])
        Q, RT = log_variables(case.NNFDK.results, Q, RT)
        case.table()
        print('NNFDK rec time:', case.NNFDK.results.rec_time[-1])
    
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
