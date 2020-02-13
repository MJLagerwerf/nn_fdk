#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:19:37 2019

@author: lagerwer
"""


import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
import h5py
import astra
t = time.time()

# ddf.import_astra_GPU()
from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
import sys
sys.path.append('../nn_fdk/')
import msdnet
from tqdm import tqdm
astra.set_gpu_index([0, 1, 2, 3])
# name_exp = 'DNN_recons'
# ex = Experiment(name_exp, ingredients=[])

# FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
# ex.observers.append(FileStorageObserver.create(FSpath))
ex = Experiment()
# %%
@ex.config
def cfg():
    phantom = 'Fourshape'
    pix = 1024
    # bpath = '/export/scratch2/lagerwer/data/NNFDK/'
    bpath = '/bigstore/lagerwer/data/NNFDK/'
    if phantom == 'Fourshape_test' or phantom == 'Fourshape':
        PH = '4S'
        src_rad = 10
        noise = ['Poisson', 2 ** 8]
    elif phantom == 'Defrise' or phantom == 'Defrise random':
        PH = 'DF'
        src_rad = 2
        noise = None
    
    # Number of angles
    angles = 360
    # Source radius
    det_rad = 0
    # Noise specifics
    
    # Number of voxels used for training, number of datasets used for training
    nTrain = 1e6
    # Number of voxels used for validation, number of datasets used for validation
    nVal = 1e6
    
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2

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
@ex.automain
def main(pix, phantom, PH,  bpath):
    # %%
    scens = [0, 1, 2]
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    data_path = [[], [], []]
    case = CT()
    print('CT object is set up')
    save_and_add_artifact(case.WV_path + f'DNN_{PH}_g.npy', case.g)
    # Do FDK recons
    case.FDK.do('Hann')
    save_and_add_artifact(case.WV_path + 'DNN_{PH}_FDKHN_rec.npy', 
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
            
        specifics = f'DNN_{PH}_NTD{nTD}NVD{nVD}'
        WV_path = case.WV_path + specifics 
        case = NNFDK_obj(CT_obj=case, nTD=nTD, nVD=nVD)
        case.NNFDK.train(4, retrain=False)
        case.NNFDK.do()
        save_and_add_artifact(WV_path + '_NNFDK4_rec.npy',
                          case.NNFDK.results.rec_axis[-1])
        Q, RT = log_variables(case.NNFDK.results, Q, RT)
        case.table()
        print('NNFDK rec time:', case.NNFDK.results.rec_time[-1])
    
    data_path = case.NNFDK.data_path
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
    if PH == '4S':
        niter = [20, 50, 100]        
    elif PH == 'DF':
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

