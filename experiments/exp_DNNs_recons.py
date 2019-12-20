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
t = time.time()

ddf.import_astra_GPU()
from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'DNN_recons'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))

# %%
@ex.config
def cfg():
    phantom = 'Fourshape_test'
    it_i = 0
    if it_i == 0:
        nTD = 1
        nVD = 0
    elif it_i == 1:
        nTD = 1
        nVD = 1
    else:
        nTD = 10
        nVD = 10
    pix = 1024
    bpath = '/bigstore/lagerwer/data/NNFDK/'

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
def main(pix, phantom, nTD, nVD, bpath):
    # Specific phantom
    
    if phantom == 'Fourshape_test':
        PH = '4S'
        src_rad = 10
        noise = ['Poisson', 2 ** 8]
    elif phantom == 'Defrise':
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
    t1 = time.time()
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                     Exp_bin, bin_param, nTD + nVD,
                                     base_path=bpath)
    print('Creating training and validation datasets took', time.time() - t1,
          'seconds')
    
    # %% Create a test phantom
    voxels = [pix, pix, pix]
    # Create a data object
    t2 = time.time()
    data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad, det_rad)#,
    #                       compute_xHQ=True)
    print('Making phantom and mask took', time.time() -t2, 'seconds')
    # The amount of projection angles in the measurements
    # Source to center of rotation radius
    
    t3 = time.time()
    # %% Create the circular cone beam CT class
    case = ddf.CCB_CT(data_obj)#, angles, src_rad, det_rad, noise)
    print('Making data and operators took', time.time()-t3, 'seconds')
    # Initialize the algorithms (FDK, SIRT)
    t4 = time.time()
    case.init_algo()
    
    # %% Create NN-FDK algorithm setup
    # Create binned filter space and expansion operator
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         case.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
    case.FDK_bin_nn = case.FDK_op * Exp_op
    
    # Create the NN-FDK object
    case.NNFDK = nn.NNFDK_class(case, nTrain, nTD, nVal, nVD, Exp_bin, Exp_op,
                                 bin_param, base_path=bpath)
    case.rec_methods += [case.NNFDK]
    print('Initializing algorithms took', time.time() - t4, 'seconds')
    
    # %% set up paths
    specifics = f'DNN_{PH}_NTD{nTD}NVD{nVD}'
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    data_path, full_path = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param,
                                             base_path=bpath)
    WV_path = case.WV_path + specifics 
    save_and_add_artifact(WV_path + '_g.npy', case.g)
    # %% Do FDK recons
    case.FDK.do('Hann')
    print('FDK rec time:', case.FDK.results.rec_time[0])
    
    Q, RT = log_variables(case.FDK.results, Q, RT)
    save_and_add_artifact(WV_path + '_FDKHN_rec.npy',
            case.FDK.results.rec_axis[-1])
    # %% Do NN-FDK recons
    case.NNFDK.train(4)
    case.NNFDK.do()
    print('NNFDK rec time:', case.NNFDK.results.rec_time[0])
        
    save_and_add_artifact(WV_path + '_NNFDK4_rec.npy',
                          case.NNFDK.results.rec_axis[-1])
    Q, RT = log_variables(case.NNFDK.results, Q, RT)
    
    # %% Set up DNNs
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
    gc.collect()
    return Q

