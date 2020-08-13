#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:49:57 2020

@author: lagerwer
"""


import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
import sys
import gc
sys.path.append('../nn_fdk/')
t = time.time()

from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
import h5py
name_exp = 'recon_pb'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))

# %%
@ex.config
def cfg():
    it_i = 2
    pix = 1024
    det_rad = 0
    nTD, nTrain = 10, int(1e6)
    nVD, nVal = 5, int(1e6)
    exp_type = 'angles'
    
    phantom = 'Fourshape_test'
    PH = '4S'
    src_rad = 10
    angles = 360
    
    # var
    I0 = 2 ** 8
    noise = ['Poisson', I0]
    specifics = 'results_per_batch'

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    # bpath = '/export/scratch2/lagerwer/data/NNFDK/'
    bpath = '/bigstore/lagerwer/data/NNFDK/'
    rec_meth = 'MSD'
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
def main(specifics, bpath, rec_meth):
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    
    # %% do NN-FDK recons
    NNFDK_obj(case)
    # # %% Set up DNNs
    list_tr = [i for i in range(10)]
    list_v = [i + 10 for i in range(5)]
    if rec_meth == 'NNFDK':
        case.NNFDK.train(4, retrain=False)
        path = '/export/scratch2/lagerwer/NNFDK_results/nnfdk_p_epoch/network_epoch'
        epochs = [10 * i for i in range(13)]
        epochs += [2 ** i for i in range(8)]
        epochs += [131]
        epochs = np.sort(epochs)
        for e in epochs:
            case.NNFDK.do(NW_path=f'{path}{e}')
            save_and_add_artifact(f'{WV_path}_NNFDK4_rec_e{e}.npy', 
                          case.NNFDK.results.rec_axis[-1])
        Q, RT = log_variables(case.NNFDK.results, Q, RT)
        case.table()
        # # print('NNFDK rec time:', case.NNFDK.results.rec_time[-1])
        gc.collect()


    elif rec_meth == 'MSD':
    # %% Do MSD
        import MSD_functions as msd
        nTD, nVD = 10, 5
        WV_path = case.WV_path + specifics 
        case.MSD = msd.MSD_class(case, data_path)
        case.rec_methods += [case.MSD]
        
        case.MSD.add2sp_list(list_tr, list_v)
        print('added lists')
        path = f'{bpath}4S_V1024_A32_SR10/L2/MSD/nTD10nVD5/net_slices_seen'
        epochs = [10 * i for i in range(20)]
        epochs += [2 ** i for i in range(17)]
        epochs = np.sort(epochs)
        for e in epochs:
            case.MSD.do(NW_path=f'{path}{e}.h5')
            save_and_add_artifact(f'{WV_path}_MSD_rec_e{e}.npy',
                              case.MSD.results.rec_axis[-1])
        # print('MSD rec time:', case.MSD.results.rec_time[-1])
    
        Q, RT = log_variables(case.MSD.results, Q, RT)
        case.table()
        gc.collect()
    # %% Do Unet
    elif rec_meth == 'Unet': 
        import Unet_functions as unet
    
        WV_path = case.WV_path + specifics 
        case.Unet = unet.Unet_class(case, data_path)
        case.rec_methods += [case.Unet]
        case.Unet.add2sp_list(list_tr, list_v)
        
        print('added lists')
        epochs = [10 * i for i in range(20)]
        path = f'{bpath}4S_V1024_A32_SR10/L2/Unet/nTD10nVD5/weights_slices_seen'
        epochs += [2 ** i for i in range(17)]
        epochs = np.sort(epochs)
        for e in epochs:
            case.Unet.do(NW_path=f'{path}{e}.torch')
            save_and_add_artifact(f'{WV_path}_Unet_rec_e{e}.npy',
                                  case.Unet.results.rec_axis[-1])
        Q, RT = log_variables(case.Unet.results, Q, RT)

    # %% Save all the QMs
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    save_table(case, WV_path)

    case = None
    return Q