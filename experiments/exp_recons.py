#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:13:20 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
import sys
sys.path.append('../nn_fdk/')
t = time.time()

from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
import h5py
name_exp = 'recons'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))

# %%
@ex.config
def cfg():
    it_i = 0
    pix = 1024
    det_rad = 0
    nTD, nTrain = 10, int(1e6)
    nVD, nVal = 5, int(1e6)
    exp_type = 'noise'
    
    if exp_type == 'noise':
        phantom = 'Fourshape'
        PH = '4S'
        src_rad = 10
        angles = 360
        
        # var
        I0s = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]
        noise = ['Poisson', I0s[it_i]]
        specifics = f'4S_I0{noise[1]}'
    elif exp_type == 'angles':
        phantom = 'Fourshape'
        PH = '4S'
        src_rad = 10
        noise = None

        # var
        angs = [8, 16, 32, 64, 128]
        angles = angs[it_i]
        specifics = f'4S_ang{angles}'
    elif exp_type == 'cone angle':
        phantom = 'Defrise random'
        PH = 'DF'
        angles = 360
        noise = None
        
        # var
        rads = [2, 3, 5, 7.5, 10]
        src_rad = rads[it_i]
        specifics = f'DF_SR{src_rad}'
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    # bpath = '/export/scratch2/lagerwer/data/NNFDK/'
    bpath = '/bigstore/lagerwer/data/NNFDK/'
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
def main(specifics):
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
    save_and_add_artifact(WV_path + '_FDKHN_rec.npy', 
                          case.FDK.results.rec_axis[-1])
    
    print('FDK rec time:', case.FDK.results.rec_time[0])
    Q, RT = log_variables(case.FDK.results, Q, RT)
    # %% do NN-FDK recons
    NNFDK_obj(case)
    case.NNFDK.train(4, retrain=False)

    case.NNFDK.do()
    save_and_add_artifact(WV_path + '_NNFDK4_rec.npy', 
                      case.NNFDK.results.rec_axis[-1])
    Q, RT = log_variables(case.NNFDK.results, Q, RT)
    case.table()
    print('NNFDK rec time:', case.NNFDK.results.rec_time[-1])
    
    # %% Set up DNNs
    list_tr = [i for i in range(10)]
    list_v = [i + 10 for i in range(5)]

    
    # %% Do MSD
    import MSD_functions as msd

    nTD, nVD = 10, 5
    WV_path = case.WV_path + specifics 
    case.MSD = msd.MSD_class(case, data_path)
    case.rec_methods += [case.MSD]
    
    case.MSD.add2sp_list(list_tr, list_v)
    print('added lists')

    case.MSD.do()
    print('MSD rec time:', case.MSD.results.rec_time[-1])
    
    save_and_add_artifact(WV_path + '_MSD_rec.npy',
                          case.MSD.results.rec_axis[-1])
    Q, RT = log_variables(case.MSD.results, Q, RT)

    # %% Do Unet
    import Unet_functions as unet
    

    WV_path = case.WV_path + specifics 
    case.Unet = unet.Unet_class(case, data_path)
    case.rec_methods += [case.Unet]
    case.Unet.add2sp_list(list_tr, list_v)
    print('added lists')

    case.Unet.do()
    print('Unet rec time:', case.Unet.results.rec_time[-1])
    
    save_and_add_artifact(WV_path + '_Unet_rec.npy',
                          case.Unet.results.rec_axis[-1])
    Q, RT = log_variables(case.Unet.results, Q, RT)

    # %% Do SIRT recons
    niter = [20, 50, 100, 200]        

    case.SIRT_NN.do(niter)
    for ni in range(len(niter)):
        save_and_add_artifact(f'{WV_path}_SIRT{niter[ni]}_rec.npy',
                              case.SIRT_NN.results.rec_axis[ni])

    Q, RT = log_variables(case.SIRT_NN.results, Q, RT)
    
    # %% Save all the QMs
    save_and_add_artifact(WV_path + '_Q.npy', Q)
    save_and_add_artifact(WV_path + '_RT.npy', RT)

    save_table(case, WV_path)

    case = None
    return Q