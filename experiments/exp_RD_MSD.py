#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:34:50 2020

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
t = time.time()

#ddf.import_astra_GPU()
from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'RD_MSD'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))
    
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
    train = True
    # Total number of voxels used for training
    nVox = 1e6
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
    
    filts = ['Ram-Lak', 'Hann']


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

#%%
@ex.automain
def main(nTD, nVD, train, bpath, stop_crit, specifics):
    # Specific phantom
    # %%
    t1 = time.time()
    Q = np.zeros((0, 3))
    RT = np.zeros((0))
    
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics 
    
    print('Creating training and validation datasets took and')
    print('Making phantom and mask took', time.time() - t1, 'seconds')
    # The amount of projection angles in the measurements
    # Source to center of rotation radius
    
    # %%
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
        
    if train:
        print('Started training function')
        case.MSD.train(list_tr, list_v, stop_crit=stop_crit, ratio=3)
    
    else:
        case.MSD.add2sp_list(list_tr, list_v)
        case.MSD.do()
        # %%
        print('MSD rec time:', case.MSD.results.rec_time[0])
#        print('NNFDK rec time:', case.NNFDK.results.rec_time[0])
#        print('FDK rec time:', case.FDK.results.rec_time[0])
        # %%
        save_path = '/bigstore/lagerwer/NNFDK_results/figures/'
        pylab.close('all')
        case.table()
        case.show_phantom()
        case.MSD.show(save_name=f'{save_path}MSD_{specifics}_nTD{nTD}_nVD{nVD}')
    return    
