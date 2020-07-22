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
import Unet_functions as unet
t = time.time()

from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'Unet'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))

# %%
@ex.config
def cfg():
    it_i = 0
    pix = 1024
    det_rad = 0
    nTD, nTrain = 1, int(1e6)
    nVD, nVal = 1, int(1e6)
    exp_type = 'noise'
    
    if exp_type == 'noise':
        phantom = 'Fourshape_test'
        PH = '4S'
        src_rad = 10
        angles = 360
        
        # var
        I0s = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]
        noise = ['Poisson', I0s[it_i]]

    elif exp_type == 'angles':
        phantom = 'Fourshape_test'
        PH = '4S'
        src_rad = 10
        noise = None

        # var
        angs = [8, 16, 32, 64, 128]
        angles = angs[it_i]
    elif exp_type == 'cone angle':
        phantom = 'Defrise'
        PH = 'DF'
        angles = 360
        noise = None
        
        # var
        rads = [2, 3, 5, 7.5, 10]
        src_rad = rads[it_i]
    train = True
    stop_crit = 200
    epochs = 1
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    # bpath = '/export/scratch2/lagerwer/data/NNFDK/'
    bpath = '/bigstore/lagerwer/data/NNFDK/'
# %%
    
@ex.automain
def main(pix, phantom, nTD, nTrain, nVD, nVal, train, bpath, stop_crit, epochs,
         PH, angles, src_rad, det_rad, noise, Exp_bin, bin_param):
    # Specific phantom
    
    
    # %%
    t1 = time.time()
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                     Exp_bin, bin_param, nTD + nVD,
                                     base_path=bpath)
    print('Creating training and validation datasets took', time.time() - t1,
          'seconds')
    if not train:
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
        
        # %%
        case.Unet = unet.Unet_class(case, case.NNFDK.data_path)
        case.rec_methods += [case.Unet]
    else:
        data_path = nn.make_data_path(pix, phantom, angles, src_rad, noise,
                                      Exp_bin, bin_param, base_path=bpath)
        Unet = unet.Unet_class(None, data_path)
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
        print('training')
        Unet.train(list_tr, list_v, epochs=epochs, stop_crit=stop_crit, 
                        ratio=3)
    else:
        print(f'Reconstructing...')
        case.Unet.add2sp_list(list_tr, list_v)
    
    
    # %%
        case.table()
        save_path = '/bigstore/lagerwer/NNFDK_results/figures/'
        case.Unet.show(clim=False, save_name=f'{save_path}Unet_{PH}_nTD{nTD}_nVD{nVD}')

    return    
