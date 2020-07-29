#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:19:53 2020

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
name_exp = 'MSD'
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
    MSD = True
    
    train = True
    stop_crit = np.inf
    epochs = 100_000
    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    # bpath = '/export/scratch2/lagerwer/data/NNFDK/'
    bpath = '/bigstore/lagerwer/data/NNFDK/'
    save_model_pb = True
# %%
    
@ex.automain
def main(pix, phantom, nTD, nTrain, nVD, nVal, train, bpath, stop_crit, MSD,
         PH, angles, src_rad, det_rad, noise, Exp_bin, bin_param, epochs,
         save_model_pb):
    # Specific phantom

    
    # %%
    t1 = time.time()
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                     Exp_bin, bin_param, nTD + nVD,
                                     base_path=bpath)
    print('Creating training and validation datasets took', time.time() - t1,
          'seconds')
    voxels = [pix, pix, pix]
    # %% Create a test phantom
    data_path = nn.make_data_path(pix, phantom, angles, src_rad, noise,
                                      Exp_bin, bin_param, base_path=bpath)
    l_tr, l_v = nn.Preprocess_datasets.random_lists(nTD, nVD)
    list_tr = [i for i in range(10)]
    list_v = [i + 10 for i in range(5)]
    if MSD:
        import MSD_functions as msd

        MSD = msd.MSD_class(None, data_path)
        MSD.train(list_tr, list_v, stop_crit=stop_crit, ratio=3,
                  save_model_pb=save_model_pb)
    else:
        import Unet_functions as unet
        Unet = unet.Unet_class(None, data_path)
        Unet.train(list_tr, list_v, epochs=epochs, stop_crit=stop_crit, 
                        ratio=3)
    return
