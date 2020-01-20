#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:00:36 2020

@author: lagerwer
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import load_and_preprocess_CA as cap

# %%
ex = Experiment()

@ex.config
def cfg():
    it_i = 1
    bp = '/bigstore/lagerwer/data/FleXray/'
    dset = 'tubeV2'
    path = f'{bp}Walnuts/Walnut{it_i}/Projections/'
    
    ang_freq = 4
    sc = 2
    vox = 1024 // sc
    Exp_bin = 'linear'
    bin_param = 2

    pd = 'processed_data/'

    redo = False

# %%
@ex.capture
def load_and_preprocess(path, dset, sc, redo, ang_freq):
    dataset, vecs = cap.load_and_preprocess(path, dset, ang_freq=ang_freq)
    meta = ddf.load_meta(f'{path}{dset}/', 1)

    return dataset, meta

@ex.capture
def Create_dataset(dataset, meta, ang_freq, Exp_bin, bin_param):
    pix_size = meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']   
    return nn.Create_dataset_ASTRA_real(dataset, pix_size, src_rad, det_rad, 
                              ang_freq, Exp_bin, bin_param)
    
# %%
@ex.automain
def main(it_i, path, dset, sc):
    t = time.time()
    load_and_preprocess()
    save_path = path + 'NNFDK/' + dset
    B = Create_dataset()
    np.save(save_path + '/Dataset' + str(it_i - 1), B)
    print('Finished creating Dataset' + str(it_i - 1) + '_' + dset,
          time.time() - t)

    return ''



