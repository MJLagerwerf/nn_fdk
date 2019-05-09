#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:02:43 2019

@author: lagerwer
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import ddf_fdk as ddf
import time

from . import Create_datasets as CD
# %%
ex = Experiment()

@ex.config
def cfg():
    dsets = ['good', 'noisy']
    path = '/export/scratch2/lagerwer/data/FleXray/walnuts_02MAY/'
    ang_freqs = [32, 16, 8]
    sc = 8
    Exp_bin = 'linear'
    bin_param = 2
    it_i = 1


# %%
@ex.capture
def load_and_preprocess(path, dset, sc, redo):
    dataset = ddf.load_and_preprocess_real_data(path, dset, sc, redo=redo)
    meta = ddf.load_meta(path + dset + '/', sc)
    return dataset, meta

@ex.capture
def Create_dataset(dataset, meta, ang_freq, Exp_bin, bin_param):
    pix_size = meta['pix_size']
    src_rad = meta['s2o']
    det_rad = meta['o2d']   
    return CD.Create_dataset_ASTRA_real(dataset, pix_size, src_rad, det_rad, 
                              ang_freq, Exp_bin, bin_param)
    
# %%
@ex.automain
def main(it_i, path, dsets, ang_freqs, sc):
    if it_i < 10:
        case = 'walnut_0' + str(it_i) + '/'
    else:
        case = 'walnut_' + str(it_i) + '/'

    pd = 'processed_data/'
    save_path = path + case + pd
    if sc == 1:
        scaling = ''
    else:
        scaling = '_sc' + str(sc)

    dataset, meta = load_and_preprocess(path + case, dsets[0], redo=True)
    # do the high dose sparse view cases  
    for af in ang_freqs:
        t = time.time()
        B = Create_dataset(dataset, meta, af)
        np.save(save_path + 'Dataset_' + dsets[0] + '_ang_freq' + str(af) + \
                scaling, B)
        print('Finished creating Dataset_' + dsets[0] + '_ang_freq' + str(af) + \
              scaling, time.time() - t)
            
    # Do the low dose case
    t = time.time()
    ang_freq = 1
    dataset, meta = load_and_preprocess(path + case, dsets[1], redo=False)
    B = Create_dataset(dataset, meta, ang_freq)
    np.save(save_path + 'Dataset_' + dsets[1] + scaling, B)
    print('Finished creating Dataset_' + dsets[1] + scaling, time.time() - t)

    return case

