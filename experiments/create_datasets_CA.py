#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:57:37 2019

@author: lagerwer
"""

from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import os
import load_and_preprocess_CA as cap

# %%
ex = Experiment()

@ex.config
def cfg():
    it_i = 1
    path = f'/export/scratch2/lagerwer/data/FleXray/Walnuts/Walnut{it_i}' + \
                '/Projections/'
    dset = f'tubeV{2}'




# %%
@ex.capture
def load_and_preprocess(path, dset, redo):
    dataset, vecs_path = cap.load_and_preprocess(path, dset, redo=redo)
    meta = ddf.load_meta(path + dset + '/', 1)
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
def main(path, dset):

#    t = time.time()
    dataset, meta = load_and_preprocess(path, dset, redo=False)
#    B = Create_dataset(dataset, meta, ang_freq)
#    save_path = f'{path}NNFDK/{dset}'
#    np.save(f'{save_path}/Dataset{it_i-1}', B)
#    print(f'Finished creating Dataset{it_i-1}_{dset}',
#          time.time() - t, 'seconds')

    return ''

