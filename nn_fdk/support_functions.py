#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:52:52 2019

@author: lagerwer
"""
import numpy as np
import os
from itertools import compress

import sys

# %%
def text_to_acronym(text):
    PHs = ['SL', 'CS', '22El', 'C', '3S', '4S', 'HC', 'D', 'DF', 'FB']
    phantoms = ['Shepp-Logan', 'Cluttered sphere', '22 Ellipses', 'Cube',
          'Threeshape', 'Fourshape', 'Hollow cube',
          'Derenzo', 'Defrise', 'FORBILD']
    binning = ['Full', 'constant', 'linear', 'uniform']
    EBs = ['F', 'C', 'L', 'U']
    if text == 'Fourshape_test':
        text = 'Fourshape'
    if text == 'Defrise random':
        text = 'Defrise'
    if text in phantoms:
        out = list(compress(PHs, np.isin(phantoms, text)))[0]
    if text in binning:
        out = list(compress(EBs, np.isin(binning, text)))[0]
    try:
        out
    except NameError:
        sys.exit('Typo in phantom or exp_bin, you moron, you typed: ' + str(text))
    return out

# %%
def make_data_path(pix, phantom, angles, src_rad, noise, Exp_bin, bin_param,
                   base_path='/export/scratch2/lagerwer/data/NNFDK/'):
    PH = text_to_acronym(phantom)
    EB = text_to_acronym(Exp_bin)

    if noise is None:
        data_map = PH + '_V' + str(pix) + '_A' + str(angles) + \
                    '_SR' + str(src_rad) + '/'
    else:
        data_map = PH + '_V' + str(pix) + '_A' + str(angles) + '_SR' + \
                   str(src_rad) + '_I0' + str(noise[1]) + '/'
    filter_map = EB + str(bin_param) + '/'


    data_path = base_path + data_map + filter_map
    
    return data_path

def make_full_path(nTrain, nTD, nVal, nVD):
    training_map = 'nT' + '{:.0e}'.format(nTrain) + '_nTD' + str(nTD)
    validation_map =  'nV' + '{:.0e}'.format(nVal) + '_nVD' + str(nVD) + '/'
    return training_map + validation_map

# %%
def make_map_path(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
                  Exp_bin, bin_param,
                  base_path='/export/scratch2/lagerwer/data/NNFDK/'):

    data_path = make_data_path(pix, phantom, angles, src_rad, noise, Exp_bin,
                               bin_param, base_path=base_path)
        
    
    full_path = data_path + make_full_path(nTrain, nTD, nVal, nVD)

    return data_path, full_path


# %%
def number_of_datasets(path, data_type):
    if os.path.exists(path + data_type + '0.mat'):
        nDatasets = 1
        while os.path.exists(path + data_type + str(nDatasets)+ '.mat'):
            nDatasets += 1
    elif os.path.exists(path + data_type + '0.npy'):
        nDatasets = 1
        while os.path.exists(path + data_type + str(nDatasets)+ '.npy'):
            nDatasets += 1
    else:
        nDatasets = 0
    return nDatasets


# %%
def load_results(path, nMeth, nExp, files, spec, spec_var, **kwargs):
    i = 0
    if 'name_result' in kwargs:
        Q = np.zeros((nMeth, nExp))
        for f in files:
            Q[:, i] = np.load(path + str(f) + '/' + spec + str(spec_var[i]) +
                             kwargs['name_result'] + '.npy')
            i += 1
    else:
        Q = np.zeros((nMeth, nExp, 3))
        for f in files:
            Q[:, i, :] = np.load(path + str(f)+ '/' + spec + str(spec_var[i]) +
                         '_Q.npy')
            i += 1
    return Q

