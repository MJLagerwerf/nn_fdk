#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:05:09 2019

@author: lagerwer
"""
import numpy as np
import scipy.io as sp
import os
import gc
import time

from . import support_functions as sup
from . import Create_datasets as CD
# %%
def load_dataset_adapt_voxels(data_path, idData, nVox):
    # Load the dataset
    Ds = np.load(data_path + 'Dataset' + str(idData) + '.npy')
    # Make an array with all the possible voxels
    idVox = np.arange(np.shape(Ds)[0])
    # Shuffle the voxel ids
    np.random.shuffle(idVox)
    # Take the desired number of voxels for the dataset
    Ds = Ds[idVox[:int(nVox)], :]
    return Ds


def Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                  Exp_bin, bin_param, nDatasets,
                                  base_path='/export/scratch2/lagerwer/data/NNFDK/',
                                  **kwargs):
    data_path = sup.make_data_path(pix, phantom, angles, src_rad, noise,
                                   Exp_bin, bin_param)
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Check how many datasets we have
    nD = sup.number_of_datasets(data_path, 'Dataset')
    # Check if that is enough
    if nDatasets > nD:
        print('Creating new datasets')
        # Make extra datasets till we have enough
        for i in range(nDatasets - nD):
            Dataset = CD.Create_dataset_ASTRA_sim(pix, phantom, angles,
                                                  src_rad, noise, Exp_bin,
                                                  bin_param)
            np.save(data_path + 'Dataset' + str(i + nD), Dataset)
            print('Finished making Dataset', str(i + nD))
            gc.collect()
    else:
        print('We have enough datasets')

def random_lists(nTD, nVD):
    nData = np.arange(nTD + nVD)
    np.random.shuffle(nData)
    idTrain = nData[:nTD]
    idVal = nData[nTD:]
    return idTrain, idVal


def Preprocess_Data(pix, data_path, nTrain, nTD, nVal, nVD, DS_list=False,
                    **kwargs):       
    full_path = data_path + sup.make_full_path(nTrain, nTD, nVal, nVD)
    voxTD = nTrain // nTD
    voxVD = nVal // nVD
    if not 'voxMaxData' in kwargs:
        voxMaxData = np.max([int(pix ** 3 * 0.005), 10 ** 8])
    
    if voxTD > voxMaxData or voxVD > voxMaxData:
        raise ValueError('To many voxels per dataset')

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    if not DS_list:
        idTrain, idVal = random_lists(nTD, nVD)
    else:
        idTrain = DS_list[0]
        idVal = DS_list[1]
        print('Tlist', idTrain, 'Vlist', idVal)

    count = 0
    for i in idTrain:
        TD = load_dataset_adapt_voxels(data_path, i, voxTD)
        sp.savemat(full_path + 'TD' + str(count), {'TD': TD})
        count += 1
    count = 0
    for i in idVal:
        VD = load_dataset_adapt_voxels(data_path, i, voxVD)
        sp.savemat(full_path + 'VD' + str(count), {'VD': VD})
        count += 1

