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
                                  nTrain, nTD, nVal, nVD, Exp_bin, bin_param,
                                  **kwargs):

    data_path, full_path = sup.make_map_path(pix, phantom, angles, src_rad, noise,
                                         nTrain, nTD, nVal, nVD, Exp_bin,
                                         bin_param)
    voxTD = nTrain // nTD
    voxVD = nVal // nVD
    voxMaxData = np.max([int(pix ** 3 * 0.005), 10 ** 5])
    if voxTD > voxMaxData or voxVD > voxMaxData:
        raise ValueError('To many voxels per dataset')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Check how many datasets we have
    nD = sup.number_of_datasets(data_path, 'Dataset')
    # Check if that is enough
    if nTD + nVD > nD:
        print('Creating new datasets')
        # Make extra datasets till we have enough
        for i in range(nTD + nVD - nD):
            Dataset = CD.Create_dataset_ASTRA(pix, phantom, angles, src_rad,
                                              noise, Exp_bin, bin_param)
            np.save(data_path + 'Dataset' + str(i + nD), Dataset)
            print('Finished making Dataset', str(i + nD))
            gc.collect()
    else:
        print('We have enough datasets')
    # We now have nTD + nVD (or more) datasets
    if 'shuffle_TD_VD' in kwargs:
        shuffle = True
    else:
        shuffle = False
    if not shuffle:
        if os.path.exists(full_path):
            cTD = sup.number_of_datasets(full_path, 'TD')
            if cTD != nTD:
                raise ValueError('Something went wrong, not the correct number',
                                 'of training datasets,', str(cTD), 'instead of',
                                 str(nTD))
            cVD = sup.number_of_datasets(full_path, 'VD')
            if cVD != nVD:
                raise ValueError('Something went wrong, not the correct number',
                                 'of validation datasets,', str(cVD), 'instead of',
                                 str(nVD))
            print('Training and validation datasets are already available')
        else:
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            # Make pick datasets for the training and validation randomly
            nData = np.arange (nTD + nVD)
            np.random.shuffle(nData)
            idTrain = nData[:nTD]
            idVal = nData[nTD:]
            count = 0
            t1 = time.time()
            for i in idTrain:
                TD = load_dataset_adapt_voxels(data_path, i, voxTD)
                sp.savemat(full_path + 'TD' + str(count), {'TD': TD})
                count += 1
            print(time.time() -t1, 'seconds to load, take voxels and save training datasets')
            t2 = time.time()
            count = 0
            for i in idVal:
                VD = load_dataset_adapt_voxels(data_path, i, voxVD)
                sp.savemat(full_path + 'VD' + str(count), {'VD': VD})
                count += 1
            print(time.time() -t2, 'seconds to load, take voxels and save validation datasets')
    else:
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        # Make pick datasets for the training and validation randomly
        nData = np.arange (nTD + nVD)
        np.random.shuffle(nData)
        idTrain = nData[:nTD]
        idVal = nData[nTD:]
        count = 0
        t1 = time.time()
        for i in idTrain:
            TD = load_dataset_adapt_voxels(data_path, i, voxTD)
            sp.savemat(full_path + 'TD' + str(count), {'TD': TD})
            count += 1
        print(time.time() -t1, 'seconds to load, take voxels and save training datasets')
        t2 = time.time()
        count = 0
        for i in idVal:
            VD = load_dataset_adapt_voxels(data_path, i, voxVD)
            sp.savemat(full_path + 'VD' + str(count), {'VD': VD})
            count += 1
        print(time.time() -t2, 'seconds to load, take voxels and save validation datasets')
    gc.collect()