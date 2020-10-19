#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:44:35 2020

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import pylab
import os
import time
import h5py
import scipy.io as sp
import os

# %%
sc = 1
ang_freq = 1
pix = 768 // sc
it_i = 0
# Load data?
#f_load_path = None
#g_load_path = None

# Noise specifics

# Should we retrain the networks?
retrain = True
# Total number of voxels used for training
nVox = 1e6
nD = [1]#, 2, 4, 6, 8, 10]
# Number of voxels used for training, number of datasets used for training
nNodes = 4
nTrain = nVox
nTD = nD[it_i]
# Number of voxels used for validation, number of datasets used for validation
nVal = nVox

# ! ! ! ! Note that this is set to 0, this means we only use one dataset for training and validation
# If you want to set this to 1, you need to create another dataset below (uncomment walnut 3)
nVD = 0

# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2

filts = ['Hann']
#specifics = 'num_dsets_' + str(nD[it_i])

# %%
dset = 'noisy'
# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2

# bpath = '/home/budabp/projects/Walnut/datasets/Walnuts/nn_fdk/experiments/data_new/NNFDK/'
bpath = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/walnut_01/processed_data/'

# %% We need to compute the preprocessed training data (see eq 11 paper)
# We do not want to train on the same data that we want to reconstruct
# So we use walnut 2 to create training and validation data. 

if os.path.exists(f'{bpath}NNFDK/{dset}/Dataset0.npy'):
    pass
else:
    print('Starting preprocessing')
    # dataset_w2 = {'g' : 'Walnut2_proj.npy',
    #            'ground_truth' : 'walnut2_rec.npy',
    #            'mask': 'mask.npy'}
    dataset_w2 = {'g': f'{bpath}g_noisy.npy',
                  'ground_truth' : f'{bpath}ground_truth.npy',
                  'mask': f'{bpath}mask.npy'}
    
    # dataset_w3 = {'g' : 'Walnut3_proj.npy',
    #            'ground_truth' : 'walnut3_rec.npy',
    #            'mask': 'mask.npy'}
    
    # I found this details by myself in .txt file which came with dataset
    # This is correct, but my code assumes cm's and these are mm's --> /10
    pix_size = 0.149600 / 10 
    src_rad = 66.001404 / 10
    det_rad = (199.006195-66.001404) / 10
    
    # Create Dataset0.npy from walnut2
    B = nn.Create_dataset_ASTRA_real(dataset_w2, pix_size, src_rad, det_rad, 
                                  ang_freq, Exp_bin, bin_param, vox=pix)
    
    save_path = f'{bpath}NNFDK/{dset}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(f'{save_path}/Dataset0', B)
    
print('Finished preprocessing')
# # Create Dataset1.npy from walnut3
# B = nn.Create_dataset_ASTRA_real(dataset_w3, pix_size, src_rad, det_rad, 
#                               ang_freq, Exp_bin, bin_param, vox=pix)

# np.save(f'{save_path}/Dataset1', B)


# %%
# This is the walnut that we want to reconstruct
# dataset = {'g' : 'Walnut1_proj.npy',
#            'ground_truth' : 'walnut1_rec.npy',
#            'mask': 'mask.npy'}
dataset = {'g': f'{bpath}g_noisy.npy',
              'ground_truth' : f'{bpath}ground_truth.npy',
              'mask': f'{bpath}mask.npy'}

# I found this details by myself in .txt file which came with dataset
# This is correct, but my code assumes cm's and these are mm's --> /10
pix_size = 0.149600 / 10 
src_rad = 66.001404 / 10
det_rad = (199.006195-66.001404) / 10


# We do not have to defien the vox here, this will be automatically 768 // sc
data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq, 
                         zoom=False)


case = ddf.CCB_CT(data_obj)

case.init_algo()

# %% Create NN-FDK algorithm setup
# Create binned filter space and expansion operator
spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                      case.filter_space,
                                                      interp=Exp_bin)
# Create the FDK binned operator
#case.FDK_bin_nn = case.FDK_op * Exp_op

# Create the NN-FDK object
case.NNFDK = nn.NNFDK_class(case, nTrain, nTD, nVal, nVD, Exp_bin, Exp_op,
                              bin_param, base_path=bpath, dset=dset)
case.rec_methods += [case.NNFDK]



# %%
# case.FDK.do('Hann')
# case.FDK.show()

# %%
case.NNFDK.train(4, retrain=True)
case.NNFDK.do()
case.NNFDK.show()

# %%
case.table()