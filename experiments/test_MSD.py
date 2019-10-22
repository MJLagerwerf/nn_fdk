#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:53:20 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
import time
import pylab
t = time.time()
# %%
pix = 1024
# Specific phantom
phantom = 'Fourshape_test'

if phantom == 'Fourshape_test':
    PH = '4S'
    src_rad = 10
    noise = ['Poisson', 2 ** 8]
elif phantom == 'Defrise':
    PH = 'DF'
    src_rad = 2
    noise = None


# Number of angles
angles = 360
# Source radius
det_rad = 0
# Noise specifics

# Number of voxels used for training, number of datasets used for training
nTrain, nTD = 1e6, 1
# Number of voxels used for validation, number of datasets used for validation
nVal, nVD = 1e6, 1

# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2
bpath = '/bigstore/lagerwer/data/NNFDK/'


# %%
t1 = time.time()
nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                 Exp_bin, bin_param, nTD + nVD,
                                 base_path=bpath)
print('Creating training and validation datasets took', time.time() - t1,
      'seconds')

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
case.FDK.do('Hann')
case.NNFDK.train(4)
case.NNFDK.do()
# %%
case.MSD = nn.MSD_class(case, case.NNFDK.data_path)
case.rec_methods += [case.MSD]
if nVD == 0:
    list_tr, list_v = [0], None
else:
    list_tr, list_v = [0], [1]
#case.MSD.train(list_tr, list_v, stop_crit=50_000, ratio=3)
case.MSD.add2sp_list(list_tr, list_v)
case.MSD.do()

# %%


print('MSD rec time:', case.MSD.results.rec_time[0])
print('NNFDK rec time:', case.NNFDK.results.rec_time[0])
print('FDK rec time:', case.FDK.results.rec_time[0])
# %%
save_path = '/bigstore/lagerwer/NNFDK_results/figures/'
pylab.close('all')
case.table()
case.show_phantom()
case.MSD.show(clim=False, save_name=f'{save_path}MSD_{PH}_nTD{nTD}_nVD{nVD}')
case.NNFDK.show(save_name=f'{save_path}NNFDK_{PH}_nTD{nTD}_nVD{nVD}')
case.FDK.show(save_name=f'{save_path}FDK_{PH}_nTD{nTD}_nVD{nVD}')

