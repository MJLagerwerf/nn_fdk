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
def make_hann_filt(voxels, w_detu):
    rs_detu = int(2 ** (np.ceil(np.log2(voxels[0] * 2)) + 1))
    filt = np.real(np.fft.rfft(ddf.ramp_filt(rs_detu)))
    freq = 2 * np.arange(len(filt))/(rs_detu)
    filt = filt * (np.cos(freq * np.pi / 2) ** 2)  / 2 / w_detu
#    filt = filt / 2 / w_detu
    return filt
# %%
pix = 256
# Specific phantom
phantom = 'Fourshape_test'
# Number of angles
angles = 360
# Source radius
src_rad = 10
det_rad = 0
# Noise specifics
noise = ['Poisson', 2 ** 8]
# Number of voxels used for training, number of datasets used for training
nTrain, nTD = 1e6, 1
# Number of voxels used for validation, number of datasets used for validation
nVal, nVD = 1e6, 1

# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2


#'/export/scratch1/home/voxels-gpu0/data/NNFDK/'
# %%
t1 = time.time()
nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                 Exp_bin, bin_param, nTD + nVD)
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
filt = make_hann_filt(data_obj.voxels, data_obj.w_detu)
xFDK = ddf.FDK_ODL_astra_backend.FDK_astra(data_obj.g, filt,
                                           data_obj.geometry, 
                                           data_obj.reco_space, None)
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
                             bin_param)
case.rec_methods += [case.NNFDK]
print('Initializing algorithms took', time.time() - t4, 'seconds')
# %%


case.FDK.do('Hann')
# %%
case.NNFDK.train(4)
case.NNFDK.do()
# %%
#pylab.close('all')
case.table()
case.show_phantom()
#case.MSD.show(clim=False)
#case.show_xHQ()
case.NNFDK.show()
#case.NNFDK.show_filters()
#case.NNFDK.show_node_output(3)

