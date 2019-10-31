#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:43:01 2019

@author: lagerwer
"""

import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
import h5py
ddf.import_astra_GPU()
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
path = '/bigstore/lagerwer/data/FleXray/pomegranate1_02MAR/'
#path = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/walnut_01/'
dset = 'noisy'
#dset2 = 'good'
pd = 'processed_data/'
sc = 1
ang_freq = 1
redo = False
dataset = ddf.load_and_preprocess_real_data(path, dset, sc, redo=redo)
meta = ddf.load_meta(path + dset + '/', sc)

pix_size = meta['pix_size']
src_rad = meta['s2o']
det_rad = meta['o2d']
#dataset = {'g': }



# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2


# %% Create a test phantom
# Create a data object
t2 = time.time()
data_obj = ddf.real_data(dataset, pix_size, src_rad, det_rad, ang_freq,
                 zoom=True)
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

print('Initializing algorithms took', time.time() - t4, 'seconds')
# %%
offset = 1.26/360 * 2 * np.pi
rec1 = case.FDK.do('Hann', compute_results=False)
voxels = np.shape(rec1)
hann = make_hann_filt(voxels, case.w_detu)
rec = ddf.FDK_ODL_astra_backend.FDK_astra(data_obj.g, hann, case.geometry,
                                          case.reco_space, case.w_detu,
                                          ang_offset= offset)

case.FDK.comp_results(rec1, ['MSE', 'MAE_msk', 'SSIM_msk'], 0, 'no_rot', 0)
case.FDK.comp_results(rec, ['MSE', 'MAE_msk', 'SSIM_msk'], 0, 'rot', 0)

case.table()
# %%


#def show_diff(rec1, rec2, title, sp):
#    midvox = np.shape(rec1)[0] // 2
#    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
#    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#    ax1.imshow(np.rot90(rec1[:, :, midvox]- rec2[:, :, midvox]))
#    ax2.imshow(np.rot90(rec1[:, midvox, :] - rec2[:, midvox, :]))
#    ima = ax3.imshow(np.rot90(rec1[midvox, :, :] - rec2[midvox, :, :]))
#    fig.colorbar(ima, ax=(ax1, ax2, ax3))
#    fig.suptitle(title)
#    pylab.savefig(sp)
#    
#    
## %%    
#def show(rec1, title):
#    midvox = np.shape(rec1)[0] // 2
#    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
#    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#    ax1.imshow(np.rot90(rec1[:, :, midvox]))
#    ax2.imshow(np.rot90(rec1[:, midvox, :]))
#    ima = ax3.imshow(np.rot90(rec1[midvox, :, :]))
#    fig.colorbar(ima, ax=(ax1, ax2, ax3))
#    fig.suptitle(title)
#    
## %%
#
#show_diff(data_obj.f, rec1, 'GS - og rec', 'diff_GS_OG')
#
#show_diff(data_obj.f, rec, 'GS - new rec', 'diff_GS_NEW')
#
#show_diff(rec1, rec,  'new rec - og rec', 'diff_NEW_PG')
#
