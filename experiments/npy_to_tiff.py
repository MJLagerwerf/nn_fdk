#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:21:30 2019

@author: lagerwer
"""

import imageio as io
import numpy as np
import os

# %%
lpath = '/export/scratch2/lagerwer/NNFDK_results/FDK_RD_CA/'
path1 = '/export/scratch2/lagerwer/data/FleXray/Walnuts/Walnut'
path2 = '/tiffs/'

files = [i + 1 for i in range(11, 21)]
for f in files:
    arr = np.load(f'{lpath}{f}/CA_RD_FDKHN_obj.npy')
    sp = f'{path1}{f}{path2}FDK/'
    os.makedirs(sp)
    for i in range(np.size(arr, axis=-1)):        
        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])

for f in files:
    lpath = f'{path1}{f}/Projections/processed_data/'
    arr = np.load(f'{lpath}ground_truth.npy')
    sp = f'{path1}{f}{path2}GS/'
    os.makedirs(sp)
    for i in range(np.size(arr, axis=-1)):        
        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])