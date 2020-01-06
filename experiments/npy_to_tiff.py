#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:21:30 2019

@author: lagerwer
"""

import imageio as io
import numpy as np
import os
from tqdm import tqdm
# %%
lpath = '/bigstore/lagerwer/NNFDK_results/FDK_RD_CA/'
path1 = '/bigstore/lagerwer/data/FleXray/Walnuts/Walnut'
path2 = '/tiffs/'


AF = 4
dset = 'tubeV2'
#rfiles = [2 * i + 1 for i in range(21)]
rfiles = [i + 1 for i in range(21)]
#dfiles = ['{:02d}'.format(1 + i) for i in range(21)]
dfiles = [i + 1 for i in range(21)]
#for i in tqdm(range(21)):
#    arr = np.load(f'{lpath}{rfiles[i]}/RD_{dset}_AF{AF}_FDKHN_obj.npy')
#    sp = f'{path1}{dfiles[i]}Projections/processed_data{path2}{dset}/FDK/'
##    sp = f'{path1}{dfiles[i]}/{dset}{path2}FDK/'
##    print(arr)
##    print(sp)
#    os.makedirs(sp)
#    for i in (range(np.size(arr, axis=-1))):        
#        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])

for i in tqdm(range(21)):
    lpath = f'{path1}{dfiles[i]}/Projections/processed_data'
    arr = np.load(f'{lpath}/ground_truth.npy')
    sp = f'{path1}{dfiles[i]}Projections/processed_data{path2}GS/'
#    sp = f'{path1}{dfiles[i]}/{dset}{path2}GS/'
#    print(arr)
#    print(sp)
    os.makedirs(sp)
    for i in range(np.size(arr, axis=-1)):        
        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])