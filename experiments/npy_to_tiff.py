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
lpath = '/export/scratch2/lagerwer/NNFDK_results/FDK_RD/'
path1 = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/walnut_'
path2 = '/tiffs/'

dset = 'noisy'
AF = 1
rfiles = [4 * i + 1 for i in range(21)]
dfiles = ['{:02d}'.format(1 + i) for i in range(21)]
for i in range(21):
    arr = np.load(f'{lpath}{rfiles[i]}/RD_{dset}_AF{AF}_FDKHN_obj.npy')
    sp = f'{path1}{dfiles[i]}/{dset}{path2}FDK/'
#    print(arr)
#    print(sp)
    os.makedirs(sp)
    for i in tqdm(range(np.size(arr, axis=-1))):        
        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])


for f in dfiles:
    lpath = f'{path1}{f}/processed_data/'
    arr = np.load(f'{lpath}ground_truth.npy')
    sp = f'{path1}{dfiles[i]}/{dset}{path2}GS/'
#    print(arr)
#    print(sp)
    os.makedirs(sp)
    for i in tqdm(range(np.size(arr, axis=-1))):        
        io.imsave('{}stack_{:05d}.tiff'.format(sp, i), arr[:, :, i])