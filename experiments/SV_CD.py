#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:24:06 2019

@author: lagerwer
"""

import numpy as np
import nn_fdk as nn
import gc
# %%

pix = 1024
# Specific phantom
phantom = 'Fourshape'
# Number of angles
angles = 64
# Source radius
src_rad = 10
# Noise specifics

noise = None

nTrain, nTD, nVal, nVD = 1, 1, 1, 1

# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2

# %%
data_path, _ = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param)
# %%
for i in range(30):
    Dataset = nn.Create_dataset_ASTRA(pix, phantom, angles, src_rad, noise,
                                       Exp_bin, bin_param)
    np.save(data_path + 'Dataset' + str(i), Dataset)
    print('Finished making Dataset', str(i))
    gc.collect()
