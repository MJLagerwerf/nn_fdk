#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:20:52 2019

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
angles = 360
# Source radius
src_rad = 10
# Noise specifics

noise_levels = [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]

nTrain, nTD, nVal, nVD = 1e6, 100, 1e6, 100


# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2


# %%
for nl in noise_levels:
    data_path, _ = nn.make_map_path(pix, phantom, angles, src_rad,
                                             ['Gaussian', nl],
                                             nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param)
    
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad,
                                     ['Gaussian', nl], nTrain, nTD, nVal, nVD,
                                     Exp_bin, bin_param)