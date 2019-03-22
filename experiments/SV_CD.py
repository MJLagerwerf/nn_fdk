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
ang = [128]
# Source radius
src_rad = 10
# Noise specifics

noise = None

nTrain, nTD, nVal, nVD = 1e6, 100, 1e6, 100


# Specifics for the expansion operator
Exp_bin = 'linear'
bin_param = 2


# %%
for angles in ang:
    data_path, _ = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param)
    
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                 nTrain, nTD, nVal, nVD, Exp_bin, bin_param)