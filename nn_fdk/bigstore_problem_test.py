#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:21:09 2019

@author: lagerwer
"""

import msdnet 
from tqdm import tqdm
import gc


save_path = '/bigstore//lagerwer/data/NNFDK/DF_V1024_A360_SR2/L2/MSD/'

for i in tqdm(range(10000)):
    if i % 3 == 0:
        n = msdnet.network.MSDNet.from_file(f'{save_path}nTD1nVD0/regr_params.h5',
                                            gpu=True)
    if i % 3 == 1:
        n = msdnet.network.MSDNet.from_file(f'{save_path}nTD1nVD1/regr_params.h5',
                                            gpu=True)
    if i % 3 == 2:
        n = msdnet.network.MSDNet.from_file(f'{save_path}nTD10nVD5/regr_params.h5',
                                            gpu=True)
    n = None
    gc.collect()