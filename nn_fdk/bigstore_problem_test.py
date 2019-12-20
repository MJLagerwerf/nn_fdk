#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:21:09 2019

@author: lagerwer
"""

import msdnet 
from tqdm import tqdm

save_path = '/bigstore//lagerwer/data/NNFDK/DF_V1024_A360_SR2/L2/MSD/nTD1nVD1/'

for i in tqdm(range(10000)):
    n = msdnet.network.MSDNet.from_file(f'{save_path}regr_params.h5',gpu=True)
