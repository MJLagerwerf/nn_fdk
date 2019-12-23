#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:21:09 2019

@author: lagerwer
"""

import msdnet 
from tqdm import tqdm
import gc
import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
t = time.time()

ddf.import_astra_GPU()
from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'MSD'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))

# %%
@ex.config
def cfg():
    save_path = '/export/scratch2/lagerwer/data/NNFDK/4S_V256_A360_SR10_I0256/L2/MSD/'


@ex.automain
def main(save_path):
    # Specific phantom

    for i in tqdm(range(5000)):
    
        n = msdnet.network.MSDNet.from_file(f'{save_path}nTD1nVD1/regr_params.h5',
                                                gpu=True)

    return