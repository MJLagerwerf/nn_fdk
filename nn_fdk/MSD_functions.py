#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:23:30 2019

@author: lagerwer
"""

import msdnet
from pathlib import Path
import tifffile
from tqdm import tqdm
import os
import ddf_fdk as ddf
import numpy as np
import time

from . import support_functions as sup
# %%
def sort_files(fls_path):
    flsin = []
    flstg = []
    for fi, ft in zip(fls_path[0], fls_path[1]):
        flsin.extend(Path(fi).glob('*.tiff'))
        flstg.extend(Path(ft).glob('*.tiff'))

    flsin = sorted(flsin)
    flstg = sorted(flstg)
    return flsin, flstg

# %%
def train_msd(fls_tr_path, fls_v_path, save_path, stop_crit):
    # Define dilations in [1,10] as in paper.
    dilations = msdnet.dilations.IncrementDilations(10)
    
    # Create main network object for regression, with 100 layers,
    # [1,10] dilations, 1 input channel, 1 output channel, using
    # the GPU (set gpu=False to use CPU)
    n = msdnet.network.MSDNet(100, dilations, 1, 1, gpu=True)
    
    # Initialize network parameters
    n.initialize()
    
    # Define training data
    flsin, flstg = sort_files(fls_tr_path)
    # %%
    
    # Create list of datapoints (i.e. input/target pairs)
    dats = []
    for i in range(len(flsin)):
        # Create datapoint with file names
        d = msdnet.data.ImageFileDataPoint(str(flsin[i]), str(flstg[i]))
        # Augment data by rotating and flipping
        d_augm = msdnet.data.RotateAndFlipDataPoint(d)
        # Add augmented datapoint to list
        dats.append(d_augm)
    # Note: The above can also be achieved using a utility function for such 'simple' cases:
    # dats = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/noiseless/*.tiff', augment=True)
    
    # Normalize input and output of network to zero mean and unit variance using
    # training data images
    n.normalizeinout(dats)
    
    # Use image batches of a single image
    bprov = msdnet.data.BatchProvider(dats, 1)
    # %%
    # Define validation data (not using augmentation)
    flsin, flstg = sort_files(fls_v_path)
    
    datsv = []
    # we do not use all the data?
    for i in range(4, len(flsin), 8):
        d = msdnet.data.ImageFileDataPoint(str(flsin[i]),str(flstg[i]))
        datsv.append(d)
    # Note: The above can also be achieved using a utility function for such 'simple' cases:
    # datsv = msdnet.utils.load_simple_data('val/noisy/*.tiff', 'val/noiseless/*.tiff', augment=False)
    
    # Validate with Mean-Squared Error
    val = msdnet.validate.MSEValidation(datsv)
    
    # Use ADAM training algorithms
    t = msdnet.train.AdamAlgorithm(n)
    
    # %%
    # Log error metrics to console
    consolelog = msdnet.loggers.ConsoleLogger()
    # Log error metrics to file
    filelog = msdnet.loggers.FileLogger(f'{save_path}log_regr.txt')
    # Log typical, worst, and best images to image files
    imagelog = msdnet.loggers.ImageLogger(f'{save_path}log_regr', onlyifbetter=True)
    
    # %%
    # Train network until program is stopped manually
    # Network parameters are saved in regr_params.h5
    # Validation is run after every len(datsv) (=25)
    # training steps.
    msdnet.train.train(n, t, val, bprov, f'{save_path}regr_params.h5',
                       loggers=[consolelog,filelog,imagelog],
                       val_every=len(datsv), stopcrit=stop_crit)



# %%
class MSD_class(ddf.algorithm_class.algorithm_class):
    def __init__(self, CT_obj, data_path):
        self.CT_obj = CT_obj
        self.method = 'MSD'
        self.data_path = data_path
        self.sp_list = []
        self.t_train = []
    
    def train(self, list_tr, list_v, stop_crit=50):
        t = time.time()
        fls_tr_path, fls_v_path = self.add2sp_list(list_tr, list_v)
        train_msd(fls_tr_path, fls_v_path, self.sp_list[-1], stop_crit)
        print('Training took:', time.time()-t, 'seconds')
        self.t_train += [time.time() - t]

    def add2sp_list(self, list_tr, list_v):
        fls_tr_path = [[], []]
        fls_v_path = [[], []]
        lpath = f'{self.data_path}tiffs/Dataset'
        for i in list_tr:
            fls_tr_path[0] += [f'{lpath}{i}/FDK']
            fls_tr_path[1] += [f'{lpath}{i}/HQ']
        
        for i in list_v:
            fls_v_path[0] += [f'{lpath}{i}/FDK']
            fls_v_path[1] += [f'{lpath}{i}/HQ']
        self.nTD, self.nVD = len(fls_tr_path[0]), len(fls_v_path[0])
        save_path = f'{self.data_path}MSD/nTD{self.nTD}nVD{self.nVD}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.sp_list += [save_path]
        return fls_tr_path, fls_v_path

    def do(self, nr=-1, compute_results=True, measures=['MSE', 'MAE', 'SSIM']):
        t = time.time()
        save_path = self.sp_list[nr]
        # Make folder for output
        recfolder = Path(f'{save_path}Recon/')
        recfolder.mkdir(exist_ok=True)        
        infolder = Path(f'{save_path}Recon/in/')
        infolder.mkdir(exist_ok=True)
        outfolder = Path(f'{save_path}Recon/out/')
        outfolder.mkdir(exist_ok=True)
        
        # Load network from file
        n = msdnet.network.MSDNet.from_file(f'{save_path}regr_params.h5',
                                            gpu=True)
        
        rec = self.CT_obj.FDK.do('Hann', compute_results=False)
        sup.save_as_tiffs(rec, f'{infolder}/')
        # Process all test images

        flsin = sorted(Path(infolder).glob('*.tiff'))
        rec = np.zeros(np.shape(rec))
        for i in tqdm(range(len(flsin))):
            # Create datapoint with only input image
            d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
            # Compute network output
            output = n.forward(d.input)
            rec[:, :, i] = output
            # Save network output to file
            tifffile.imsave(outfolder / 'msd_{:05d}.tiff'.format(i), output[0])
        
        param = f'nTD={self.nTD}, nVD={self.nVD}'
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, '', param, t_rec)
        else:
            return rec
        

# %%        
        
#path = '/export/scratch2/lagerwer/data/NNFDK/4S_V256_A360_SR10_I0256/L2/'
#path1 = f'{path}tiffs/'
#fls_tr_path = [[f'{path1}/Dataset0/FDK/'],
#               [f'{path1}/Dataset0/HQ/']]
#fls_v_path = [[ f'{path1}/Dataset1/FDK/'], 
#              [ f'{path1}/Dataset1/HQ/']]
#
#save_path = f'{path}MSD/nTD{len(fls_tr_path[0])}nVD{len(fls_v_path[0])}/'
#if not os.path.exists(save_path):
#    os.makedirs(save_path)
#    
##train_msd(fls_tr_path, fls_v_path, save_path)
#
## Make folder for output
#outfolder = Path(f'{save_path}Recon/')
#outfolder.mkdir(exist_ok=True)
#
## Load network from file
#n = msdnet.network.MSDNet.from_file(f'{save_path}regr_params.h5', gpu=True)
#
#
## Process all test images
#flsin = sorted(Path(fls_tr_path[0][0]).glob('*.tiff'))
#
#rec = np.zeros((256, 256, 256))
#for i in tqdm(range(len(flsin))):
#    # Create datapoint with only input image
#    d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
#    # Compute network output
#    output = n.forward(d.input)
#    rec[:, :, i] = output
#    # Save network output to file
#    tifffile.imsave(outfolder / 'msd_{:05d}.tiff'.format(i), output[0])
    

    
# First, create lists of input files (noisy) and target files (noiseless)
#flsin = []
#flstg = []
#files = ['{:02d}'.format(i + 1) for i in range(21)]
##path = '/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY'
#dset = 'good_AF16'
##for i in range(1,16):
##    flsin.extend(Path(path + '/walnut_{}/{}/tiffs/FDK/'.format(files[i], dset)).glob('*.tiff'))
##    flstg.extend(Path(path + '/walnut_{}/{}/tiffs/GS/'.format(files[i], dset)).glob('*.tiff'))
#for fi, ft in zip(fls_tr_path[0], fls_tr_path[1]):
#    flsin.extend(Path(fi).glob('*.tiff'))
#    flstg.extend(Path(ft).glob('*.tiff'))
#
#flsin = sorted(flsin)
#flstg = sorted(flstg)

#flsin = []
#flstg = []
#for i in range(16, 21):
#    flsin.extend(Path(path + '/walnut_{}/{}/tiffs/FDK/'.format(files[i])).glob('*.tiff'))
#    flstg.extend(Path(path + '/walnut_{}/{}/tiffs/GS/'.format(files[i])).glob('*.tiff'))
#    
#  
#flsin = sorted(flsin)
#flstg = sorted(flstg)