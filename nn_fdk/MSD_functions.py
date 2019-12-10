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
def sort_files(fls_path, dset_one=False, ratio=None):
    flsin = []
    flstg = []
    for fi, ft in zip(fls_path[0], fls_path[1]):
        flsin.extend(Path(fi).glob('*.tiff'))
        flstg.extend(Path(ft).glob('*.tiff'))

    flsin = sorted(flsin)
    flstg = sorted(flstg)

    if not dset_one:    
        return flsin, flstg
    else:
        flsin_tr, flsin_v, flstg_tr, flstg_v = [], [], [], []
    
        for i in range(len(flsin)):
            if (i % (ratio + 1)) == (ratio):
                flsin_v += [flsin[i]]
                flstg_v += [flstg[i]]
            else:
                flsin_tr += [flsin[i]]
                flstg_tr += [flstg[i]]
        return flsin_tr, flstg_tr, flsin_v, flstg_v
    

# %%
def train_msd(fls_tr_path, fls_v_path, save_path, stop_crit, ratio):
    t = time.time()
    # Define dilations in [1,10] as in paper.
    dilations = msdnet.dilations.IncrementDilations(10)
    
    # Create main network object for regression, with 100 layers,
    # [1,10] dilations, 1 input channel, 1 output channel, using
    # the GPU (set gpu=False to use CPU)
    n = msdnet.network.MSDNet(100, dilations, 1, 1, gpu=True)
    
    # Initialize network parameters
    n.initialize()
    if fls_v_path is None:
        # Define training data & validation data
        flsin_tr, flstg_tr, flsin_v, flstg_v = sort_files(fls_tr_path,
                                                          dset_one=True,
                                                          ratio=ratio)
    else:
        # Define training data
        flsin_tr, flstg_tr = sort_files(fls_tr_path)
    print('Finished defining the data', time.time() - t, 'seconds')
    # Create list of datapoints (i.e. input/target pairs)
    dats = []
    for i in range(len(flsin_tr)):
        # Create datapoint with file names
        d = msdnet.data.ImageFileDataPoint(str(flsin_tr[i]), str(flstg_tr[i]))
#        # Augment data by rotating and flipping
#        d_augm = msdnet.data.RotateAndFlipDataPoint(d)
        # Add augmented datapoint to list
        dats.append(d)
    # Note: The above can also be achieved using a utility function for such
    #'simple' cases:
    # dats = msdnet.utils.load_simple_data('train/noisy/*.tiff',
    #'train/noiseless/*.tiff', augment=True)
    
    # Normalize input and output of network to zero mean and unit variance using
    # training data images
    t = time.time()
    print('Started normalizing')
    n.normalizeinout(dats)
    print('Finished normalizing', t - time.time(), 'seconds')
    # Use image batches of a single image
    bprov = msdnet.data.BatchProvider(dats, 1)
    # %%
    # Define validation data (not using augmentation)
    datsv = []
    # we do not use all the data?
    if fls_v_path is None:
        for i in range(len(flsin_v)):
            d = msdnet.data.ImageFileDataPoint(str(flsin_v[i]),str(flstg_v[i]))
            datsv.append(d)
    else:
        flsin_v, flstg_v = sort_files(fls_v_path)
        for i in range(4, len(flsin_v), 8):
            d = msdnet.data.ImageFileDataPoint(str(flsin_v[i]),str(flstg_v[i]))
            datsv.append(d)
    # Note: The above can also be achieved using a utility function for such 
    #'simple' cases:
    # datsv = msdnet.utils.load_simple_data('val/noisy/*.tiff', 
    #'val/noiseless/*.tiff', augment=False)
    
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
    imagelog = msdnet.loggers.ImageLogger(f'{save_path}log_regr',
                                          onlyifbetter=True)
    
    # %%
    # Train network until program is stopped manually
    # Network parameters are saved in regr_params.h5
    # Validation is run after every len(datsv) (=25)
    # training steps.
    print('Started training')
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
    
    def train(self, list_tr, list_v, stop_crit=50, ratio=None):
        t = time.time()
        fls_tr_path, fls_v_path = self.add2sp_list(list_tr, list_v)
        if (list_v is None) and (ratio is None):
            raise ValueError('Pass a ratio if you want to train on one dset')
        train_msd(fls_tr_path, fls_v_path, self.sp_list[-1], stop_crit, ratio)
        print('Training took:', time.time()-t, 'seconds')
        self.t_train += [time.time() - t]

    def add2sp_list(self, list_tr, list_v):
        fls_tr_path = [[], []]
        fls_v_path = [[], []]
        lpath = f'{self.data_path}tiffs/Dataset'
        for i in list_tr:
            fls_tr_path[0] += [f'{lpath}{i}/FDK']
            fls_tr_path[1] += [f'{lpath}{i}/HQ']
        self.nTD= len(fls_tr_path[0])
        
        if list_v is None:
            self.nVD =  0
            fls_v_path = None
        else:
            for i in list_v:
                fls_v_path[0] += [f'{lpath}{i}/FDK']
                fls_v_path[1] += [f'{lpath}{i}/HQ']
            self.nVD =  len(fls_v_path[0])
        
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
        print('Loaded network')
        rec = self.CT_obj.FDK.do('Hann', compute_results=False)
        print('Done FDK reconstruciton')#/ 2 / self.CT_obj.w_detu
        sup.save_as_tiffs(rec, f'{infolder}/')

        
        # Process all test images
        
        flsin = sorted(Path(infolder).glob('*.tiff'))
#        print(flsin)
        rec = np.zeros(np.shape(rec))
        for i in tqdm(range(len(flsin))):
            # Create datapoint with only input image
            d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
            # Compute network output
            output = n.forward(d.input)
            rec[:, :, i] = output[0] 
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
#fls_path = fls_tr_path
#flsin = []
#flstg = []
#for fi, ft in zip(fls_path[0], fls_path[1]):
#    flsin.extend(Path(fi).glob('*.tiff'))
#    flstg.extend(Path(ft).glob('*.tiff'))
#
#flsin = sorted(flsin)
#flstg = sorted(flstg)
#
#flsin_tr = []
#flsin_v = []
#flstg_tr = []
#flstg_v = []
#
#for i in range(len(flsin)):
#    if i % 4 == 3:
#        print(i)
#        flsin_v += [flsin[i]]
#        flstg_v += [flstg[i]]
#    else:
#        flsin_tr += [flsin[i]]
#        flstg_tr += [flstg[i]]

        
    

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