#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:59:43 2020

@author: lagerwer
"""

import numpy as np
from pathlib import Path
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
import gc
import pylab
import os
import time
import h5py
import msdnet
from tqdm import tqdm
import tifffile
from torch.utils.data import DataLoader
import torch
import shutil
# %%
path = 'python_data/results/'
ex = Experiment()

# %%
@ex.config
def cfg():
    it_i = 0
    pix = 256
    # Specific phantom
    phantom = 'Cube'
    # Number of angles
    angles = 360
    
    # Load data?
    f_load_path = None
    g_load_path = None
    
    # Source radius
    src_rad = 10
    # Noise specifics
    noise = None#['Poisson', 2 ** 8]

    # Should we retrain the networks?
    retrain = False
    # Total number of voxels used for training
    nVox = 1e2
    nD = 1
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = 1
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = 0

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = f'V{pix}'
    filts = 'Ram-Lak'
    # bpath = '/bigstore/lagerwer/data/NNFDK/'
    bpath = '/export/scratch2/lagerwer/NNFDK/'
    nTests = 5
    # loc = '/ufs/lagerwer/mount/scan7'
    loc = ''
    save_path = f'{loc}/bigstore/lagerwer/data/NNFDK/4S_V1024_A360_SR10_I0256/L2/'


# %%
@ex.capture
def create_datasets(pix, phantom, angles, src_rad, noise, nTD, nVD, Exp_bin,
                    bin_param, bpath):
    nn.Create_TrainingValidationData(pix, phantom, angles, src_rad, noise,
                                 Exp_bin, bin_param, nTD + nVD,
                                 base_path=bpath)

        
@ex.capture
def CT(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, f_load_path, g_load_path, bpath):
    
    voxels = [pix, pix, pix]
    det_rad = 0
    if g_load_path is not None:
        if f_load_path is not None:
            data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                                   det_rad, load_data_g=g_load_path,
                                   load_data_f=f_load_path)
        else:
            data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                               det_rad, load_data_g=g_load_path)
            
    else:
        data_obj = ddf.phantom(voxels, phantom, angles, noise, src_rad,
                                   det_rad)

    CT_obj = ddf.CCB_CT(data_obj)
    CT_obj.init_algo()
    spf_space, Exp_op = ddf.support_functions.ExpOp_builder(bin_param,
                                                         CT_obj.filter_space,
                                                         interp=Exp_bin)
    # Create the FDK binned operator
    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

    # Create the NN-FDK object
    CT_obj.NNFDK = nn.NNFDK_class(CT_obj, nTrain, nTD, nVal, nVD, Exp_bin,
                                   Exp_op, bin_param, base_path=bpath)
    CT_obj.rec_methods += [CT_obj.NNFDK]
    return CT_obj

# %%
@ex.capture
def make_map_path(pix, phantom, angles, src_rad, noise, nTrain, nTD, nVal, nVD,
              Exp_bin, bin_param, bpath):
    data_path, full_path = nn.make_map_path(pix, phantom, angles, src_rad,
                                             noise, nTrain, nTD, nVal, nVD,
                                             Exp_bin, bin_param,
                                             base_path=bpath)
    return data_path, full_path

@ex.capture
def save_and_add_artifact(path, arr):
    np.save(path, arr)
    ex.add_artifact(path)

@ex.capture
def save_network(case, full_path, NW_path):
    NW_full = h5py.File(full_path + NW_path, 'r')
    NW = h5py.File(case.WV_path + NW_path, 'w')

    NW_full.copy(str(case.NNFDK.network[-1]['nNW']), NW, name='NW')
    NW_full.close()
    NW.close()
    ex.add_artifact(case.WV_path + NW_path)
    
@ex.capture
def save_table(case, WV_path):
    case.table()
    latex_table = open(WV_path + '_latex_table.txt', 'w')
    latex_table.write(case.table_latex)
    latex_table.close()
    ex.add_artifact(WV_path + '_latex_table.txt')

@ex.capture
def log_variables(results, Q, RT):
    Q = np.append(Q, results.Q, axis=0)
    RT = np.append(RT, results.rec_time)
    return Q, RT

@ex.capture
def make_rec_tiffs(case, save_path):
    # Make folder for output
    recfolder = Path(f'{save_path}Recon/')
    if os.path.exists(recfolder):
        shutil.rmtree(recfolder)
    recfolder.mkdir(exist_ok=True)
    
    infolder = Path(f'{save_path}Recon/in/')
    if os.path.exists(infolder):
        shutil.rmtree(infolder)
    infolder.mkdir(exist_ok=True)
    
    outfolder = Path(f'{save_path}Recon/out/')
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    outfolder.mkdir(exist_ok=True)
    rec = case.FDK.do('Hann', compute_results=False)
    print('Done FDK reconstruciton')
    nn.save_as_tiffs(rec, f'{infolder}/')
    return recfolder, infolder, outfolder

# %%
@ex.automain
def main(nTests, retrain, filts, specifics, save_path, pix):
    Q = np.zeros((0, 3))
    RT_MSD = np.zeros(nTests)
    RT_Unet = np.zeros(nTests)
    RT_SIRT = np.zeros((nTests, 3))
    
    create_datasets()
    # Create a test dataset
    case = CT()
    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics
    print('Finished making data objects')
    case.NNFDK.train(4, retrain=False)
    for nt in range(nTests):
        # %% MSD
        t = time.time()
        dilations = msdnet.dilations.IncrementDilations(10)
        print('hoi')
        n = msdnet.network.MSDNet.from_file(
            f'{save_path}MSD/nTD1nVD1/regr_params.h5',gpu=True)
        
        recfolder, infolder, outfolder = make_rec_tiffs(case, save_path)
        flsin = sorted(Path(infolder).glob('*.tiff'))
        d = msdnet.data.ImageFileDataPoint(flsin)
        rec = np.zeros((pix, pix, pix))
        for i in tqdm(range(len(flsin))):
            # Create datapoint with only input image
            d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
            # Compute network output
            output = n.forward(d.input)
            rec[:, :, i] = output[0]
            # Save network output to file
            tifffile.imsave(outfolder / 'msd_{:05d}.tiff'.format(i), output[0])
        RT_MSD[nt] = time.time() - t
        # %% Unet
        t = time.time()
        model = nn.Unet_functions.UNetRegressionModel(1, 1, parallel=False)
        weights_file = Path(
            f'{save_path}/Unet/nTD1nVD1/weights.torch').expanduser().resolve()
        input_dir = Path(infolder).expanduser().resolve()
        input_spec = input_dir
        #        print(input_dir)
        ds = nn.Unet_functions.load_concat_data([input_spec], [input_spec])
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        # Prepare output directory
        output_dir = Path(outfolder).expanduser().resolve()
        rec = np.zeros((pix, pix, pix))
        with torch.no_grad():
            for (i, (inp, tar)) in tqdm(enumerate(dl), mininterval=5.0):
                model.set_input(inp)
                output = model.net(model.input)
                output = output.detach().cpu().squeeze().numpy()
                rec[:, :, i] = output
                output_path = str(output_dir / f"unet_{i:05d}.tiff")
                tifffile.imsave(output_path, output)
        RT_Unet[nt] = time.time() - t
    # %% Rest
        case.FDK.do(filts)
        case.NNFDK.do()
        niter = [50, 100, 200]
        case.SIRT_NN.do(niter)
        RT_SIRT[nt, :] = case.SIRT_NN.results.rec_time[-3:]

    RT_FDK = case.FDK.results.rec_time
    RT_NNFDK = case.NNFDK.results.rec_time
    save_and_add_artifact(f'{WV_path}_RT_MSD', RT_MSD)
    save_and_add_artifact(f'{WV_path}_RT_Unet', RT_Unet)
    save_and_add_artifact(f'{WV_path}_RT_FDK', RT_FDK)
    save_and_add_artifact(f'{WV_path}_RT_NNFDK', RT_NNFDK)
    save_and_add_artifact(f'{WV_path}_RT_SIRT', RT_SIRT)
    
    print('Finished NNFDKs')
    # save_table(case, WV_path)

    
    case = None
    gc.collect()
    return Q
