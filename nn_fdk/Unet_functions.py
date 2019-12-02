#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:33:34 2019

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
import torch
from on_the_fly.unet import UNetRegressionModel
from on_the_fly.dataset import (ImageDataset, BatchSliceDataset,
                                CroppingDataset)
from torch.utils.data import DataLoader

from nn_fdk import support_functions as sup
# %%
def save_network(model, path):
    path = Path(path).expanduser().resolve()
    # Clear the L and G buffers before saving:
    model.msd.clear_buffers()
    torch.save(model.net.state_dict(), path)

# %%
def train_unet(model, slab_size, fls_tr_path, save_path, epochs):
    input_spec = fls_tr_path[0]
    target_spec = fls_tr_path[1]
    weights_file = f'{save_path}weights'
    ds = ImageDataset(input_spec, target_spec)
    ds = CroppingDataset(ds, remove_slices=5, remove_sides=5)
    ds = BatchSliceDataset(ds, slab_size // 2, slab_size // 2, reflect=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    
    print("Setting normalization parameters")
    model.set_normalization(dl)

    print("Training...")
    for epoch in tqdm(range(epochs), mininterval=5.0):
        # Train
        training_error = 0.0
        for (input, target) in tqdm(dl, mininterval=5.0):
            model.learn(input, target)
            training_error += model.get_loss()
    
        training_error = training_error / len(dl)
        save_network(model, f'{weights_file}_E{epoch}')
    #    _run.log_scalar("Training error", training_error.item())

    # Always save final network parameters
    save_network(model, weights_file)
    
    


# %%
class Unet_class(ddf.algorithm_class.algorithm_class):
    def __init__(self, CT_obj, data_path, slab_size=1):
        self.CT_obj = CT_obj
        self.method = 'Unet'
        self.data_path = data_path
        self.sp_list = []
        self.t_train = []
        net_opts = dict(depth=100, width=1, loss_function='L2',
                dilation='MSD', reflect=True, conv3d=False)

        self.slab_size = slab_size
        self.model = UNetRegressionModel(
            c_in = slab_size,
            c_out = 1,
            **net_opts,
        )

    
    def train(self, list_tr, epochs=1):
        t = time.time()
        fls_tr_path = self.add2sp_list(list_tr)

        train_unet(self.model, self.slab_size, fls_tr_path, 
                   self.sp_list[-1], epochs)
        print('Training took:', time.time()-t, 'seconds')
        self.t_train += [time.time() - t]

    def add2sp_list(self, list_tr):
        fls_tr_path = [[], []]
        lpath = f'{self.data_path}tiffs/Dataset'
        for i in list_tr:
            fls_tr_path[0] += [f'{lpath}{i}/FDK']
            fls_tr_path[1] += [f'{lpath}{i}/HQ']
        self.nTD = len(fls_tr_path[0])
        
        
        save_path = f'{self.data_path}Unet/nTD{self.nTD}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.sp_list += [save_path]
        return fls_tr_path

    def do(self, epoch=None, nr=-1, compute_results=True,
           measures=['MSE', 'MAE', 'SSIM'], use_training_set=False):
        t = time.time()
        save_path = self.sp_list[nr]
        if epoch is None:
            weights_file = Path(f'{save_path}weights').expanduser().resolve()
        else:
            weights_file = Path(f'{save_path}weights_E{epoch}').expanduser(
                    ).resolve()
        self.model.load_network(save_file=weights_file)
        # Make folder for output
        recfolder = Path(f'{save_path}Recon/')
        recfolder.mkdir(exist_ok=True)        
        outfolder = Path(f'{save_path}Recon/out/')
        outfolder.mkdir(exist_ok=True)
        if use_training_set:
            infolder = Path(f'{self.data_path}tiffs/Dataset0/FDK/')
        else:
            infolder = Path(f'{save_path}Recon/in/')
            infolder.mkdir(exist_ok=True)
            rec = self.CT_obj.FDK.do('Hann', compute_results=False) 
            sup.save_as_tiffs(rec, f'{infolder}/')
        
        input_dir = Path(infolder).expanduser().resolve()
        input_spec = input_dir
        ds = ImageDataset(input_spec, input_spec)
        ds = BatchSliceDataset(ds, self.slab_size // 2, self.slab_size // 2,
                               reflect=True)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        
        # Prepare output directory
        output_dir = Path(outfolder).expanduser().resolve()
        output_dir.mkdir(exist_ok=True)

        rec = np.zeros(np.shape(rec))
        self.model.net.eval()
        with torch.no_grad():
            for (i, (inp, _)) in tqdm(enumerate(dl), mininterval=5.0):
                self.model.set_input(inp)
                output = self.model.net(self.model.input)
                output = output.detach().cpu().squeeze().numpy()
                rec[:, :, i] = output
                output_path = str(output_dir / f"unet_{i:05d}.tif")
                tifffile.imsave(output_path, output)
        
        if epoch is None:
            param = f'nTD={self.nTD}'
        else:
             param = f'nTD={self.nTD}, epoch = {epoch}'
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, '', param, t_rec)
        else:
            return rec
        
