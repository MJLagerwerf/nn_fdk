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
import pylab


from nn_fdk import support_functions as sup
from timeit import default_timer as timer
import logging
import torch
import torch.nn as tnn
import torch.nn.functional as F
import msd_pytorch as mp
from msd_pytorch.msd_model import MSDModel
from torch.utils.data import DataLoader

# This code is copied and adapted from:
# %% https://github.com/milesial/Pytorch-UNet


class double_conv(tnn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = tnn.Sequential(
            tnn.Conv2d(in_ch, out_ch, 3, padding=1),
            tnn.BatchNorm2d(out_ch),
            tnn.ReLU(inplace=True),
            tnn.Conv2d(out_ch, out_ch, 3, padding=1),
            tnn.BatchNorm2d(out_ch),
            tnn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(tnn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(tnn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = tnn.Sequential(tnn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(tnn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = tnn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = tnn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(tnn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = tnn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(tnn.Module):
    def __init__(self, n_channels, n_classes, n_features=64):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, n_features)
        self.down1 = down(n_features, 2 * n_features)
        self.down2 = down(2 * n_features, 4 * n_features)
        self.down3 = down(4 * n_features, 8 * n_features)
        self.down4 = down(8 * n_features, 8 * n_features)
        self.up1 = up(16 * n_features, 4 * n_features)
        self.up2 = up(8 * n_features, 2 * n_features)
        self.up3 = up(4 * n_features, n_features)
        self.up4 = up(2 * n_features, n_features)
        self.outc = outconv(n_features, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        Hp, Wp = ((-H % 16), (-W % 16))
        padding = (Wp // 2, Wp - Wp // 2, Hp // 2, Hp - Hp // 2)
        reflect = tnn.ReflectionPad2d(padding)
        x = reflect(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        H2 = H + padding[2] + padding[3]
        W2 = W + padding[0] + padding[1]
        return x[:, :, padding[2] : H2 - padding[3], padding[0] : W2 - padding[1]]

    def clear_buffers(self):
        pass


class UNetRegressionModel(MSDModel):
    def __init__(self, c_in, c_out, *, loss="L2", parallel=False,):

        super().__init__(c_in, c_out, 1, 1, [1])

        loss_functions = {'L1': tnn.L1Loss(),
                          'L2': tnn.MSELoss()}

        self.criterion = loss_functions[loss]

        # Overwrite self.msd with unet
        self.msd = UNet(c_in, c_out)

        # Define the whole network:
        self.net = tnn.Sequential(self.scale_in, self.msd, self.scale_out)
        self.net.cuda()
        if parallel:
            self.net = tnn.DataParallel(self.net)

        # Train only MSD parameters:
        self.init_optimizer(self.msd)
        

# %%
def load_concat_data(inp, tar):
    if len(inp) == 1 and len(tar) == 1:
        inp = Path(inp[0]).expanduser().resolve()
        tar = Path(tar[0]).expanduser().resolve()
        print(inp)
        train_ds = mp.ImageDataset(inp / '*tiff', tar / '*.tiff')
    else:
        ds = []
        for tig, ttg in inp, tar:
            ds += mp.ImageDataset(tig, ttg)
    
        train_ds = torch.utils.data.ConcatDataset(ds)

    return train_ds
        
# %%
def train_unet(model, slab_size, fls_tr_path, save_path, epochs):
    batch_size = 1
    train_input_glob = fls_tr_path[0]
    train_target_glob = fls_tr_path[1]
    val_input_glob = False
    val_target_glob = False
    weights_path = f'{save_path}weights'
    logging.info("Load training dataset")
    # Create train (always) and validation (only if specified) datasets.
    

    train_ds = load_concat_data(train_input_glob, train_target_glob)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    
    if val_input_glob:
        logging.info("Load validation set")
        val_ds = mp.ImageDataset(val_input_glob, val_target_glob)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    else:
        logging.info("No validation set loaded")
        val_dl = None

    logging.info("Create network model")

    weights_path = Path(weights_path).expanduser().resolve()
    if weights_path.exists():
        logging.warning(f"Overwriting weights file {weights_path}")

    # The network works best if the input data has mean zero and has a
    # standard deviation of 1. To achieve this, we get a rough estimate of
    # correction parameters from the training data. These parameters are
    # not updated after this step and are stored in the network, so that
    # they are not lost when the network is saved to and loaded from disk.
    logging.info("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    logging.info("Done estimating normalization parameters")

    logging.info("Starting training...")
    best_validation_error = np.inf
    validation_error = 0.0
    training_errors = np.zeros(epochs)
    for epoch in range(epochs):
        start = timer()
        # Train
        model.train(train_dl, 1)
        # Compute training error
        train_error = model.validate(train_dl)
#        ex.log_scalar("Training error", train_error)
        logging.info(f"{epoch:05} Training error: {train_error: 0.6f}")
        training_errors[epoch] = train_error
        # Compute validation error
        if val_dl is not None:
            validation_error = model.validate(val_dl)
#            ex.log_scalar("Validation error", validation_error)
            logging.info(f"{epoch:05} Validation error: {validation_error: 0.6f}")
        # Save network if worthwile
        if validation_error < best_validation_error or val_dl is None:
            best_validation_error = validation_error
            model.save(f"{weights_path}_epoch_{epoch}.torch", epoch)
#            ex.add_artifact(f"{weights_path}_epoch_{epoch}.torch")

        end = timer()
#        ex.log_scalar("Iteration time", end - start)
        logging.info(f"{epoch:05} Iteration time: {end-start: 0.6f}")

    # Always save final network parameters
    model.save(f"{weights_path}.torch", epoch)
    np.save(f'{weights_path}_training_error', training_errors)
#    ex.add_artifact(f"{weights_path}.torch")


def save_training_results(idx, TrPath, HQPath, OutPath, spath, title): 
    inp = tifffile.imread(f'{TrPath}/stack_{idx:05d}.tiff')
    tar = tifffile.imread(f'{HQPath}/stack_{idx:05d}.tiff')
    out = tifffile.imread(f'{OutPath}/unet_{idx:05d}.tiff')
#    clim = [np.min(inp), np.max(inp)]
    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ima = ax1.imshow((inp),cmap='gray')
    fig.colorbar(ima, ax=(ax1))
    ax1.set_title('input')   
    ax1.set_xticks([],[])
    ax1.set_yticks([],[])
    ima = ax2.imshow((tar), cmap='gray')
    fig.colorbar(ima, ax=(ax2))
    ax2.set_title('target')
    ax2.set_xticks([],[])
    ax2.set_yticks([],[])
    ima = ax3.imshow((out), cmap='gray') 
    fig.colorbar(ima, ax=(ax3))
    ax3.set_xticks([],[])
    ax3.set_yticks([],[])
    ax3.set_title('output')
    fig.suptitle(title)
    fig.show()
    pylab.savefig(spath, bbox_inches='tight')

# %%
class Unet_class(ddf.algorithm_class.algorithm_class):
    def __init__(self, CT_obj, data_path, slab_size=1):
        self.CT_obj = CT_obj
        self.method = 'Unet'
        self.data_path = data_path
        self.sp_list = []
        self.t_train = []

        self.slab_size = slab_size
        self.model = UNetRegressionModel(1, 1, parallel=False)

    
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
            weights_file = Path(f'{save_path}weights_epoch_{epoch}.torch'
                                ).expanduser().resolve()
        print(epoch)
        print(weights_file)
        self.model.load(weights_file)
        # Make folder for output
        recfolder = Path(f'{save_path}Recon/')
        recfolder.mkdir(exist_ok=True)        
        outfolder = Path(f'{save_path}Recon/out/')
        outfolder.mkdir(exist_ok=True)
        if use_training_set:
            infolder = Path(f'{self.data_path}tiffs/Dataset0/FDK/')
            HQfolder = Path(f'{self.data_path}tiffs/Dataset0/HQ/')
            rec = self.CT_obj.reco_space.zero()
            MSE = np.zeros(np.shape(rec)[0])
        else:
            infolder = Path(f'{save_path}Recon/in/')
            infolder.mkdir(exist_ok=True)
            rec = self.CT_obj.FDK.do('Hann', compute_results=False) 
            sup.save_as_tiffs(rec, f'{infolder}/')
        
        input_dir = Path(infolder).expanduser().resolve()
        input_spec = input_dir
        ds = mp.ImageDataset(input_spec, input_spec)
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
                if use_training_set:
                    HQ = tifffile.imread(f'{HQfolder}/stack_{i:05d}.tiff')
                    MSE[i] = np.linalg.norm(HQ - output) ** 2
                output_path = str(output_dir / f"unet_{i:05d}.tiff")
                tifffile.imsave(output_path, output)
        
        if epoch is None:
            es = ''
        else:
            es = f'_epoch_{epoch}'
        if use_training_set:
            best = np.argmin(MSE)
            save_training_results(best, infolder, HQfolder, outfolder,
                                  f'{save_path}best{es}.png', 'Best')
            typical = np.argmin(np.abs(MSE - np.median(MSE)))
            save_training_results(typical, infolder, HQfolder, outfolder,
                                  f'{save_path}typical{es}.png', 'Typical')
            worst = np.argmax(MSE)
            save_training_results(worst, infolder, HQfolder, outfolder,
                                  f'{save_path}worst{es}.png', 'Worst')
            
        
            
        if epoch is None:
            param = f'nTD={self.nTD}'
        else:
             param = f'nTD={self.nTD}, epoch = {epoch}'
        t_rec = time.time() - t
        if compute_results:
            self.comp_results(rec, measures, '', param, t_rec)
        else:
            return rec
        
