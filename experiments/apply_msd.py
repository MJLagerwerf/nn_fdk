#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:13:42 2019

@author: lagerwer
"""

# Import code
import msdnet
from pathlib import Path
import tifffile
from tqdm import tqdm

dset = 'noisy'
# Make folder for output
outfolder = Path('/export/scratch2/lagerwer/NNFDK_results/MSD/{}/'.format(dset))
outfolder.mkdir(exist_ok=True)

# Load network from file
n = msdnet.network.MSDNet.from_file('regr_params{}.h5'.format(dset), gpu=True)

# Process all test images
flsin = sorted(Path('/export/scratch2/lagerwer/data/FleXray/walnuts_10MAY/' \
                    'walnut_21/{}/tiffs/FDK/'.format(dset)).glob('*.tiff'))
for i in tqdm(range(len(flsin))):
    # Create datapoint with only input image
    d = msdnet.data.ImageFileDataPoint(str(flsin[i]))
    # Compute network output
    output = n.forward(d.input)
    # Save network output to file
    tifffile.imsave(outfolder / 'msd_{:05d}.tiff'.format(i), output[0])