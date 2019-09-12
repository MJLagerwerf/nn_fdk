#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:53:57 2019

@author: lagerwer
"""

import odl
import astra
import numpy as np
import ddf_fdk as ddf
import nn_fdk as nn
import time
import pylab
import imageio
import os
import NesterovGradient
import scipy.ndimage as sp
from tqdm import tqdm

# %%
def transform(obj):
    x, y, z = np.shape(obj)
    out = np.zeros((x, z, y))
    trafo = lambda image : np.transpose(np.flipud(image))    
    for i in tqdm(range(np.size(obj, 0))):
        out[i, :, :] = trafo(obj[i, :, :])
    return out


def preprocess_data_portrait(path, dset, redo):
    pp = f'{path}processed_data/'
    if not os.path.exists(pp):
        os.makedirs(pp)
    sp = f'{pp}g_{dset}'

    if not os.path.exists(f'{sp}.npy') or redo:
        dark = ddf.read_experimental_data.read_raw(path + dset, 'di0')
        dark = transform(dark)
        flat = ddf.read_experimental_data.read_raw(path + dset, 'io0')
        flat = transform(flat)
        proj = ddf.read_experimental_data.read_raw(path + dset, 'scan_0')
        proj = transform(proj)
        # if there is a dead pixel, give it the minimum photon count from proj
        max_photon_count = proj.max()
        proj[proj == 0] = max_photon_count + 1
        min_photon_count = proj.min()
        proj[proj == max_photon_count + 1] = min_photon_count
        # %%
        proj = (proj - dark) / (flat.mean(0) - dark)
        proj = -np.log(proj)
        
        # We know that the first and the last projection angle overlap,
        # so only consider the first
        proj = proj[:-1, :, :]
        
        # Make sure that the amount of detectors rows and collums are even
        if np.size(proj, 2) % 2 != 0:
            line_u = np.zeros((proj.shape[0], proj.shape[1], 1))
            proj = np.concatenate((proj, line_u), 2)
        if np.size(proj, 1) % 2 != 0:
            line_u = np.zeros((proj.shape[0], 1, proj.shape[2]))
            proj = np.concatenate((proj, line_u), 1)
        
        proj = proj[::-1,...]
        proj = np.transpose(proj, (0, 2, 1))
        np.save(sp, proj)
    else:
        proj = np.load(f'{sp}.npy')
    return proj

# %%
def do_AGD(meta, vecs, sc, g, niter):
    start = time.time()
    ang, u, v = g.shape
    g_vec = np.transpose(g.copy(), (2, 0, 1))
    vox = 1024 // sc
    pix_size = meta['pix_size'] * sc
    src_rad = meta['s2o']
    det_rad = meta['o2d']   
    magn = src_rad / (src_rad + det_rad)
    bounds = vox * pix_size * magn / 2 
    vol_geom = astra.create_vol_geom(vox, vox, vox, -bounds, bounds, -bounds,
                                     bounds, -bounds, bounds)    
    
    proj_geom = astra.create_proj_geom('cone_vec', v, u, vecs)
    
    rec = np.zeros(astra.geom_size(vol_geom), dtype=np.float32)
    rec_id = astra.data3d.link('-vol', vol_geom, rec)
    # %%
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

    #    rec_id = astra.data3d.create('-vol', vol_geom)
    
    proj_id = astra.data3d.create('-proj3d', proj_geom, g_vec)
    
    astra.plugin.register(NesterovGradient.AcceleratedGradientPlugin)
    cfg_agd = astra.astra_dict('AGD-PLUGIN')
    cfg_agd['ProjectionDataId'] = proj_id
    cfg_agd['ReconstructionDataId'] = rec_id
    cfg_agd['ProjectorId'] = projector_id
    cfg_agd['option'] = {}
    cfg_agd['option']['MinConstraint'] = 0
    alg_id = astra.algorithm.create(cfg_agd)
    
    # Run Nesterov Accelerated Gradient Descent algorithm with 'nb_iter' iterations

    astra.algorithm.run(alg_id, niter)
    rec = np.transpose(rec, (2, 1, 0))
    # %%
    pylab.figure()
    pylab.imshow(rec[vox // 2, :, :])
    pylab.figure()
    pylab.imshow(rec[:, vox // 2, :])
    pylab.figure()
    pylab.imshow(rec[:, :, vox // 2])

    
    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)
    print((time.time() - start), 'Finished AGD 50 reconstructionn')
    return rec

# %%
def load_and_preprocess(path, dset, redo=False, ang_freq=None):
    preprocess_data_portrait(path, dset, redo)
    pp = f'{path}processed_data/'
    if not (os.path.exists(f'{pp}ground_truth.npy') and \
            os.path.exists(f'{pp}mask.npy')) or redo:
        print('Computing mask and ground truth for this dataset')
        compute_GS_and_mask(path, redo)
    else:
        print('Already computed mask and ground truth for this dataset')

    dataset = {'g' : f'{pp}g_{dset}.npy',
               'ground_truth' : f'{pp}ground_truth.npy',
               'mask' : f'{pp}mask.npy'}
    vecs = np.loadtxt(f'{path}{dset}/scan_geom_corrected.geom')[:-1, :] / 10
    if ang_freq is not None:
        vecs = vecs[::ang_freq, :]
    return dataset, vecs
    

# %%
def compute_GS_and_mask(path, redo):
    save_path = f'{path}processed_data/'
    
    niter = 50
    sc = 2
    vecs = np.zeros((0, 12))
    g = np.zeros((0, 768, 972))
    for i in [1, 2, 3]:
        dset = f'tubeV{i}'
        lp = f'{path}{dset}/'
        lpvecs = f'{lp}scan_geom_corrected.geom'
        vecs = np.append(vecs, np.loadtxt(lpvecs)[:-1, :] / 10, axis=0)
        g = np.append(g, preprocess_data_portrait(path, dset, redo=redo), 
                      axis=0)
        if i == 2:
            meta = ddf.load_meta(lp, 1)
            

    rec = do_AGD(meta, vecs, sc, g, niter)
    
    vox = np.shape(rec)[0]
    
    save = np.zeros((3, vox, vox))
    save[0, :, :], save[1, :, :] = rec[:, :, vox // 2], rec[:, vox // 2, :]
    save[2, :, :] = rec[vox // 2, :, :]
    np.save(f'{save_path}rec_ax_AGD50', save)
    # %%
    np.save(f'{save_path}rec_AGD50', rec)
    end = time.time()
    
    # %%
    rec *= (rec > 0.03)
    edge = vox // 32
    edge_t = np.zeros(np.shape(rec))
    edge_t[edge: -edge, edge: -edge, edge: -edge] = 1
    rec *= edge_t
    del edge_t
    save[0, :, :], save[1, :, :] = rec[:, :, vox // 2], rec[:, vox // 2, :]
    save[2, :, :] = rec[vox // 2, :, :]
    np.save(f'{save_path}GT_ax', save)
    diter = int(1.5 * 2 **(np.log2(vox) - 5))
    it = 5
    mask = sp.binary_erosion(rec, iterations=it)
    mask = sp.binary_dilation(mask, iterations=diter + it)
    save[0, :, :], save[1, :, :] = mask[:, :, vox // 2], mask[:, vox // 2, :]
    save[2, :, :] = mask[vox // 2, :, :]
    np.save(f'{save_path}mask_ax', save)
    # %%
    
    np.save(f'{save_path}ground_truth.npy', rec)
    np.save(f'{save_path}mask.npy', mask)
        
    t3 = time.time()
    print(t3-end, 'Finished computing mask and ground truth')

# %%
#path = '/export/scratch2/lagerwer/data/FleXray/Walnuts/Walnut1/Projections/'
#dset = f'tubeV{2}'
#pd = 'processed_data/'
#ang_freq = 1
#redo = False
#sc = 4
#
#load_and_preprocess(path, dset, redo=redo)