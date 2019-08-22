#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:05:20 2018

@author: lagerwer
"""
import numpy as np
import astra
import odl
import numexpr
import time
import ddf_fdk as ddf

# %%
def sigmoid(x):
    '''Sigmoid function'''
    return numexpr.evaluate("1./(1.+exp(-x))")

def hidden_layer(x, y, q, b):
    ''' Perceptron hidden layer'''
    return numexpr.evaluate('y + q * (1 / (1 + exp(-(x - b))))')

def outer_layer(x, b, sc2_1, sc2_2):
    ''' Outer layer'''
    return numexpr.evaluate('2 * (1 / (1 + exp(-(x - b))) - .25) * sc2_1' \
                            '+ sc2_2')
# %%
def Exp_op_FFT(Exp_op, h, filter2d, Resize_Op, w_du):
    rs_filt = Resize_Op(Exp_op(h))
    f_filt = np.real(np.fft.rfft(np.fft.ifftshift(rs_filt)))
    for a in range(np.shape(filter2d)[0]):
        filter2d[a, :] = f_filt * 4 * w_du
    return filter2d


# %%
def NNFDK_astra(g, NW, geom, reco_space, Exp_op, node_output, ang_freq=None):
    # %% Create geometry
    # Make a circular scanning geometry
    ang, u, v = g.shape
    minvox = reco_space.min_pt[0]
    maxvox = reco_space.max_pt[0]
    vox = np.shape(reco_space)[0]
    vol_geom = astra.create_vol_geom(vox, vox, vox, minvox, maxvox, minvox,
                                     maxvox, minvox, maxvox)

    # Build a vecs vector from the geometry, or load it
    if type(geom) == np.ndarray:
        vecs = geom
    elif type(geom) == odl.tomo.geometry.conebeam.ConeFlatGeometry:
        vecs = ddf.astra_conebeam_3d_geom_to_vec(geom)
    proj_geom = astra.create_proj_geom('cone_vec', v, u, vecs)
    g = np.transpose(np.asarray(g), (2, 0, 1))
    # %%
    proj_id = astra.data3d.create('-proj3d', proj_geom, g)
    rec = np.zeros(astra.geom_size(vol_geom), dtype=np.float32)

    rec_tot = np.zeros(astra.geom_size(vol_geom), dtype=np.float32)


    rec_id = astra.data3d.link('-vol', vol_geom, rec)



    fullFilterSize = int(2 ** (np.ceil(np.log2(u)) + 1))
    halfFilterSize = fullFilterSize // 2 + 1
    filter2d = np.zeros((ang, halfFilterSize))
    Resize_Op = odl.ResizingOperator(Exp_op.range, ran_shp=(fullFilterSize,))
    # %% Make a filter geometry

    filter_geom = astra.create_proj_geom('parallel', w_du,  halfFilterSize,
                                         np.zeros(ang))


    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    # %%
    # Set a container list for the learned filters
    h_e = []
    if node_output:
        mid = v // 2
        node_output_axis = []
        
    for i in range(NW['nNodes']):
        h = NW['l1'][:-1, i] * 2 * NW['sc1'][0, :]
        h_e += [h]
        b = NW['l1'][-1, i] + np.sum(NW['l1'][:-1, i]) + 2 * np.dot(
                        NW['l1'][:-1, i], NW['sc1'][1, :])
        filter2d = Exp_op_FFT(Exp_op, h, filter2d, Resize_Op, w_du)

        filter_id = astra.data2d.create('-sino', filter_geom, filter2d)
        cfg['option'] = { 'FilterSinogramId': filter_id}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec_tot = hidden_layer(rec, rec_tot, NW['l2'][i], b)
        if node_output:
            rec2 = hidden_layer(rec, 0, NW['l2'][i], b)
            node_output_axis += [rec2[:, :, mid], rec2[:, mid, :],
                                 rec2[mid, :, :]]

        
    # Make a numpy array of the filter list
    h_e = np.asarray(h_e)
    # b_o = self.network['l2'][-1]
    rec_tot = outer_layer(rec_tot, NW['l2'][-1], NW['sc2'][0], NW['sc2'][1])
    rec_tot = np.transpose(rec_tot, (2, 1, 0))


    # %% Make the matrix columns of the matrix B


    # %%
    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    if node_output:
        return rec_tot, h_e, node_output_axis
    else:
        return rec_tot, h_e
