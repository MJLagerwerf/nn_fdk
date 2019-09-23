#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:41:32 2019

@author: lagerwer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:34:48 2019

@author: lagerwer
"""


import numpy as np
import ddf_fdk as ddf
ddf.import_astra_GPU()
import nn_fdk as nn
import h5py
import time
import pylab
import os
import gc

from sacred.observers import FileStorageObserver
from sacred import Experiment
from os import environ
name_exp = 'node_output'
ex = Experiment(name_exp, ingredients=[])

FSpath = '/export/scratch2/lagerwer/NNFDK_results/' + name_exp
ex.observers.append(FileStorageObserver.create(FSpath))
#url=mongo_url, db_name='sacred'))

def load_network(fnNetwork, nHiddenNodes, NWs=1):
    f = h5py.File(fnNetwork + '.hdf5', 'r+')
    # Check at which network we are
#    nn = f'{NWs}/'
    nn = 'NW/'
    # load old network, get them in numpy arrays
    # TODO: Also load the list of training and validation error
    l1 = np.asarray(f[nn + 'l1'])
    l2 = np.asarray(f[nn + 'l2'])
    sc1 = np.asarray(f[nn + 'sc1'])
    sc2 = np.asarray(f[nn + 'sc2'])
    l_tE = np.asarray(f[nn + 'l_tE']) 
    l_vE = np.asarray(f[nn + 'l_vE'])
    tE = f[nn].attrs['T_MSE']
    vE = f[nn].attrs['V_MSE']
    # return old network
    print('Loaded old network, network has', str(nHiddenNodes),
          'hidden nodes')
    f.close()

    return {'l1' : l1, 'l2' : l2, 'sc1' : sc1, 'sc2' : sc2,
            'nNodes' : nHiddenNodes, 
            'l_tE' : l_tE, 'l_vE' : l_vE, 'tE' : tE, 'vE' : vE}


# %%
@ex.config
def cfg():
    pix = 1024
    # Specific phantom
    phantom = 'Fourshape_test'
    # Number of angles
    angles = 360
    # Source radius
    src_rad = 10
    # Noise specifics
    I0 = 2 ** 8
    noise = ['Poisson', I0]
    bpath = '/export/scratch3/lagerwer/data/NNFDK/'
    # Load data?
    lp = '/export/scratch2/lagerwer/NNFDK_results/NOI_var_1024/1/'
    f_load_path = None
    g_load_path = f'{lp}I0256_g.npy'
    
    # Total number of voxels used for training
    nVox = 1e6
    nD = 100
    # Number of voxels used for training, number of datasets used for training
    nTrain = nVox
    nTD = nD
    # Number of voxels used for validation, number of datasets used for validation
    nVal = nVox
    nVD = nD
    nNodes = [1, 2, 4, 8, 16]

    # Specifics for the expansion operator
    Exp_bin = 'linear'
    bin_param = 2
    specifics = 'node_output'
    filts = ['Hann']

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
#    CT_obj.FDK_bin_nn = CT_obj.FDK_op * Exp_op

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
                                             Exp_bin, bin_param, bpath)
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
    
# %%
@ex.automain
def main(nNodes, filts, specifics, lp):
    t = time.time()
    Q = np.zeros((0, 3))

    # Create a test dataset
    t0 = time.time()
    case = CT()

    # Create the paths where the objects are saved
    data_path, full_path = make_map_path()
    WV_path = case.WV_path + specifics


    t1 = time.time()
    print('Finished setting up the inverse problem. Took:', (t1 - t0) / 60,
          'minutes')

    for i in nNodes:
        if i == 1:
            case.NNFDK.network = [load_network(f'{lp}network_{i}', i)]
        else:
            case.NNFDK.network += [load_network(f'{lp}network_{i}', i)]
        case.NNFDK.do(node_output=True)
        
        save_and_add_artifact(f'{WV_path}_NNFDK{i}_rec.npy',
                              case.NNFDK.node_out_axis)
        print(case.NNFDK.node_out_axis)

#    Q, RT = log_variables(case.NNFDK.results, Q, RT)
#    
#    save_and_add_artifact(WV_path + '_Q.npy', Q)
#    save_and_add_artifact(WV_path + '_RT.npy', RT)
#    t2 = time.time()
    print('Finished NNFDKs. Took:', (t2 - t1) / 60, 'minutes')
    save_table(case, WV_path)

    case = None
    gc.collect()
    return Q

