#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:51:44 2019

@author: lagerwer
"""


import numpy as np
import ddf_fdk as ddf
import os
import gc
import h5py
import numexpr
import pylab
import time

from . import Network_class as N
from . import TrainingData_class as TDC
from . import NNFDK_astra_backend as NNFDK_astra
from . import support_functions as sup
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
def train_network(nHiddenNodes, full_path, **kwargs):
    if 'retrain' in kwargs:
        retrain = kwargs['retrain']
    else:
        retrain = False
    # Set a path to save the network
    fnNetwork = full_path + '/network_' + str(nHiddenNodes)
    # Check how many TD and VD datasets we have
    nTD = sup.number_of_datasets(full_path, 'TD')
    nVD = sup.number_of_datasets(full_path, 'VD')
    TD_fls = [full_path + 'TD' + str(i) for i in range(nTD)]
    TD_dn = 'TD'
    VD_fls = [full_path + 'VD' + str(i) for i in range(nVD)]
    VD_dn = 'VD'
    # Open hdf5 file for your network
    # Check if we already have a network trained for this number of nodes
    if os.path.exists(fnNetwork + '.hdf5'):
        f = h5py.File(fnNetwork + '.hdf5', 'r+')
        # Check at which network we are
        nNWs = 1
        while str(nNWs) in f:
            nNWs += 1
        if retrain:
            nNetworks = str(nNWs) + '/'
            f.create_group(nNetworks)
        else:
            nn = str(nNWs - 1) + '/'
            # load old network, get them in numpy arrays
            # TODO: Also load the list of training and validation error
            l1 = np.asarray(f[nn + 'l1'])
            l2 = np.asarray(f[nn + 'l2'])
            sc1 = np.asarray(f[nn + 'sc1'])
            sc2 = np.asarray(f[nn + 'sc2'])
            # return old network
            print('Loaded old network, network has', str(nHiddenNodes),
                  'hidden nodes')
            f.close()

            return {'l1' : l1, 'l2' : l2, 'sc1' : sc1, 'sc2' : sc2,
                    'nNodes' : nHiddenNodes, 'nNW' : nNWs - 1}

    # We have no network trained with this number of nodes
    else:
        # Create a hdf5 file for networks with this number of nodes
        f = h5py.File(fnNetwork + '.hdf5', 'w')
        f.attrs['nNodes'] = nHiddenNodes
        nNetworks = str(1) + '/'
        f.create_group(nNetworks)
        nNWs = 1

    # Put everything a the correct classes
    trainData = TDC.MATTrainingData(TD_fls, dataname=TD_dn)
    valData = TDC.MATTrainingData(VD_fls, dataname=VD_dn)
    NW_obj = N.Network(nHiddenNodes, trainData, valData)
    # Train a network
    print('Training new network, network has', str(nHiddenNodes),
      'hidden nodes')
    NW_obj.train()

    # Save the number of datasets used for training/validation
    f[nNetworks].attrs['nTD'] = nTD
    f[nNetworks].attrs['nVD'] = nVD
    # Save the training MSE and validation MSE
    f[nNetworks].attrs['T_MSE'] = NW_obj.trErr
    f[nNetworks].attrs['V_MSE'] = NW_obj.valErr
    # Save the network parameters
    f.create_dataset(nNetworks + 'l1', data=NW_obj.l1)
    f.create_dataset(nNetworks + 'l2', data=NW_obj.l2)
    # Fix the scaling operators
    sc1 = 1 / (NW_obj.minmax[1] - NW_obj.minmax[0])
    sc1 = np.concatenate(([sc1,], [NW_obj.minmax[0] * sc1,]), 0)
    sc2 = np.array([NW_obj.minmax[3] - NW_obj.minmax[2], NW_obj.minmax[2]])
    lst_valError = NW_obj.lst_valError
    lst_traError = NW_obj.lst_traError
    f.create_dataset(nNetworks + 'l_vE', data=lst_valError)
    f.create_dataset(nNetworks + 'l_tE', data=lst_traError)
    # Save the scaling operators
    f.create_dataset(nNetworks + 'sc1', data=sc1)
    f.create_dataset(nNetworks + 'sc2', data=sc2)
    f.close()
    del trainData.normalized, valData.normalized
    gc.collect()
    for it in range(nTD):
        os.remove(trainData.fn[it])
    for iv in range(nVD):
        os.remove(valData.fn[iv])
    return {'l1' : NW_obj.l1, 'l2' : NW_obj.l2, 'sc1' : sc1, 'sc2' : sc2,
            'nNodes' : nHiddenNodes, 'nNW' : nNWs, 'l_vE' : lst_valError,
            'l_tE': lst_traError}

# %%

class NNFDK_class(ddf.algorithm_class.algorithm_class):
    def __init__(self, CT_obj, nTrain, nTD, nVal, nVD, Exp_bin, Exp_op,
                 bin_param):
        self.CT_obj = CT_obj
        self.method = 'NN-FDK'
        self.Exp_bin = Exp_bin
        self.Exp_op = Exp_op
        self.bin_param = bin_param
        self.nTrain = nTrain
        self.nTD = nTD
        self.nVal = nVal
        self.nVD = nVD

    def train(self, nHiddenNodes, **kwargs):
        # Create the load_path containing all specifics
        data_path, full_path = sup.make_map_path(self.CT_obj.pix,
                                                 self.CT_obj.phantom.PH,
                                                 self.CT_obj.angles,
                                                 self.CT_obj.src_rad,
                                                 self.CT_obj.noise,
                                                 self.nTrain, self.nTD,
                                                 self.nVal, self.nVD,
                                                 self.Exp_bin, self.bin_param)
        if 'retrain' in kwargs:
            if hasattr(self, 'network'):
                self.network += [train_network(nHiddenNodes, full_path,
                                                   retrain=kwargs['retrain'])]
            else:
                self.network = [train_network(nHiddenNodes, full_path,
                                                  retrain=kwargs['retrain'])]
        else:
            if hasattr(self, 'network'):
                self.network += [train_network(nHiddenNodes, full_path)]
            else:
                self.network = [train_network(nHiddenNodes, full_path)]

    def do(self, nwNumber=-1, compute_results='yes',
           measures=['MSR', 'MAE', 'SSIM'], backend='ASTRA'):
        t = time.time()
        NW = self.network[nwNumber] # To improve readability
        if backend == 'ASTRA':
            rec, h_e = NNFDK_astra.NNFDK_astra(self.CT_obj.g, NW,
                                               self.CT_obj.geometry,
                                               self.Exp_op)
        else:
            # Take the network requested
            F = self.CT_obj.reco_space.zero()
            # Set a container list for the learned filters
            h_e = []
            for i in range(NW['nNodes']):
                # h_i = self.network['l1'][:-1, i], b_i = self.network['l1'][-1, i]
                h = NW['l1'][:-1, i] * 2 * NW['sc1'][0, :]
                h_e += [h]
                b = NW['l1'][-1, i] + np.sum(NW['l1'][:-1, i]) + 2 * np.dot(
                                NW['l1'][:-1, i], NW['sc1'][1, :])
                # q_i = self.network['l2'][i]
                FDK = self.CT_obj.FDK_bin_nn(h)
                F = hidden_layer(FDK, F, NW['l2'][i], b)
            # Make a numpy array of the filter list
            h_e = np.asarray(h_e)
            # b_o = self.network['l2'][-1]
            rec = outer_layer(F, NW['l2'][-1], NW['sc2'][0], NW['sc2'][1])
        t_rec = time.time() - t
        if compute_results == 'yes':
            self.comp_results(rec, measures, h_e,
                              'HiddenNodes' + str(NW['nNodes']), t_rec)
        else:
            return rec

    def plot_filt(self, h, fontsize=20):
        fig, (ax1, ax2) = pylab.subplots(1, 2, figsize=[15, 6])
        xf = np.asarray(self.CT_obj.fourier_filter_space.grid)
        x = np.asarray(self.CT_obj.filter_space.grid)
        for i in range(self.network['nNodes']):
            hf = np.real(np.asarray(self.CT_obj.pd_FFT(h[i, :])))
            ax1.plot(x, h[i, :])
            ax2.plot(xf, hf)
        ax1.set_title('Filter', fontsize=fontsize)
        ax1.set_ylabel('$h(u)$', fontsize=fontsize)
        ax1.set_xlabel('$u$', fontsize=fontsize)
        ax2.set_title('Fourier transformed filter', fontsize=fontsize)
        ax2.set_ylabel('$\hat{h}(\omega)$', fontsize=fontsize)
        ax2.set_xlabel('$\omega$', fontsize=fontsize)
        fig.show()

    def show_filt(self, nwNumber=-1):
        self.plot_filt(self.results.var[nwNumber])


