#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:32:38 2018

@author: lagerwer
"""


import numpy as np
from six.moves import range
from scipy.signal import fftconvolve
import scipy.ndimage.filters as snf
import numpy.linalg as na
import scipy.io as sio


import numexpr
import time
import scipy.sparse as ss
import scipy.linalg as la
try:
    import scipy.linalg.fblas as fblas
    hasfblas=True
except:
    hasfblas=False



def sigmoid(x):
    '''Sigmoid function'''
    return numexpr.evaluate("1./(1.+exp(-x))")

class Network(object):
    '''
    The neural network object that performs all training and reconstruction.
    :param nHiddenNodes: The number of hidden nodes in the network.
    :type nHiddenNodes: :class:`int`
    :param projector: The projector to use.
    :type projector: A ``Projector`` object (see, for example: :mod:`nnfbp.SimpleCPUProjector`)
    :param trainData: The training data set.
    :type trainData: A ``DataSet`` object (see: :mod:`nnfbp.DataSet`)
    :param valData: The validation data set.
    :type valData: A ``DataSet`` object (see: :mod:`nnfbp.DataSet`)
    :param reductor: Optional reductor to use.
    :type reductor: A ``Reductor`` object (see: :mod:`nnfbp.Reductors`, default:``LogSymReductor``)
    :param nTrain: Number of pixels to pick out of training set.
    :type nTrain: :class:`int`
    :param nVal: Number of pixels to pick out of validation set.
    :type nVal: :class:`int`
    :param tmpDir: Optional temporary directory to use.
    :type tmpDir: :class:`string`
    :param createEmptyClass: Used internally when loading from disk, to create an empty object. Do not use directly.
    :type createEmptyClass: :class:`boolean`
    '''

    def __init__(self, nHiddenNodes, trainData, valData, setinit=None):
        self.tTD = trainData
        self.vTD = valData
        self.nHid = nHiddenNodes
        self.nIn = self.tTD.getDataBlock(0).shape[1]-1
        self.jacDiff = np.zeros((self.nHid) * (self.nIn + 1) + self.nHid + 1)
        self.jac2 = np.zeros(((self.nHid) * (self.nIn+1) + self.nHid + 1,
                              (self.nHid) * (self.nIn+1) + self.nHid + 1))
        self.setinit = setinit

    def __inittrain(self):
        '''Initialize training parameters, create actual training and validation
        sets by picking random pixels from the datasets'''
        self.l1 = 2 * np.random.rand(self.nIn+1, self.nHid) - 1
        if self.setinit is not None:
            self.l1.fill(0)
            nd = self.nIn/self.setinit[0]
            for i, j in enumerate(self.setinit[1]):
                self.l1[j * nd:(j + 1) * nd, i] = 2 * np.random.rand(nd) - 1
                self.l1[-1, i] = 2 * np.random.rand(1) - 1
        beta = 0.7 * self.nHid ** (1. / (self.nIn))
        l1norm = np.linalg.norm(self.l1)
        self.l1 *= beta / l1norm
        self.l2 = 2 * np.random.rand(self.nHid + 1) - 1
        self.l2 /= np.linalg.norm(self.l2)
        self.minl1 = self.l1.copy()
        self.minl2 = self.l2.copy()
        self.minmax = self.tTD.getMinMax()
        self.tTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2],
                               self.minmax[3])
        self.vTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2],
                               self.minmax[3])
        self.ident = np.eye((self.nHid) * (self.nIn + 1) + self.nHid + 1)


    def __processDataBlock(self, data_in, return_data=False):
        ''' Returns output values (``vals``), 'correct' output values (``valOut``) and
        hidden node output values (``hiddenOut``) from a block of data.'''
        data = data_in[:, :-1]
        valOut = data_in[:, -1].copy()
        hiddenOut = np.empty((data.shape[0], self.l1.shape[1] + 1))
        hiddenOut[:, 0:self.l1.shape[1]] = sigmoid(np.dot(data, self.l1[:-1, :])
                                            - self.l1[-1,:])
        hiddenOut[:, -1] = -1
        rawVals = np.dot(hiddenOut, self.l2)
        vals = sigmoid(rawVals)
        if return_data:
            data_in = data_in.copy()
            data_in[:, -1] = -np.ones(data_in.shape[0])
            return vals, valOut, hiddenOut, data_in
        else:
            return vals, valOut, hiddenOut


    def __getTSE(self, dat):
        '''Returns the total squared error of a data block'''
        tse = 0.
        size = 0.
        for i in range(dat.nBlocks):
            data = dat.getDataBlock(i)
            vals, valOut, hiddenOut = self.__processDataBlock(data)
            size += vals.size
            tse += numexpr.evaluate('sum((vals - valOut)**2)')
        return tse / size

    def __setJac2(self):
        '''Calculates :math:`J^T J` and :math:`J^T e` for the training data.
        Used for Levenberg-Marquardt method.'''
        self.jac2.fill(0)
        self.jacDiff.fill(0)
        for i in range(self.tTD.nBlocks):

            data = self.tTD.getDataBlock(i)

            vals, valOut, hiddenOut, data = self.__processDataBlock(data,
                                                            return_data=True)

            diffs = numexpr.evaluate('valOut - vals')
            d0 = numexpr.evaluate('-vals * (1 - vals)')
            ot = (np.outer(d0, self.l2))
            dj = numexpr.evaluate('hiddenOut * (1 - hiddenOut) * ot')

            jac = np.empty((data.shape[0], (self.nHid) * (self.nIn+1) +
                            self.nHid + 1))

            I = np.tile(np.arange(data.shape[0]), (self.nHid + 1,
                        1)).flatten('F')
            J = np.arange(data.shape[0] * (self.nHid + 1))

            Q = ss.csc_matrix((dj.flatten(), np.vstack((J, I))),
                              (data.shape[0] * (self.nHid + 1), data.shape[0]))


            jac[:, 0:self.nHid + 1] = ss.spdiags(d0, 0, data.shape[0],
                                               data.shape[0]).dot(hiddenOut)

            # Kan dit sneller door MKL direct aan te roepen?
            Q2 = np.reshape(Q.dot(data), (data.shape[0], (self.nIn + 1) *
                                                              (self.nHid + 1)))

            jac[:, self.nHid + 1:jac.shape[1]] = Q2[:, 0:Q2.shape[1] -
                                                    (self.nIn+1)]

            if hasfblas:
                self.jac2 += fblas.dgemm(1.0, a=jac.T, b=jac.T, trans_b=True)
                self.jacDiff += fblas.dgemv(1.0, a=jac.T, x=diffs)
            else:
                self.jac2 += np.dot(jac.T,jac)
                self.jacDiff += np.dot(jac.T,diffs)



    def train(self):
        '''Train the network using the Levenberg-Marquardt method.'''
        self.__inittrain()
        mu = 100000.
        muUpdate = 10
        self.lst_valError = []
        self.lst_traError = []
        prevValError = np.Inf
        bestCounter = 0
        tse = self.__getTSE(self.tTD)
        self.lst_traError += [tse]
        curTime = time.time()
        self.allls = []
        for i in range(1000000):
            self.__setJac2()
            try:
                dw = -la.cho_solve(la.cho_factor(self.jac2 + mu * self.ident),
                                   self.jacDiff)
            except la.LinAlgError:
                break
            done = -1
            while done <= 0:
                self.l2 += dw[0:self.nHid + 1]
                for k in range(self.nHid):
                    start = self.nHid + 1 + k * (self.nIn+1)
                    if self.setinit is not None:
                        nd = self.nIn/self.setinit[0]
                        j = self.setinit[1][k]
                        self.l1[j * nd: (j + 1) * nd, k] += dw[start +j * nd:
                                                            start + (j + 1) * nd]
                        self.l1[-1, k] += dw[start + self.nIn]
                    else:
                        self.l1[:, k] += dw[start:start + self.nIn + 1]
                newtse = self.__getTSE(self.tTD)

                if newtse < tse:
                    if done == -1:
                        mu /= muUpdate
                    if mu <= 1e-100:
                        mu = 1e-99
                    done = 1
                else:
                    done = 0
                    mu *= muUpdate
                    if mu >= 1e20:
                        done = 2
                        break
                    self.l2 -= dw[0:self.nHid + 1]
                    for k in range(self.nHid):
                        start = self.nHid + 1 + k * (self.nIn+1)
                        if self.setinit is not None:
                            nd = self.nIn / self.setinit[0]
                            j = self.setinit[1][k]
                            self.l1[j * nd:(j + 1) * nd, k] -= dw[start + j * nd:
                                                            start + (j + 1) * nd]
                            self.l1[-1, k] -= dw[start + self.nIn]
                        else:
                            self.l1[:, k] -= dw[start:start + self.nIn+1]
                    try:
                        dw = -la.cho_solve(la.cho_factor(self.jac2 + mu *
                                                         self.ident),
                                                            self.jacDiff)
                    except la.LinAlgError:
                        done = 2
            gradSize = np.linalg.norm(self.jacDiff)
            if done == 2:
                break
            curValErr = self.__getTSE(self.vTD)
            if curValErr > prevValError:
                bestCounter += 1
            else:
                prevValError = curValErr
                self.lst_valError += [prevValError]
                self.minl1 = self.l1.copy()
                self.minl2 = self.l2.copy()
                if (newtse / tse < 0.999):
                    bestCounter = 0
                else:
                    bestCounter += 1
            if bestCounter == 25:
                print('25 times no improvement')
                break
            if(gradSize < 1e-8):
                print('Gradsize was too small')
                break
            tse = newtse
            self.lst_traError += [tse]
            if i % 1 == 0:
                print('Validation set error: {}'.format(prevValError))
            self.allls.append([self.minl1, self.minl2])
        self.l1 = self.minl1
        self.l2 = self.minl2
        self.lst_valError = np.array(self.lst_valError)
        self.lst_traError = np.array(self.lst_traError)
        self.valErr = prevValError
        self.trErr = tse



    def saveAllToDisk(self, fn):
        for i, k in enumerate(self.allls):
            sio.savemat(fn + "{}.mat".format(i), {'l1': k[0], 'l2': k[1],
                        'minmax': self.minmax}, do_compression=True)

    def saveToDisk(self, fn):
        '''Save a trained network to disk, so that it can be used later
        without retraining.
        :param fn: Filename to save it to.
        :type fn: :class:`string`
        '''
        sio.savemat(fn, {'l1': self.l1, 'l2': self.l2,
                        'minmax': self.minmax}, do_compression=True)


