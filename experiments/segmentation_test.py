#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:42:08 2020

@author: lagerwer
"""

import numpy as np
import scipy.ndimage as sp
import scipy.interpolate as si
import time 
import pylab

from skimage.segmentation import flood, watershed
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
# %%
def show(arr):
    if len(arr.shape) == 3:
        mid = np.shape(arr)[0] // 2
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        ax1.imshow(np.rot90(arr[:, :, mid]))
        
        ax2.imshow(np.rot90(arr[:, mid, :]))
        
        ax3.imshow(np.rot90(arr[mid, :, :]))
    else:
        pylab.figure()
        pylab.imshow(np.rot90(arr))
    
def threshold(arr, thr, greater):
    one = np.ones(np.shape(arr))
    if greater:
        if len(thr) == 1:
            return (arr >= thr) * one
        elif len(thr) == 2:
            return ((arr >= thr[0]) * one - (arr >= thr[1])) * arr 
    else: 
        if len(thr) == 1:
            return (arr <= thr) * one
        elif len(thr) == 2:
            return ((arr <= thr[0]) * one - (arr >= thr[1])) * arr         

def find_the_relevant_maxima(loc_max, hist, low_bound=0.1, up_bound=0.7):
    # Check if the maxima are the ones we need
    if hist[1][loc_max[0]] < low_bound:
        # if this is true, the first maxima is the background
        if hist[1][loc_max[-1]] > up_bound:
            # if this is true, the last maximum is not the shell
            return loc_max[1:-1]
        else:
            return loc_max[1:]
    else:
        if hist[1][loc_max[-1]] > up_bound:
            # if this is true, the last maximum is not the shell
            return loc_max[:-1]
        else:
            return loc_max        
    
def compute_volumes(rec, sigma=1, dilation_shell=3, dil_ero=2,
                    give_masks=False):
    mid = rec.shape[0] // 2
    its = int(768 / 100 * dilation_shell)

    rec_filt = gaussian_filter(rec, sigma)

    # Determine the local maxima, the last maximum is the peak corresponding 
    # the shell 
    bins = 10
    binit = 0
    hist = np.histogram(rec_filt, bins=bins)
    
    loc_max = argrelextrema(hist[0], np.greater)[0]
    loc_max = find_the_relevant_maxima(loc_max, hist)
    print(loc_max)
    while len(loc_max) != 2:
        bins += 5
        binit += 1
        hist = np.histogram(rec_filt, bins=bins)
        loc_max = argrelextrema(hist[0], np.greater)[0]
        loc_max = find_the_relevant_maxima(loc_max, hist)
        print(loc_max)
    
    # The maximum relating to the shell is the last one
    # if we needed smaller bins to find it we should go more to the left
    shell_max = loc_max[-1] - binit // 2
    # shell_mask = threshold(rec_filt, [hist[1][shell_max]], True)
    shell_mask = threshold(rec_filt, [hist[1][shell_max]], True)
    # Remove some false positives
    shell_mask = sp.binary_erosion(shell_mask, iterations=dil_ero)
    # Dilate to make sure the full shell is found
    shell_mask = sp.binary_dilation(shell_mask, iterations=its + dil_ero)
    
    kernel_mask = (threshold(rec_filt, [hist[1][3]], True) - shell_mask) > 0
    kernel = ((rec * kernel_mask > hist[1][3]))
    shell = ((rec * shell_mask > hist[1][3]))
    
    # Determine the inner space of the walnut
    markers = np.ones(np.shape(rec))
    markers[1:-1, 1:-1, 1:-1] = 0
    markers[mid, mid, mid] = 2
    t = time.time()
    walnut_in = watershed(shell, markers=markers, mask=(shell==0))
    # print(time.time() - t)
    # Empty space is the kernel minus the innerspace
    emp_spc = (np.invert(kernel_mask) *  (walnut_in==2))

    v_S = np.sum(shell)
    v_ES = np.sum(emp_spc)
    v_K = np.sum(kernel)
    
    # show(shell * rec)
    # show(emp_spc * rec)
    # show(kernel * rec)
    if give_masks:
        return [v_S, v_ES, v_K], shell, emp_spc, kernel
    else:
        return v_S, v_ES, v_K


# %%
pylab.close('all')
volumes = np.zeros((6, 5, 3))
# for i in range(5):
wnr = 16 + 3
exp = wnr - 15
bpath = '/export/scratch2/lagerwer'
GTpath = f'{bpath}/data/FleXray/walnuts_10MAY/walnut_{wnr}/processed_data/ground_truth.npy'
recpath = f'{bpath}/NNFDK_results/full_recon_vols/{exp}/noisy_walnut_{wnr}_'
GT = np.load(GTpath)
# vs_rec, shell, emp_spc, kernel = compute_volumes(GT, give_masks=True)
# # print('GT volumes:', vs_GT)

np.save(f'{recpath}GT_shell.npy', shell)
np.save(f'{recpath}GTemp_spc.npy', emp_spc)
np.save(f'{recpath}GT_kernel.npy', kernel)
meths = ['FDKHN', 'NNFDK4', 'MSD', 'Unet', 'SIRT200'] # 
# for m in meths:
#     rec = np.load(f'{recpath}{m}_full_rec.npy')
#     vs_rec, shell, emp_spc, kernel = compute_volumes(rec, give_masks=True)
#     np.save(f'{recpath}{m}_shell.npy', shell)
#     np.save(f'{recpath}{m}_emp_spc.npy', emp_spc)
#     np.save(f'{recpath}{m}_kernel.npy', kernel)
#     print(m, ' volumes:', vs_rec)

# np.save(f'{bpath}/NNFDK_results/full_recon_vols/segmentation_vols', volumes)
# %%
pylab.close('all')
# vs_MSD, shell_MSD, emp_spc_MSD, kernel_MSD = compute_volumes(np.load(
#     f'{recpath}MSD_full_rec.npy'), give_masks=True)
t = time.time()
sigma = 1
dilation_shell = 3
dil_ero = 2
# rec = np.load(f'{recpath}Unet_full_rec.npy')
rec = GT.copy()
# rec = np.load(f'{recpath}FDKHN_full_rec.npy')
mid = rec.shape[0] // 2
its = int(768 / 100 * dilation_shell)

rec_filt = gaussian_filter(rec, 2)

# Determine the local maxima, the last maximum is the peak corresponding the shell 
bins = 10
binit = 0
hist = np.histogram(rec_filt, bins=bins)

loc_max = argrelextrema(hist[0], np.greater)[0]
loc_max = find_the_relevant_maxima(loc_max, hist)
print(loc_max)
while len(loc_max) != 2:
    bins += 5
    binit += 1
    hist = np.histogram(rec_filt, bins=bins)
    loc_max = argrelextrema(hist[0], np.greater)[0]
    loc_max = find_the_relevant_maxima(loc_max, hist)
    print(loc_max)

# The maximum relating to the shell is the last one
# if we needed smaller bins to find it we should go more to the left
shell_max = loc_max[-1] - binit // 2
# shell_mask = threshold(rec_filt, [hist[1][shell_max]], True)
shell_mask = threshold(rec_filt, [hist[1][shell_max]], True)
# Remove some false positives
shell_mask = sp.binary_erosion(shell_mask, iterations=dil_ero)
# Dilate to make sure the full shell is found
shell_mask = sp.binary_dilation(shell_mask, iterations=its + dil_ero)

kernel_mask = (threshold(rec_filt, [hist[1][3]], True) - shell_mask) > 0
kernel = ((rec * kernel_mask > hist[1][3]))
shell = ((rec * shell_mask > hist[1][3]))


markers = np.ones(np.shape(rec))
markers[1:-1, 1:-1, 1:-1] = 0
# markers[mid- blc:mid+blc, mid-blc:mid+blc, mid-blc:mid+blc] = 2
markers[mid, mid, mid] = 2
t = time.time()
walnut_in = watershed(shell, markers=markers, mask=(shell==0))
print(time.time() - t)
# Empty space is the kernel minus the innerspace
emp_spc = (np.invert(kernel_mask) *  (walnut_in==2))
show(emp_spc)
show(kernel)
# show(kernel_mask)
v_S = np.sum(shell)
v_ES = np.sum(emp_spc)
v_K = np.sum(kernel)
print(v_S, v_ES, v_K)

show(shell * rec)
show(emp_spc * rec)
show(kernel * rec)

# print(time.time() - t)



# GTarr = GT
# # GTarr = GT[:, :, 384]
# # GTarr = GT[:,  384, :]
# # GTarr /= GTarr.max()
# # GTarr = 1 - GTarr
# 
# from skimage.morphology import reconstruction
# dil = 1
# its = int(768 / 100 * 3)
# its2 = 2
# h = (GT.max() - GT.min()) / 8
# t = time.time()
# GT_filt = gaussian_filter(GTarr, dil)

# print('gaussian filt took', time.time() - t, 'seconds')
# # seed = np.copy(GT_filt)
# # seed[1:-1, 1:-1] = GT_filt.min()
# seed = GT_filt - h
# mask = GT_filt
# t = time.time()
# # dilated = reconstruction(seed, mask, method='dilation')
# print('finished first dilation, it took', time.time() -t, 'seconds')
# # %%
# pylab.close('all')
# # show(GT_filt)
# # show(dilated)

# hist = np.histogram(GT_filt, bins=10)
# # Determine the left boundary of the rightmost peak
# # rightbin_boundary = np.max(hist[1][np.argsort(hist[0])[-3:]])

# pylab.figure()
# pylab.plot(hist[1][:-1], hist[0], lw=2)
# show(GT_filt)

# from scipy.signal import argrelextrema
# # the last local maximum is the one we need, but lets go 3 bins to the left of that
# loc_max = argrelextrema(hist[0], np.greater)[0][-1]

# shell = threshold(GT_filt, [hist[1][loc_max]], True)
# # # shell1 = sp.binary_erosion(shell, iterations=5)
# # t = time.time()
# show(shell)
# shell1 = sp.binary_erosion(shell, iterations=its2)
# shell1 = sp.binary_dilation(shell, iterations=its + its2)
# show(shell1)
# # print('binary dilation took', time.time() - t, 'seconds')
# # # show(shell)
# # show(shell)
# # show(shell1)
# kernel = (threshold(GT_filt, [hist[1][3]], True) - shell1) > 0
# kernel1 = (kernel - shell1 > 0)
# kernel2 = sp.binary_dilation(kernel1, iterations=its)
# show(kernel)
# # show(kernel1)
# show(kernel2)

# show(GT * kernel)
# show(GT * shell1)





