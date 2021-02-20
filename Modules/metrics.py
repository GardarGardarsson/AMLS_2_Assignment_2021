#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S U P E R   R E S O L U T I O N   M E T R I C S
------------------------------------------------------------------------------
This module holds functions necessary for evaluating the performance of super-
resolution machine learning models. The two key metrics for evaluation are
Peak Signal to Noise Ratio (PSNR) and the Structural Similarity (SSIM) index.
PSNR is closely related to the Mean Squared Error (MSE) of a prediction and
it's true value. While PSNR is valid for describing pixelwise differences, it
does not capture the human perception of similarity, which explains the 
necessity for the alternate SSIM metric. 

This module will define two types of evaluation functions: 
    1. Numpy-based functions (lower-caps naming convention)
    2. Tensor-based functions (upper-caps naming convention)
    
Numpy based functions are used for evaluation after training the model, i.e.
of the images the models output, while the Tensor functions are used as 
metrics of the models, to be monitored during training.

Created on Fri Feb 19 21:25:51 2021

@author: gardar
"""

import numpy as np
from skimage.metrics import structural_similarity as SSIM
import tensorflow as tf

# Mean Squared Error
def MSE(target, reference):
    """
    Calculate MSE of two images
    
    Parameters:
    -----------
    target = Predicted RGB image
    reference = Original RGB image
    
    Returns:
    --------
    Mean Squared Error of Target and Reference
    """
    return np.mean(np.square(target-reference))

# Peak Signal to Noise Ratio
def PSNR(target, reference, max_px_val=1.0):
    """
    Calculate PSNR of two images
    
    Parameters:
    -----------
    target = Predicted RGB image
    reference = Original RGB image
    max_px_val = Maximum image pixel value, e.g. 255 or 1.0
    
    Return:
    -------
    Peak Signal to Noise Ratio in dB
    """
    return 20*np.log10(max_px_val / np.sqrt(MSE(target,reference)))

# Function that returns a dictionary of the metrics
def evaluate(target, reference):
    """
    Evaluate two images with regards to MSE, PSNR and SSIM
    
    Parameters:
    -----------
    target = Predicted RGB image
    reference = Original RGB image
    
    Returns:
    --------
    Dictionary of metrics, MSE, PSNR and SSIM
    """
    # Make and populate dictionary
    metrics = {'MSE' : MSE(target,reference),
               'PSNR': PSNR(target,reference),
               'SSIM': SSIM(target,reference,multichannel=True)}
    
    return metrics

# Tensor based MSE 
def mse(y_pred, y_true):
    """
    Calculate MSE of two tensor images
    
    Parameters:
    -----------
    y_pred = Predicted RGB image
    y_true = Original RGB image
    
    Returns:
    --------
    Mean Squared Error of Target and Reference
    """
    return tf.keras.losses.MSE(y_true, y_pred)

# Tensor based PSNR
def psnr(y_pred, y_true, max_px_val=1.0):
    """
    Calculate PSNR of two tensor images
    
    Parameters:
    -----------
    y_pred = Predicted RGB image
    y_true = Original RGB image
    max_px_val = Maximum image pixel value, e.g. 255 or 1.0
    
    Return:
    -------
    Peak Signal to Noise Ratio in dB
    """
    return tf.image.psnr(y_pred,
                         y_true,
                         max_val=max_px_val)
# Tensor based SSIM
def ssim(y_pred, y_true):
    """
    Calculate SSIM of two tensor images
    
    Parameters:
    -----------
    target = Predicted RGB image
    reference = Original RGB image
    
    Return:
    -------
    Peak Signal to Noise Ratio in dB
    """
    return tf.image.ssim_multiscale(img1=y_pred,
                                    img2=y_true,
                                    max_val=1.0,
                                    filter_size=4,
                                    filter_sigma=1.5,
                                    k1=0.01,
                                    k2=0.03)
