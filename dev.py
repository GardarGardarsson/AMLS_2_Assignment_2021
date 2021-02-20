#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:01:40 2021

@author: gardar
"""

import Modules.model as model
from ISR.utils import datahandler as ISR_dh

if __name__ == '__main__':

    # %% Instantiate datahandler...
    """
    I N S T A N T I A T E   D A T A H A N D L E R
    """
    # Define parameters for the training task, let's use 48x48 patches for the x2 downscaled images
    lr_patch_size = 48
    scale         = 2
    batch_size    = 4
    
    # We now create a datahandler for the training, and point it to the location of the LR and HR images
    datahandler = ISR_dh.DataHandler(lr_dir = './Datasets/DIV2K_train_LR_bicubic/X2/',
                                     hr_dir = './Datasets/DIV2K_train_HR/',
                                     patch_size = lr_patch_size, 
                                     scale = scale,
                                     n_validation_samples = 100)
    
    # Generate a validation set
    validation_set = datahandler.get_validation_set(batch_size = batch_size)
    
    # %% Build model...
    """
    B U I L D   M O D E L 
    """
    srcnn = model.SRCNN(conv_layers = 5, 
                        conv_filters = [32, 64, 64, 64, 32],
                        conv_kernel_sizes = [(3,3),(3,3),(3,3),(3,3),(3,3)],
                        conv_strides = [(1,1),(1,1),(1,1),(1,1),(1,1)],
                        deconv_layers = 1, 
                        deconv_filters = [3],
                        deconv_kernel_sizes = [(2,2)],
                        deconv_strides = [(2,2)])
    
    srcnn.model.summary()
    
     # %% Train model... 
     """
     T R A I N   M O D E L 
     """
     srcnn.train(total_epochs = 10,
                 batch_size = batch_size,
                 datahandler = datahandler, 
                 validation_set = validation_set)  
     # %% 
     print(srcnn.val_history)