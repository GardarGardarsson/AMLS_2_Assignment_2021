#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 20:11:18 2021

@author: gardar
"""
import os
from PIL import Image
import numpy as np
from Modules import user_interface as ui

def loadImages(directory, file_extension='.png', loading_message = 'Loading images...', num_imgs = None):
    
    # List container to hold image arrays
    images = []
    
    # Generate a sorted list of filenames for all images found in directory
    imgFiles = sorted([filename for filename in os.listdir(directory) if file_extension in filename])
    
    # Number of images to load counting from 0
    numImgs  = len(imgFiles) - 1
    
    print('Found {} in directory'.format(len(imgFiles)))
    
    # Load image files from directory
    for i,filename in enumerate(imgFiles):
        
        img = Image.open(directory + filename) # Open image with Pillow function
        images.append(np.asarray(img))         # Append to image list as a numpy array
        
        # Print loading progress on every tenth and last image
        if numImgs != 0 and ( not i % 10 or not i % numImgs ):
            
            # Use print progress function from homemade user-interface module
            ui.print_progress(iteration = i,
                              total     = numImgs,
                              message   = loading_message)
            
        # Stop if number of images to load is reached, num_imgs set to None by default so will load all from dir
        if i == num_imgs:
            break
    
    # Return the loaded images
    return images

# Add a 4th dimension (batch_size x h x w x ch) to images for our model's input layer
def reshapeImgs(img_arr):
    for i,img in enumerate(img_arr):
        h,w,ch = img.shape
        img = np.reshape(img, (1,h,w,ch))
        img_arr[i] = img / 255.0
    return img_arr
    
# Reshape 4D tensor to 3D image object
def reshapeTensor(tensor_arr):
    for i,tensor in enumerate(tensor_arr):
        b,h,w,ch = tensor.shape
        tensor = np.reshape(tensor, (h,w,ch))
        tensor_arr[i] = tensor
    return tensor_arr
