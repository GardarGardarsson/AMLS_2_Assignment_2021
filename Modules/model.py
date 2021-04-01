#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:48:58 2021

@author: gardar
"""

from Modules import metrics

import os
import pickle
import numpy as np
import pandas as pd
from time import time

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, PReLu
from tensorflow.keras.optimizers import Adam

class SRCNN:
    """
    Super-Resolution Convolutional Neural Network class
    """
    def __init__(self,
                 conv_layers,   conv_filters,   conv_kernel_sizes,   conv_strides, 
                 deconv_layers, deconv_filters, deconv_kernel_sizes, deconv_strides):
        
        # Convolutional configuration
        self.c_layers  = conv_layers
        self.c_filters = conv_filters
        self.c_kernels = conv_kernel_sizes
        self.c_strides = conv_strides
        
        # Deconvolutional configuration
        self.d_layers  = deconv_layers
        self.d_filters = deconv_filters
        self.d_kernels = deconv_kernel_sizes
        self.d_strides = deconv_strides
    
        # Initialise model instance
        self.model = Sequential()
        
        # Build model according to configuration - Convolutional layers
        for l,f,k,s in zip(range(self.c_layers), self.c_filters, self.c_kernels, self.c_strides):
            # Layer zero differs from the rest as it has a defined input_shape parameter
            if not l:
                self.model.add(Conv2D(filters=f, kernel_size=k, strides=s, padding='same', activation=None, use_bias = True, kernel_initializer = 'he_normal', input_shape=(None,None,3)))
                self.model.add(PReLu())
            else:
                self.model.add(Conv2D(filters=f, kernel_size=k, strides=s, padding='same', activation=None, use_bias = True, kernel_initializer = 'he_normal'))
                self.model.add(PReLu())
    
        # Build model according to configuration - Deconvolutional layers
        for l,f,k,s in zip(range(self.d_layers), self.d_filters, self.d_kernels, self.d_strides):
            self.model.add(Conv2DTranspose(filters=f, kernel_size=k, strides=s, padding='same', activation=None))
        
        # Update model name from configuration
        self.model._name = self.nameModel()
        
        # Compile model...
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mse',metrics=[metrics.psnr,metrics.ssim])
        # ... and build
        self.model.build()
        
        # Training history
        self.train_history = pd.DataFrame()
        self.val_history   = pd.DataFrame()
                
        # Initialise variable for best model path and validation loss records
        self.best_model_path   = './Models/SRCNN/' + self.model.name + '/'
        self.losses_location   = 'losses/'
        self.losses_filename   = 'val_loss.pkl'
        self.loc_val_history   = 'val_history.pkl'
        self.loc_train_history = 'train_history.pkl'
        self.tensorboard_log   = './logs/' + self.model.name + '/'
    
    # Automatically generate name for model from configuration
    def nameModel(self):
        c_filterStr=''
        for f in self.c_filters:
            c_filterStr+='-'+str(f)
        
        c_kernelStr=''
        
        for k in self.c_kernels:
            c_kernelStr+='-'+str(k[0])+str(k[1])
        
        c_strideStr=''
        
        for s in self.c_strides:
            c_strideStr+='-'+str(s[0])+str(s[1])
            
        d_filterStr=''
        for f in self.d_filters:
            d_filterStr+='-'+str(f)
        
        d_kernelStr=''
        
        for k in self.d_kernels:
            d_kernelStr+='-'+str(k[0])+str(k[1])
        
        d_strideStr=''
        
        for s in self.d_strides:
            d_strideStr+='-'+str(s[0])+str(s[1])
        
        name = 'Conv-{clayers}_Flt{cfilters}_Krnl{ckernels}_Strd{cstrides}-Deconv-{dlayers}_Flt{dfilters}_Krnl{dkernels}_Strd{dstrides}'.format(clayers=self.c_layers,
                                                                             cfilters=c_filterStr,
                                                                             ckernels=c_kernelStr,
                                                                             cstrides=c_strideStr,
                                                                             dlayers=self.d_layers,
                                                                             dfilters=d_filterStr,
                                                                             dkernels=d_kernelStr,
                                                                             dstrides=d_strideStr)
        return name
    
    def train(self, total_epochs, batch_size, datahandler, validation_set):

        """
        I N I T I A L I S A T I O N   F O R   T R A I N I N G
        """
        # Training settings
        steps_per_epoch = np.floor(800 / batch_size).astype(int)
        
        # Get validation variables from set
        x_val = validation_set['lr'].astype('float32')
        y_val = validation_set['hr'].astype('float32')
        
        # Create the directories if they don't exist
        if not os.path.isdir(self.best_model_path):
            os.makedirs(self.best_model_path)
            
        if not os.path.isdir(self.best_model_path + self.losses_location):
            os.makedirs(self.best_model_path + self.losses_location)  
            
        # If a validation record exists we load it
        if os.path.exists(self.best_model_path + self.losses_location + self.losses_filename):
            with open(self.best_model_path + self.losses_location + self.losses_filename, 'rb') as file:
                validation_losses     = pickle.load(file)
                best_validation_loss  = validation_losses['val_loss']
                
        # Else we start from up high
        else:
            best_validation_loss = 1e6
            
        """
        L O G G E R
        """
        # Print statistics about the training procedure, let user know something is happening before first epoch finishes
        print('-'*60+'\n'+'{msg:^60s}'.format(msg="... Initiating Training Session ...") + '\n' + '-'*60)
        
        for epoch in range(total_epochs):
            
            # Define a TensorBoard for visualisation of the training process
            tensorboard = TensorBoard(log_dir=self.tensorboard_log,
                                      histogram_freq=1, 
                                      write_graph = True)
            
            # Hook TensorBoard up with our SRCNN model
            tensorboard.set_model(self.model)
            
            # Start stopwatch
            epoch_start = time()
            
            for step in range(steps_per_epoch):
                
                """
                T R A I N
                """
                # Get a new training batch, n number of image patches set by batch_size
                batch = datahandler.get_batch(batch_size = batch_size, flatness = .1)
                
                # Extract training samples and labels 
                x_train = batch['lr'].astype('float32')
                y_train = batch['hr'].astype('float32')
                
                # Train on batch
                training_losses = self.model.train_on_batch(x = x_train,
                                                            y = y_train)
                
                # Format training losses with descriptive keys for TensorBoard
                training_losses = {'train_'+str(key):val for (key,val) in zip(self.model.metrics_names,training_losses)}
                
                # Keep history
                self.train_history = self.train_history.append(training_losses, ignore_index=True)
            
            """ 
            V A L I D A T E
            """
            # Get validation losses
            validation_losses = self.model.evaluate(x_val, 
                                                    y_val, 
                                                    batch_size=batch_size,
                                                    verbose=0)
            
            # Format validation losses with descriptive key for TensorBoard
            validation_losses = {'val_'+str(key):val for (key,val) in zip(self.model.metrics_names,validation_losses)}
            
            # Keep history
            self.val_history = self.val_history.append(validation_losses, ignore_index=True)

            """
            U P D A T E   T E N S O R B O A R D
            """           
            # Push validation losses to TensorBoard logs
            tensorboard.on_epoch_end(epoch, validation_losses)
            
            # Stop stopwatch
            elapsed_time = time() - epoch_start
            
            """
            L O G G E R
            """
            # Print statistics about the training procedure
            print('-'*60+'\n'+'{msg:^60s}'.format(msg="E P O C H   {e}/{te}").format(e=epoch+1, te=total_epochs) + '\n' + '-'*60)
            msg='Runtime'
            print(f' {msg:25} ==> {elapsed_time:10}s')
            for metric,score in training_losses.items():
                print(f' {metric:25} ==> {score:10}')
            for metric,score in validation_losses.items():
                print(f' {metric:25} ==> {score:10}')
            
                        
            """
            S A V E   B E S T   M O D E L S   -   S T O R E   V A L I D A T I O N   L O S S E S
            """
            # If current validation loss is lower than the best known one
            if validation_losses['val_loss'] < best_validation_loss:
                # Update best validation loss
                best_validation_loss = validation_losses['val_loss'] 
                # Save the model - I might want to save the weights as well, it might be useful to initialise the x3 / x4 models on the x2 model
                self.saveWeights()      
                print('*'*60)                   
                print("Best model weights w/ val loss {l:.6f} saved to {p}".format(l=validation_losses['val_loss'],
                                                                                   p=self.best_model_path))
                print('*'*60)     
        
                with open(self.best_model_path + self.losses_location + self.losses_filename, 'wb') as file:
                    pickle.dump(validation_losses, file, protocol=pickle.HIGHEST_PROTOCOL)
               
            
        # Close TensorBoard
        tensorboard.on_train_end(None)
        
        # Store losses history
        self.storeHistory()
    
    # Method to load model weights and history
    def loadWeights(self):
        self.model.load_weights(self.best_model_path)
        self.loadHistory()
        
    # Method to save model weights
    def saveWeights(self):
        self.model.save_weights(self.best_model_path)
        print('Model weights saved to: {}'.format(self.best_model_path))
    
    # Store training records dataframe
    def storeHistory(self):
        path = self.best_model_path+self.losses_location+self.loc_train_history
        self.train_history.to_pickle(path)
        
        path = self.best_model_path+self.losses_location+self.loc_val_history
        self.val_history.to_pickle(path)
    
    # Load training records dataframe
    def loadHistory(self): 
        path = self.best_model_path+self.losses_location+self.loc_train_history
        self.train_history = pd.read_pickle(path)
        
        path = self.best_model_path+self.losses_location+self.loc_val_history
        self.val_history   = pd.read_pickle(path)
