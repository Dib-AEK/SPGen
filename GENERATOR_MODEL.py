# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:38:36 2022

@author: Abdelkader DIB
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, Lambda, Minimum, Maximum, Concatenate, Lambda, Reshape, Dense
from tensorflow_probability import distributions as tfd
import Functions as F
import os
from tensorflow.keras import Sequential, initializers
from tqdm import tqdm
from Layers import GenEnc, GenPredictionBlock, negloglik, myLoss, GenSamplingLayer

class Generator(Model):
    """ 
        This model generate a speed profile for on link, the inputs are :
              'linkLength','MeanSpeed','in_velocity','min_velocity',
                         'max_velocity','out_velocity'
    """
    
    def __init__(self,inputs,velocity_scaling_factor,network_scaling_factor,units, 
                 name='generator', resolution = 11, l1 = 0, l2=0,train_history=[]):
        super().__init__()      
        self.config_dict = {k: v for k, v in locals().items() if k not in ('self', 'name','__class__')}
        self.__dict__.update({k: v for k, v in locals().items() if k not in ('self', 'name','__class__')})
        self._name = name
        self.fig_name        = name + '_units'+str(units)
        
        # Layers used for states generation
        self.enc_layer    = GenEnc(units= units,  
                                   l1   = l1,
                                   l2   = l2,
                                   name = 'Enc_Layer')
        # layers used in the generator model
        self.pred_layer   = GenPredictionBlock(units = units, 
                                               l1    = l1,
                                               l2    = l2,
                                               name  = 'Gen_Layer') 

    def call(self,inputs):
        x,x_dec = inputs        
        states = self.enc_layer(x)   
        x,mean,_ = self.pred_layer(x_dec,states)  

        return x,mean
    
    def train(self,batch_size, learning_rate, epochs, data_path, val_data_path, 
              extra_data_path,**kwargs):
       
        self.data = F.GenDataset( data_path=data_path, inputs=self.inputs, 
                                  batch_size=batch_size,
                                  NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                                  VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor,
                                  extra_data_path=extra_data_path)
        self.val_data = F.GenDataset( data_path=val_data_path, inputs=self.inputs, 
                                      batch_size=256,
                                      NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                                      VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor)
        #self.dataset = self.data.get_encoder_dataset()
        print("    training ...")

        # Learning rate decay
        callbacks = []
        scheduler = lambda epoch, lr: learning_rate * np.exp(-4 * epoch / epochs) #tf.keras.optimizers.Adam(learning_rate=learning_rate)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

        # Compile the model :
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate)
        lossFunction = [negloglik(), myLoss()] #['negloglik','mse']]

        self.compile(optimizer=optimizer,
                     loss=lossFunction,
                     loss_weights=[1, 10])
        # Fitting the data :        
        self.train_history = self.fit(self.data,
                                      callbacks=callbacks,
                                      epochs=epochs,
                                      validation_data=self.val_data,
                                      max_queue_size=8,
                                      ** kwargs)
        self.config_dict['train_history'] = self.train_history.history
    
    def plot_history(self):
        F.create_folder_if_doesnt_exists("images\generators")
        path = 'images/Generators'
        F.create_folder_if_doesnt_exists(path)
        training_history = self.train_history;
        plt.subplots(figsize=(10, 5))
        plt.plot(training_history.history['loss'], linewidth=2, label='Loss')
        if 'val_loss' in training_history.history.keys():
            plt.plot(training_history.history['val_loss'], linewidth=2, label='Val loss')
        plt.legend(fontsize=16)
        plt.xlabel('Epochs')
        #plt.ylim([-5, 10])
        # plt.yticks([i for i in range(11)])
        plt.grid()
        plt.ylabel('Loss')
        fl = str(np.round(np.mean(self.train_history.history['loss'][-3:]), 2))
        if 'val_loss' in training_history.history.keys():
            fvl = str(np.round(np.mean(self.train_history.history['val_loss'][-3:]), 2))
            plt.title('Loss Over Epochs \n final loss: {} - finale val loss: {}'.format(fl, fvl))
        else:
            plt.title('Loss Over Epochs \n final loss: {}'.format(fl))
        plt.savefig('images\Generators\ ' +self.fig_name + '_loss.jpeg')
 
    def make_inference_model(self):
        ### states generator model
        inputs = tf.keras.Input((len(self.inputs),))
        states = self.enc_layer(inputs)
        self.encoder_model = Model(inputs=inputs,outputs=states)

        ### predictions model
        resolution = self.resolution
        pred_layer = self.pred_layer    
        gen_sampling_layer = GenSamplingLayer(resolution) #x<=(out,k,i)
        relu_layer = ReLU()

        # MODEL
        input_states = [tf.keras.Input((self.units,)) for i in range(self.pred_layer.rnn_states)]
        shifted_input = tf.keras.Input((1,1))
        k = tf.keras.Input((),dtype=tf.int32)
        i = tf.keras.Input((),dtype=tf.int32)

        out,_,out_states =  pred_layer(shifted_input,input_states) 
        out = gen_sampling_layer((out,k,i))
        out = relu_layer(out)

        self.decoder_model = Model((shifted_input,input_states,k,i),(out,out_states))
                        
    def predict_sequence(self,input_seq,k=5):
        
        if (not hasattr(self, "decoder_model")) or (not hasattr(self, "encoder_model")):
            self.make_inference_model()
        
        # Encode the input as state vectors.
        states     = self.encoder_model(input_seq)
        # Generate empty target sequence of length 1.
        output     = tf.zeros((input_seq.shape[0], 1, 1))
        # Sampling loop for a batch of sequences
        out_sequence = []

        for i in tqdm(tf.range(0,self.resolution),desc='Profile Generation'):
            output, states = self.decoder_model((output,states,k,i))
            out_sequence.append(output)
        return tf.concat(out_sequence,1)[:,:,0]*self.velocity_scaling_factor
     
    def get_config(self): 
       # This is overwriting the get_config method in order to get the config of the layer :
       config = super(Generator, self).get_config()
       config.update(self.config_dict)
       return config  
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)   
 


























"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, Lambda, Minimum, Maximum, Concatenate, Lambda, Reshape, Dense
from tensorflow_probability import distributions as tfd
import Functions as F
import os
from tensorflow.keras import Sequential, initializers
from tqdm import tqdm
from Layers import GenEnc, GenPredictionBlock, negloglik, myLoss

class Generator(Model):
    def __init__(self,inputs,velocity_scaling_factor,network_scaling_factor,units, 
                 name='generator', resolution = 11, l1 = 0, l2=0,train_history=[]):
        super().__init__()      
        self.config_dict = {k: v for k, v in locals().items() if k not in ('self', 'name','__class__')}
        self.__dict__.update({k: v for k, v in locals().items() if k not in ('self', 'name','__class__')})
        self._name = name
        self.fig_name        = name + '_units'+str(units)
        
        # Layers used in the encoder model
        self.enc_layer    = GenEnc(units= units,  
                                   l1   = l1,
                                   l2   = l2,
                                   name = 'Enc_Layer')
        # layers used in the generator model
        self.pred_layer   = GenPredictionBlock(units = units, 
                                               l1    = l1,
                                               l2    = l2,
                                               name  = 'Gen_Layer') 
        # self.reshape = Reshape((-1,))
        # self.mean = Dense(1,bias_initializer=initializers.Zeros(),
        #                      kernel_initializer = initializers.Constant(1/11),
        #                      trainable=False)

    def call(self,inputs):
        x,x_dec = inputs        
        states = self.enc_layer(x)   
        x,mean,_ = self.pred_layer(x_dec,states)  
        
        # x_m = self.reshape(x)
        # x_m = self.mean(x_m)
        return x,mean
    
    def train(self,batch_size, learning_rate, epochs, data_path, val_data_path, network_columns,
              extra_data_path,**kwargs):
       
        self.data = F.GenDataset( data_path=data_path, inputs=self.inputs, 
                                  network_columns=network_columns, batch_size=batch_size,
                                  NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                                  VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor,
                                  extra_data_path=extra_data_path)
        self.val_data = F.GenDataset( data_path=val_data_path, inputs=self.inputs, 
                                      network_columns=network_columns, batch_size=256,
                                      NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                                      VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor)
        #self.dataset = self.data.get_encoder_dataset()
        print("    training ...")

        # Learning rate decay
        callbacks = []
        scheduler = lambda epoch, lr: learning_rate * np.exp(-5 * epoch / epochs)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

        # Compile the model :
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        lossFunction = [negloglik(), myLoss()] #['mean_squared_logarithmic_error','mse']]

        self.compile(optimizer=optimizer,
                     loss=lossFunction,
                     loss_weights=[1, 20])
        # Fitting the data :        
        self.train_history = self.fit(self.data,
                                      callbacks=callbacks,
                                      epochs=epochs,
                                      validation_data=self.val_data,
                                      max_queue_size=8,
                                      ** kwargs)
        self.config_dict['train_history'] = self.train_history.history
    
    def plot_history(self):
        F.create_folder_if_doesnt_exists("images\generators")
        path = 'images/Generators'
        F.create_folder_if_doesnt_exists(path)
        training_history = self.train_history;
        plt.subplots(figsize=(10, 5))
        plt.plot(training_history.history['loss'], linewidth=2, label='Loss')
        if 'val_loss' in training_history.history.keys():
            plt.plot(training_history.history['val_loss'], linewidth=2, label='Val loss')
        plt.legend(fontsize=16)
        plt.xlabel('Epochs')
        #plt.ylim([-5, 10])
        # plt.yticks([i for i in range(11)])
        plt.grid()
        plt.ylabel('Loss')
        fl = str(np.round(np.mean(self.train_history.history['loss'][-3:]), 2))
        if 'val_loss' in training_history.history.keys():
            fvl = str(np.round(np.mean(self.train_history.history['val_loss'][-3:]), 2))
            plt.title('Loss Over Epochs \n final loss: {} - finale val loss: {}'.format(fl, fvl))
        else:
            plt.title('Loss Over Epochs \n final loss: {}'.format(fl))
        plt.savefig('images\Generators\ ' +self.fig_name + '_loss.jpeg')
 
    def make_inference_model(self):
        # Encoder model
        inputs = tf.keras.Input((len(self.inputs),))
        states = self.enc_layer(inputs)
        self.encoder_model = Model(inputs=inputs,outputs=states)
                
        # Decoder model  
        class enc(tf.keras.layers.Layer):
            def __init__(self,L,resolution,name='decoder'):
                super().__init__()
                self._name = name
                self.resolution = resolution
                self.L = L
                self.relu = ReLU()
                
            def call(self, shifted_outputs,input_states,how='max',k=5,i=0):
                assert how in ['max','max_from_k','sample'],"How must be max, or sample_from_k or just sample"
                out,m,out_states =  self.L(shifted_outputs,input_states)                    
                if how == 'max':
                    out = m
                else:
                    if how=='max_from_k':
                        ki = k*2 if (i==0 or i==self.resolution-1) else k
                        samples = out.sample(ki)
                        prob    = tf.squeeze(out.log_prob(samples),axis=-1)
                        idx_max_prob = tf.math.argmax(prob,axis=0)
                        samples = tf.transpose(samples,perm=(1,2,3,0))
                        out = tf.gather(samples, indices=idx_max_prob,axis=-1,batch_dims=1)
                        out = tf.squeeze(out,axis=-1)
                    else:
                        pass
                
                out = self.relu(out)
                return out,out_states

        self.decoder_model=  Sequential([enc(L = self.pred_layer,
                                         resolution = self.resolution)])
                        
    def predict_sequence(self,input_seq,how="max",k=5):
        
        if (not hasattr(self, "decoder_model")) or (not hasattr(self, "encoder_model")):
            self.make_inference_model()
        
        # Encode the input as state vectors.
        states     = self.encoder_model(input_seq)
        # Generate empty target sequence of length 1.
        output     = tf.zeros((input_seq.shape[0], 1, 1))
        # Sampling loop for a batch of sequences
        out_sequence = []

        for i in tqdm(tf.range(0,self.resolution),desc='Profile Generation'):
            output, states = self.decoder_model(shifted_outputs = output, 
                                                input_states = states, 
                                                how = how, 
                                                i = i,
                                                k=k)
            out_sequence.append(output)
        return tf.concat(out_sequence,1)[:,:,0]*self.velocity_scaling_factor
    
    def plot_some_profiles(self,n=64,k=5,how='multi',data_path=None,network_columns=None,suffix=''):
        # load data if it doesn't exist
        if (not hasattr(self,'data')):
            self.data = F.GenDataset( data_path=data_path, inputs=self.inputs, 
                         network_columns=network_columns, batch_size=512,
                         NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                         VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor)
        
        X,Y,_ = self.data.__getitem__(0)
        X  = X[0][0:n,:]
        Y  = Y[0][0:n]

        Yh   = self.predict_sequence(X,how = how,k=k)
        path_dir = "images/Generators/"+self.fig_name
        F.create_folder_if_doesnt_exists(path_dir)
        for i in range(n):
            plt.subplots(figsize=(10,5))
            plt.plot(Y[i]*self.velocity_scaling_factor)
            plt.plot(Yh[i])
            plt.grid()
            plt.ylim([0,max(plt.ylim()[1]+10,70)])
            plt.savefig(path_dir+"/"+str(i)+"_"+how+'_'+suffix+".jpeg")
            plt.close()
        colors = ['k','gray','firebrick','sienna','darkkhaki','olivedrab','darkgreen',
                  'darkcyan','navy','darkorchid']
        plt.subplots(figsize=(30,15))
        for i in range(len(colors)):
            plt.plot(Y[i]*self.velocity_scaling_factor,linewidth=2.5,linestyle='-',color=colors[i])
            plt.plot(Yh[i],linewidth=2.5,linestyle='--',color=colors[i])
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Position within the link',fontsize=20)
        plt.ylabel('Velocity [km/h]',fontsize=20)
        plt.grid()
        plt.legend(['GPS profile','Generated profile'],fontsize=20)
        plt.savefig(path_dir+"/presentation_image_profiles.jpeg")
     
    def get_config(self): 
       # This is overwriting the get_config method in order to get the config of the layer :
       config = super(Generator, self).get_config()
       config.update(self.config_dict)
       return config  
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)   
 
"""
