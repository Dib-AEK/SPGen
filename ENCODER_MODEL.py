# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:38:36 2022

@author: Abdelkader DIB
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, Lambda, Minimum, Maximum, Concatenate, Input
from tensorflow_probability import distributions as tfd
import Functions as F
import os
from tensorflow.keras import Sequential
from tqdm import tqdm
from Layers import (EncEnc, EncPredictionBlock, negloglik, MinMaxConstraints, EncSamplingLayer)


class Encoder(Model):
    """ This is an RNN model capturing the distribution of minimum, maximum, and exit
    speed for a sequence of links that forms a trip"""

    def __init__(self, inputs, outputs, units, l1, l2, velocity_scaling_factor,
                 network_scaling_factor, name='First_stage'):

        super(Encoder, self).__init__()        
        self.config_dict = {k: v for k, v in locals().items() if k not in ('self', 'name','__class__')}
        self.__dict__.update({k: v for k, v in locals().items() if k not in ('self', 'name','__class__')})
        self._name = name
        self.fig_name = name + '_units' + str(units)
        #################################
        # Define layers:
        #################################
        self.enc_layer = EncEnc(preprocessing_units=units,
                                l1=l1,
                                l2=l2,
                                name=name + '_bilstm_layer',
                                _dtype='float32')
        self.pred_layer = EncPredictionBlock(units=units,
                                             l1=l1,
                                             l2=l2,
                                             outputs=outputs,
                                             name=name + '_prediction_layer',
                                             _dtype='float32')

    def call(self, inputs):
        x, shifted_y = inputs
        # bi-lstm
        x = self.enc_layer(x)
        # Predictions
        x, m, _ = self.pred_layer(x, shifted_y)
        return x, m

     def train(self, batch_size, learning_rate, epochs, data_path, validation_data_path, network_columns, max_length, 
                    sample_weights_on, sample_weight, parallel = False, **kwargs):

        self.data = F.EncDataset(data_path=data_path, input_columns=self.inputs, output_columns=self.outputs,
                               network_columns=network_columns, NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                               VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor, batch_size=batch_size,
                               training=True, max_length=max_length, dtype='float32',parallel=parallel,
                               sample_weights_on=sample_weights_on,sample_weight=sample_weight)
        self.val_data = F.EncDataset(data_path=validation_data_path, input_columns=self.inputs, output_columns=self.outputs,
                           network_columns=network_columns, NETWORK_SCALING_FACTOR=self.network_scaling_factor,
                           VELOCITY_SCALING_FACTOR=self.velocity_scaling_factor, batch_size=128,
                           training=True, max_length=max_length, dtype='float32',parallel=parallel,
                           sample_weights_on=sample_weights_on,sample_weight=sample_weight)
        
        print("    training ...")
        # Learning rate decay
        callbacks = []
        scheduler = lambda epoch, lr: learning_rate * np.exp(-4 * epoch / epochs)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

        # Compile the model :
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate) #tf.keras.optimizers.Adam(learning_rate=learning_rate)
        lossFunction = [negloglik(), 'mean_squared_logarithmic_error']

        self.compile(optimizer=optimizer,
                     loss=lossFunction,
                     loss_weights=[1, 10])
        # Fitting the data :        
        self.train_history = self.fit(self.data,
                                      validation_data = self.val_data,
                                      callbacks=callbacks,
                                      epochs=epochs,
                                      ** kwargs)

    def plot_history(self):
        path = 'images/Encoder_prob'
        F.create_folder_if_doesnt_exists(path)
        training_history = self.train_history;
        plt.subplots(figsize=(10, 5))
        plt.plot(training_history.history['loss'], linewidth=2, label='Loss')
        if 'val_loss' in training_history.history.keys():
            plt.plot(training_history.history['val_loss'], linewidth=2, label='Val loss')
        plt.legend(fontsize=16)
        plt.xlabel('Epochs')
        plt.grid()
        plt.ylabel('Loss')
        fl = str(np.round(np.mean(self.train_history.history['loss'][-3:]), 2))
        if 'val_loss' in training_history.history.keys():
            fvl = str(np.round(np.mean(self.train_history.history['val_loss'][-3:]), 2))
            plt.title('Loss Over Epochs \n final loss: {} - finale val loss: {}'.format(fl, fvl))
        else:
            plt.title('Loss Over Epochs \n final loss: {}'.format(fl))
        plt.savefig(os.path.join(path, self.fig_name + '.jpeg'))

    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(Encoder, self).get_config()
        config.update(self.config_dict)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def make_inference_model(self, return_it=False):
        ### bi_lstm model
        inputs = Input((None,self.layers[0].weights[0].shape[0]))
        outputs = self.enc_layer(inputs)
        self.encoder_model = Model(inputs,outputs)
        ### predictions model
        outputs = self.outputs
        pred_layer = self.pred_layer
        sampling_layer = EncSamplingLayer()
        constraint_layer = MinMaxConstraints(outputs)
        relu_layer = ReLU()
        #Model
        enc_sqs_inputs = tf.keras.Input((1,self.units),dtype=tf.float32)
        shifted_inputs = tf.keras.Input((1,len(outputs)),dtype=tf.float32)
        states_inputs  = [tf.keras.Input((self.units,),dtype=tf.float32) for i in range(self.pred_layer.rnn_states)]
        k = tf.keras.Input((),dtype=tf.int32)
        inputs = (enc_sqs_inputs,shifted_inputs,states_inputs,k)

        out, _ , out_states = pred_layer(enc_sqs_inputs, shifted_inputs, states=states_inputs)
        out = sampling_layer((out,k))
        out = relu_layer(out)
        out = constraint_layer(out)        
        self.decoder_model = Model((enc_sqs_inputs,shifted_inputs,states_inputs,k),(out,out_states))

    def initialize_states_and_outputs(self, in_velocity, batch_size):
        assert type(in_velocity)==list, "Input velocity should be a list"
        # Generate the first sequence element :
        output_shape = len(self.outputs)
        states_len = self.units
        if len(in_velocity) > 1:
            io = self.outputs.index('out_velocity')
            i1 = output_shape- io - 1
            output = tf.concat([tf.zeros([batch_size, 1, io]),
                                in_velocity,
                                tf.zeros([batch_size, 1, i1])],
                               axis=-1)
        else:
            output = tf.zeros((batch_size, 1, output_shape)) * in_velocity[0]
        # Generate initial states :
        states = self.pred_layer.get_zeros_states(batch_size)
        return states, output

    def predict_sequence(self, input_seq, in_velocity=[0], k=3):
        if (not hasattr(self, 'decoder_model')) or (not hasattr(self, 'encoder_model')):
            self.make_inference_model()

        # Pass the bi_lstm the sequence
        x = self.encoder_model(input_seq)

        # initilize states and outputs:
        states, output = self.initialize_states_and_outputs(in_velocity, input_seq.shape[0])

        # Pass the lstm layer throught "x" to predict outputs
        output_sequence = tf.constant(output)
        for i in tqdm(range(x.shape[1]), desc='      Inference '):
            output, states = self.decoder_model((tf.gather(x, axis=1, indices=[i]),output,states,k))
            # add the result to the output_sequence            
            output_sequence = tf.concat([output_sequence, output], axis=1)

        return output_sequence[:, 1:, :]


