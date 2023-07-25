# -*- coding: utf-8 -*-
"""
Created on Wed 23/02/2023

@author: Abdelkader DIB
"""

from ENCODER_MODEL import Encoder
import Functions as F
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from Layers import negloglik

# Define data path
data_path = os.path.join(os.path.dirname(os.getcwd()),'routes_filtering','training_data.parquet')
validation_data_path = os.path.join(os.path.dirname(os.getcwd()),'routes_filtering','validation_data.parquet')

network_columns = ['averageSpeed', 'freeFlowSpeed','speed_limit','linkLength', 'slope', 'shape_parameter', 'cos_enter','sin_enter', 'cos_exit',
                   'sin_exit', 'FC_1', 'FC_2', 'FC_3', 'FC_4', 'FC_5', 'TRAFFIC_SIGNAL', 'STOP', 'YIELD', 'PEDESTRIAN_CROSSING']
# Hyperparameters
UNITS = 100
L1 = 5e-4
L2 = 2e-4
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
EPOCHS = 200
VELOCITY_SCALING_FACTOR = 40
NETWORK_SCALING_FACTOR   =  {'averageSpeed':40, 'freeFlowSpeed':40,'speed_limit':40,'linkLength':100, 'shape_parameter':0.5, 'cos_enter':1,
                            'sin_enter':1, 'cos_exit':1,'sin_exit':1, 'FC_1':1, 'FC_2':1, 'FC_3':1, 'FC_4':1, 'FC_5':1, 'TRAFFIC_SIGNAL':1,
                             'STOP':1,'MeanSpeed':40}
ENCODER_NAME   = "ENCODER"
MODEL_INPUTS   =  ['speed_limit','linkLength', 'cos_enter','sin_enter', 'cos_exit',
                   'sin_exit', 'FC_1', 'FC_2', 'FC_3', 'FC_4', 'FC_5', 'TRAFFIC_SIGNAL', 'STOP','MeanSpeed']
MODEL_OUTPUTS  = ['min_velocity','max_velocity','out_velocity']
MAX_LENGTH = 800
WEIGHTS_ON = ['TRAFFIC_SIGNAL','STOP']
WEIGHT = 5.

# Define and train the model
with tf.device('GPU:0'):
    first_stage_model = Encoder(inputs=MODEL_INPUTS, outputs=MODEL_OUTPUTS, units=UNITS, l1=L1, l2=L2, name=ENCODER_NAME,
                                velocity_scaling_factor=VELOCITY_SCALING_FACTOR, network_scaling_factor=NETWORK_SCALING_FACTOR)
    first_stage_model.train(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, data_path=data_path, 
                            validation_data_path=validation_data_path, network_columns=network_columns, 
                            max_length=MAX_LENGTH,parallel=False, sample_weights_on=WEIGHTS_ON, sample_weight=WEIGHT)
    first_stage_model.save("First_stage_model")

first_stage_model.make_inference_model()
first_stage_model.encoder_model.save('models/enc_encoder.h5')
first_stage_model.decoder_model.save('models/enc_decoder.h5')



    
