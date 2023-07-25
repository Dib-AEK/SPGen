from GENERATOR_MODEL import Generator
import Functions as F
import tensorflow as tf
import numpy as np
import os
import keras
from Layers import negloglik, myLoss
import matplotlib.pyplot as plt

# define paths
data_path_tispred = "data/dataframe_from_tispred_for_generator_training.pkl"
data_path = os.path.join(os.path.dirname(os.getcwd()),'routes_filtering','training_data.parquet')
validation_data_path = os.path.join(os.path.dirname(os.getcwd()),'routes_filtering','validation_data.parquet')

network_columns = ['averageSpeed', 'freeFlowSpeed','speed_limit','linkLength', 'slope', 'shape_parameter', 'cos_enter','sin_enter', 'cos_exit',
                   'sin_exit', 'FC_1', 'FC_2', 'FC_3', 'FC_4', 'FC_5', 'TRAFFIC_SIGNAL', 'STOP', 'YIELD', 'PEDESTRIAN_CROSSING']

# Hyperparameters
UNITS = 40
L1 = 5e-4
L2 = 2e-4
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 200
VELOCITY_SCALING_FACTOR = 40
NETWORK_SCALING_FACTOR   =  {'in_velocity':40,'min_velocity':40,'max_velocity':40,
                             'out_velocity':40,'MeanSpeed':40,'linkLength':100,'cos_enter':1,
                             'sin_enter':1, 'cos_exit':1,'sin_exit':1,'slope':0.05}
MODEL_INPUTS   = ['linkLength','MeanSpeed','in_velocity','min_velocity','max_velocity','out_velocity']
GENERATOR_NAME   = "GENERATOR_V3"

# define and train the model
with tf.device('GPU:0'):
    second_stage_model = Generator(inputs=MODEL_INPUTS, units=UNITS, l1=L1, l2=L2, name=GENERATOR_NAME,
                                velocity_scaling_factor=VELOCITY_SCALING_FACTOR, network_scaling_factor=NETWORK_SCALING_FACTOR)    
    second_stage_model.train( batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, data_path=data_path,
                              network_columns=network_columns,
                              extra_data_path=data_path_tispred,
                              val_data_path = validation_data_path)
    second_stage_model.save("Second_stage_model")  




