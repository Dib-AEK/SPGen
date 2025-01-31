# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:44:27 2023

@author: diba
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import pandas as pd
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import SPGen.Layers_enc as Le
import SPGen.Layers_dec as Ld
from tensorflow.keras.layers import Maximum, Concatenate
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import warnings

# TODO: check the inputs, their normalisation, and all the units.
# TODO: add some other models to make the choice possible between the models
class SPGen(Model):
    def __init__(self,enc_enc = "SPGen/enc_encoder_TMP5.h5", enc_dec= "SPGen/enc_decoder_TMP5.h5", 
                      gen_enc = "SPGen/gen_encoder_TMP2.h5", gen_dec= "SPGen/gen_decoder_TMP2.h5", 
                      json_file_path = 'SPGen/globalVariables.json', **kwargs):
        super(SPGen,self).__init__(**kwargs)
        # This will load the trained deep learning models (LSTM Based models)
        custom_objects = {"GenEnc":Ld.GenEnc, "GenPredictionBlock":Ld.GenPredictionBlock,
                          'GenSamplingLayer':Ld.GenSamplingLayer}
        self.gen_enc = tf.keras.models.load_model(gen_enc,custom_objects=custom_objects)
        self.gen_dec = tf.keras.models.load_model(gen_dec,custom_objects=custom_objects)

        custom_objects = {"EncEnc":Le.EncEnc, "EncPredictionBlock":Le.EncPredictionBlock,
                          'EncSamplingLayer':Le.EncSamplingLayer,"MinMaxConstraints":Le.MinMaxConstraints}
        self.enc_enc = tf.keras.models.load_model(enc_enc,custom_objects=custom_objects)
        self.enc_dec = tf.keras.models.load_model(enc_dec,custom_objects=custom_objects)
        # This json file contains the normalization coefficient (for data preprocessing)
        self.json_file = json.load(open(json_file_path))
        self.velocity_scaling_factor = self.json_file['NETWORK_SCALING_FACTOR']['velocity']
        self.network_scaling_factors = self.json_file['NETWORK_SCALING_FACTOR']
        # This is the resolution that we used (11 points per link), it cannot be changed, because the models are trained for this resolution
        self.resolution = self.gen_dec.layers[
            [i.name for i in self.gen_dec.layers].index('gen_sampling_layer')
            ].resolution
        # These are the inputs and outputs of each stage
        self.enc_inputs = list(self.json_file['Encoder_inputs'])
        self.enc_outputs = list(self.json_file['Encoder_outputs'])
        self.enc_units = self.enc_dec.input_shape[0][-1]
        self.gen_inputs = list(self.json_file['Generator_inputs'])
        # You can add the filter that you want here, in order to filter the resulted speed (such as limiting accelerations)
        # self.speed_filter = lambda x: savgol_filter(x,9,3)
        # two useful keras layers (will be used later for time vector construction)
        self.max_layer_for_time = Maximum()
        self.concat_for_time = Concatenate()
           
    def generator_predict_sequence(self,x,k=3):
        # x is the inputs and k is number of sampling from the pdf
        # Encode the input as states vectors:
        states     = self.gen_enc(x)
        # Generate empty target sequence of length 1.
        output     = tf.zeros((x.shape[0], 1, 1))
        # Sampling loop for a batch of sequences
        out_sequence = []
        for i in tqdm(tf.range(0,self.resolution),desc='Second stage Inference '):
            output, states = self.gen_dec((output,states,k,i))
            out_sequence.append(output)
            
        return tf.concat(out_sequence,1)[:,:,0]*self.velocity_scaling_factor

    def initialize_encoder_states_and_outputs(self, in_velocity, batch_size):
        # Generate empty lstm input sequence of length 1.
        output_shape = len(self.enc_outputs)
        states_len = self.enc_units

        io = self.enc_outputs.index('out_velocity')
        i1 = output_shape- io - 1
        output = tf.concat([tf.zeros([batch_size, 1, io]),
                            in_velocity,
                            tf.zeros([batch_size, 1, i1])],
                           axis=-1)
        
        idx_enc_predblock = [i.name for i in self.enc_dec.layers].index('First_stage_prediction_layer')
        states = self.enc_dec.layers[idx_enc_predblock].get_zeros_states(batch_size)
        return states, output    

    def encoder_predict_sequence(self, input_seq, in_velocity, k=1):
        # Pass the encoder throught the sequence
        x = self.enc_enc(input_seq)

        # initilize states and outputs:
        states, output = self.initialize_encoder_states_and_outputs(in_velocity, input_seq.shape[0])

        # Pass the lstm layer throught "x" to predict outputs
        output_sequence = tf.constant(output)
        for i in tqdm(range(x.shape[1]), desc='      First stage Inference '):
            output, states = self.enc_dec((tf.gather(x, axis=1, indices=[i]),output,states,k))
            # add the result to the output_sequence            
            output_sequence = tf.concat([output_sequence, output], axis=1)

        return output_sequence[:, 1:, :]
       
    def generate_speed_profiles_from_pandas(self,batch,k=2, in_velocity = None):
        # this function generates speed profiles from a pandas dataframe
        # batch : dataframe that contains all inputs
        # k: number of simpling (higher 'k' result in less stockasticity)
        # in_velocity : the initial velocity, if None, the initial velocity will be set to 0
        
        length_of_sequences = batch.MeanSpeed.apply(len)
        inputs              = self.enc_inputs
        scaling_factors     = self.network_scaling_factors
        if in_velocity is not None:
            in_velocity = in_velocity/self.velocity_scaling_factor
        
        # This will process the inputs and generate the profiles
        x,slopes = self.make_SPGen_inputs_from_pandas(batch,inputs,scaling_factors)         
        positions,speed_profiles = self.call((x,length_of_sequences,k, in_velocity)) 
        # Remark: position is the distance
        
        # process the slope arrays (for mobsim later on, I think slopes should be filtered (low pass filter))
        slopes = tf.ones((x.shape[0],x.shape[1],11))*tf.expand_dims(slopes,-1)
        slopes = self.delete_same_positions_points(slopes)
        
        # Filter speed, you can add the filter you want, or just remove it for no filtering
        warnings.warn(f"A filter is used here to filter speed, you can modify it, or remove it", UserWarning)        
        speed_profiles = filter_speed_new(positions,speed_profiles)       
        
        # Constract time vector from velocity and distance
        # remark: when constructing the time vector, velocity is limited to v_min
        time = self.construct_time_vector(positions,speed_profiles, min_speed=2.0)   
                 
        # put the results in the dataframe
        batch['time'] = time.numpy().tolist()
        batch['speed'] = speed_profiles.numpy().tolist()
        batch['positions'] = positions.numpy().tolist()
        batch['slopes'] = slopes.numpy().tolist()
        # Process the real GPS Profiles 
        # this step is just to make it easy to plot later on, you can remove it
        gpsProfiles = list(map(lambda x: np.array(x).T.tolist(),batch.gpsVelocity_res10.values.tolist()))   
        gpsProfiles = tf.keras.utils.pad_sequences(gpsProfiles, padding='post', dtype=np.float32) 
        gpsProfiles = self.delete_same_positions_points(gpsProfiles) 
        batch['gpsProfiles'] = gpsProfiles.numpy().tolist()
        
        # Cut long sequences (because they were padded)
        for column in ['time','speed','positions','slopes','gpsProfiles']:
            batch.loc[:,column] = batch.apply(lambda x: x[column][:x['sequence_length']*10+1],axis=1)
         
        return batch.loc[:,['linkLength','MeanSpeed','slope',
                            'time','speed','positions','slopes',
                            'gpsProfiles']]
   
    def call(self,x):
        x,lengths,k, in_velocity = x
        if in_velocity is not None:
            return self.generate_speed_profiles(x,lengths=lengths,k=k, in_velocity=in_velocity) 
        else:
            return self.generate_speed_profiles(x,lengths=lengths,k=k) 
    
    def generate_speed_profiles(self,x, in_velocity=[0], k=3, lengths=None):        
        batch_size = x.shape[0]
        # Make the same in velocity for all the batch if it is not specified for each row
        if len(in_velocity)!=batch_size:
            in_velocity = np.ones(shape=(batch_size,1,1))*in_velocity[0]
        in_velocity = np.reshape(in_velocity,(-1,1,1))
        # Prediction steps
        links_speeds   = self.encoder_predict_sequence(x,in_velocity,k)    
        #links_speeds   = self.make_speed_goes_zeros_at_the_end(links_speeds,lengths)
        links_speeds   = self.transform_links_speed(links_speeds,in_velocity,x)
        speed_profiles = self.generator_predict_sequence(links_speeds,k) 
        speed_profiles = self.gather_speed_profiles(speed_profiles,batch_size)
        # Make the positions vector
        positions      = self.make_positions_vector(x,speed_profiles.shape)
        # Delete points at the same position
        positions      = self.delete_same_positions_points(positions)
        speed_profiles = self.delete_same_positions_points(speed_profiles)                
        return positions,speed_profiles
    
    def make_speed_goes_zeros_at_the_end(self,links_speeds,lengths):
        end_0 = [1 if self.enc_outputs[i]=='out_velocity' else 0 for i in range(len(self.enc_outputs))] 
        max_len = links_speeds.shape[1]
        mask = tf.stack([tf.sequence_mask(lengths-end_0[i],max_len) for i in range(len(self.enc_outputs))],
                         axis=-1)
        mask = tf.cast(mask,dtype=tf.float32)
        return tf.math.multiply(links_speeds,mask)
                       
    def signal_validity(self,x):
        valid_points = lambda x: [True,*[False if (x[i]<0.5 and x[i-1]<0.5) else True for i in range(1,len(x))]]
        return list(map(valid_points,x))

    def delete_same_positions_points(self,x):
        shape = x.shape        
        return tf.concat([x[:,0,:] ,
                          tf.reshape(x[:,1:,1:],(shape[0],(shape[1]-1)*(shape[2]-1)))
                          ], axis=1)
    
    def make_positions_vector(self,x,shape):
        dx           = 1/(shape[-1]-1)
        rel_positions= np.round([0+i*dx for i in range(shape[-1])],2)
        positions    = tf.concat([i*tf.ones(shape=(shape[0],shape[1],1)) for i in rel_positions],axis=2)        
        links_length = tf.gather(x,axis=-1,indices=[self.enc_inputs.index('linkLength')])
        links_length = links_length*self.network_scaling_factors['linkLength']  
        
        positions    = positions*links_length        
        cum_lengths  = tf.concat([tf.zeros((shape[0],1,1)),
                                  tf.math.cumsum(links_length,axis=1)[:,:-1,:]], axis=1)
        positions    = positions + cum_lengths
        return positions
               
    def construct_time_vector(self,p,vi,min_speed = 2):
        #p,v = np.array(p), np.array(v)      
        v   = vi/3.6  # convert km/h -> m/s
        min_speed = tf.ones((v.shape[0],1),dtype='float32')*tf.constant(min_speed/3.6,dtype='float32')
        vm = (v[:,:-1]+v[:,1:])/2 # mean velocity between (i) and (i+1)
        dp = p[:,1:]-p[:,:-1]     # dx
        #########################
        # part to remove unmobile phases
        vm = self.max_layer_for_time([vm,min_speed])  # limit the minimal speed
        #########################
        time = dp/vm        # dt = dx/v  
        time = self.concat_for_time([tf.zeros((v.shape[0],1)),
                                     time])
        return tf.math.cumsum(time,axis = 1)
    
    def transform_links_speed(self,links_speeds,in_velocity,x):
        # out velocity index
        outv_idx = self.enc_outputs.index('out_velocity')
        # transform the sequence (I can optimize it -> don't recreate an 'x' tensor)
        in_velocity    = tf.concat([in_velocity,
                                    tf.gather(links_speeds,axis=-1,indices=[outv_idx])[:,:-1,:]],
                                              axis=1)
        # add in_velocity to x
        x              = tf.concat([x,in_velocity,links_speeds],axis=-1)
        x_columns      = [*self.enc_inputs,"in_velocity",*self.enc_outputs]
        
        # forme the generator input
        idx_columns    = [x_columns.index(i) for i in self.gen_inputs]
        g_inputs       = tf.gather(x,axis=-1,indices=idx_columns)
        
        return tf.reshape(g_inputs,shape=(g_inputs.shape[0]*g_inputs.shape[1],g_inputs.shape[2]))
        
    def gather_speed_profiles(self,speed_profiles,batch_size):        
        sequence_length =  int(speed_profiles.shape[0]/batch_size)
        return tf.reshape(speed_profiles,shape=(batch_size,sequence_length,speed_profiles.shape[-1]))   

    def rename_if_exists(self,df,columns:dict):
        for k,v in columns.items():
            assert (k in df.columns) or (v in df.columns), f"No {k} nor {v} exist in the trips attributes"
            if v not in df.columns:
                df = df.rename(columns={k:v})
        return df
                
    def make_SPGen_inputs_from_pandas(self,batch_in,inputs,scaling_factors):
        batch = batch_in.copy(deep=True)
        batch = self.rename_if_exists(batch, columns={'freespeed':'speed_limit',
                                                      'length':'linkLength',
                                                      'TL':'TRAFFIC_SIGNAL'})
        
        # batch = batch.merge(fc, how='left', on='trip_id')        
        assert all(elem in batch.columns for elem in inputs), "Not all inputs to speed generator are available"                
        # Get slopes before any modif:
        slopes = tf.keras.utils.pad_sequences(batch.loc[:,'slope'].tolist(), 
                                              padding='post', dtype=np.float32)
        # form the input vector:
        batch = batch.loc[:,inputs]
        for c in inputs:
            batch[c] = batch[c].apply(lambda x: list(np.array(x)/scaling_factors[c]))
        batch = list(map(lambda x: np.array(x).T.tolist(),batch.values.tolist()))   
        batch = tf.keras.utils.pad_sequences(batch, padding='post', dtype=np.float32)        
        return batch,slopes
    
 

    
    
@tf.function
def acceleration_filter(l, v):
    """
        v: velocity [m/s]
        l: length [m]
    """
    output_signal = tf.TensorArray(tf.float32, size=tf.shape(v)[0])
    y = v[0]
    output_signal = output_signal.write(0, y)

    for i in range(1, tf.shape(v)[0]):
        a = v[i] * (v[i] - y) / (l[i] - l[i - 1])
        a = tf.clip_by_value(a, -2.2, 2.2)
        dy_needed = v[i] - y
        dy_model = tf.sqrt(tf.maximum(tf.pow(y, 2) + 2 * a * (l[i] - l[i - 1]), 0.0)) - y
        y = y + tf.where(tf.abs(dy_model) < tf.abs(dy_needed), dy_model, dy_needed)
        output_signal = output_signal.write(i, y)
    output_signal = output_signal.stack()
    
    return output_signal

@tf.autograph.experimental.do_not_convert
# @tf.function(input_signature=[tf.TensorSpec(shape=(None,2), dtype=tf.float32)])
def filter_one_row_new(v_p:tf.TensorSpec(shape=(None,2), dtype=tf.float32))\
                       -> tf.TensorSpec(shape=(None,), dtype=tf.float32):
    x = tf.gather(v_p,axis=-1,indices=0)
    y = tf.gather(v_p,axis=-1,indices=1)
    return tf.numpy_function(acceleration_filter, (x,y), tf.float32)


def filter_speed_new(positions,speed): 
    """

    Parameters
    ----------
    positions : Distance in [m]
    speed : speed in [km/h].

    Returns
    -------
    filtered_results : filtered speed

    """
    v_p = tf.concat([tf.expand_dims(positions,-1),
                     tf.expand_dims(speed,-1)/3.6],axis=-1)
    filtered_results = tf.map_fn(filter_one_row_new, 
                                 v_p,
                                 dtype=tf.float32,
                                 parallel_iterations = 12,
                                 fn_output_signature = tf.TensorSpec(shape=None, dtype=tf.float32)
                                 )
    return filtered_results*3.6



def interp_and_filter(xn:tf.TensorSpec((None,), tf.float32),
                      x:tf.TensorSpec((None,), tf.float32),
                      y:tf.TensorSpec((None,), tf.float32)) -> tf.TensorSpec((None,), tf.float32):
    # Apply cupy.interp
    yn = np.interp(xn, x, y).astype(np.float32)
    # Apply cupyx.scipy.signal.savgol_filter
    yn = savgol_filter(yn, window_length=5, polyorder=3, mode = "constant", cval = 0.0)
    # Get back values at x
    y = np.interp(x, xn, yn)
    return tf.cast(y,tf.float32)

@tf.autograph.experimental.do_not_convert
# @tf.function(input_signature=[tf.TensorSpec(shape=(None,2), dtype=tf.float32)])
def filter_one_row(v_p:tf.TensorSpec(shape=(None,2), dtype=tf.float32))\
                   -> tf.TensorSpec(shape=(None,), dtype=tf.float32):
    x = tf.gather(v_p,axis=-1,indices=0)
    y = tf.gather(v_p,axis=-1,indices=1)
    num = tf.math.ceil((x[-1] - 0) / 3) # x is the position, so 3 is 3 meter
    xn = tf.linspace(start = 0.0, stop = x[-1], num = tf.cast(num, tf.int32))
    return tf.numpy_function(interp_and_filter, (xn,x,y), tf.float32)


@tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,None), dtype=tf.float32)])
def filter_speed(positions,speed):  
    v_p = tf.concat([tf.expand_dims(positions,-1),
                     tf.expand_dims(speed,-1)],axis=-1)
    filtered_results = tf.map_fn(filter_one_row, 
                                 v_p,
                                 dtype=tf.float32,
                                 parallel_iterations = 12,
                                 fn_output_signature = tf.TensorSpec(shape=None, dtype=tf.float32)
                                 )
    return filtered_results


