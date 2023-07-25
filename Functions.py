# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:07:48 2022

@author: Abdelkader DIB
"""
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd
import os
import pyarrow.parquet as pq
from tensorflow.keras.layers import ReLU
import concurrent.futures

####################### some data and basic functions #########################

def shapeParameter(x):
    """ a shape parameter that I defines as length of a curve to its euc distance"""
    euc_dist = np.sqrt((x[-1] - x[0]) ** 2 + 1)
    dist = sum(np.sqrt([(x[i] - x[i - 1]) ** 2 + 0.1 ** 2 for i in range(1, 11)]))
    return euc_dist / dist


def sumPositifAcceleration(x):
    """sum of positif acceleration in a speed profiles (acc to distance) """
    d = np.diff(x)
    return sum(d[d > 0])


class MetaSingleton(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in MetaSingleton.__instances:
            MetaSingleton.__instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return MetaSingleton.__instances[cls]


def create_folder_if_doesnt_exists(path):
    it_exists = os.path.exists(path)
    if not it_exists:
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)


def Accel(row):
    v = row['velocity'] / 3.6
    vm = ((v[1:] + v[:-1]) / 2)
    dx = row['link_length'] / 10
    dt = dx / vm
    acc = np.diff(v) / dt
    return [max(acc), min(acc)]


@tf.function
def Get_min_max_out(x, idx_min, idx_max, idx_out):
    min_x = tf.gather(x, indices=[idx_min], axis=-1)
    out_x = tf.gather(x, indices=[idx_out], axis=-1)
    max_x = tf.gather(x, indices=[idx_max], axis=-1)
    return min_x, max_x, out_x


def top_k(x, k=3):
    values, idx = tf.math.top_k(x, k=k)
    values = values / tf.math.reduce_sum(values, axis=-1, keepdims=True)
    idx_in_idx = tf.expand_dims(tfd.Categorical(probs=values).sample(), axis=-1)
    idx = tf.gather(idx, indices=idx_in_idx, axis=-1, batch_dims=2)
    return idx



class EncDataset(Sequence):
    """This class is a tf sequence used when training the model"""

    def __init__(self, data_path, input_columns, output_columns, network_columns, NETWORK_SCALING_FACTOR,
                 VELOCITY_SCALING_FACTOR, batch_size=128, training=True, max_length=1000, dtype='float32',parallel=False,
                 sample_weights_on=['TRAFFIC_SIGNAL','STOP','YIELD','PEDESTRIAN_CROSSING'],sample_weight=5.):
        super(EncDataset, self).__init__()
              
        self.data_path = data_path
        self.batch_size = batch_size
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.training = training
        self.network_columns = network_columns
        self.input_columns_in_network = [network_columns.index(i) for i in input_columns if i in network_columns]
        self.input_columns_not_in_network = [i for i in input_columns if i not in network_columns]
        self.NETWORK_SCALING_FACTOR = np.array([NETWORK_SCALING_FACTOR[network_columns[c]] 
                                                for c in self.input_columns_in_network])
        self.VELOCITY_SCALING_FACTOR = VELOCITY_SCALING_FACTOR
        self.out_velocity_index = output_columns.index('out_velocity')
        self.parquet_file = self.get_parquet_file()
        self.max_length = max_length
        self.dtype = dtype
        self.relu = ReLU()
        self.iterator = self.make_enc_data_generator() if not parallel else self.make_enc_data_generator_multiprocessing()
        self.sample_weights_on = sample_weights_on
        self.sample_weights_on_idx = [input_columns.index(i) for i in sample_weights_on if i != 'slope']
        self.slope_index = input_columns.index('slope') if 'slope' in  sample_weights_on else -1
        self.sample_weight = sample_weight
        
        # make asserts
        assert all(item in network_columns for item in input_columns[:len(
            self.input_columns_in_network)]), 'network columns should be first in input columns'

    def sample_weight_on_max_velocity(self,v):
        if v<=80: return 0
        if 80<v<=100: return 0.5
        if 100<v<=160: return 1
        if v>160: return 0
        
        
    def preprocess_data_for_encoder(self, data):
        dtype = self.dtype
        data = data.to_pandas()
        network = data.NetworkFeatures.values.tolist()
        network = tf.keras.utils.pad_sequences(list(map(list, network)), padding='post', dtype=dtype)
        network = tf.gather(network, axis=-1, indices=self.input_columns_in_network)
        network = network / tf.constant(self.NETWORK_SCALING_FACTOR, dtype=dtype)
        network = network[:, :self.max_length, :]
        
        if self.training and len(self.sample_weights_on):
            sample_weights = tf.gather(network,axis=-1,indices=self.sample_weights_on_idx)
            sample_weights = tf.math.reduce_sum(sample_weights,axis=-1,keepdims=False)*self.sample_weight+1.
            #max velocity weight:
            max_velocityS100 = data.loc[:, 'max_velocity'].apply(lambda x: [self.sample_weight_on_max_velocity(i) for i in x])
            max_velocityS100 = tf.keras.utils.pad_sequences(max_velocityS100, padding='post', dtype=dtype)[:, :self.max_length]
            sample_weights   = sample_weights+max_velocityS100*self.sample_weight
             
            if self.slope_index>=0:
                slopes = tf.gather(network,axis=-1,indices=self.slope_index)
                slopes = tf.cast(tf.less(0.3,tf.math.abs(slopes)),'float32')
                sample_weights = sample_weights + slopes*self.sample_weight/2
                #sample_weights = sample_weights + tf.math.abs(slopes)*1.2
        else:
            sample_weights=[]
        
        other_inputs = [data.loc[:, c] for c in self.input_columns_not_in_network]
        other_inputs = [tf.keras.utils.pad_sequences(out, padding='post', dtype=dtype) for out in other_inputs]
        other_inputs = [tf.expand_dims(out, axis=-1) / tf.constant(self.VELOCITY_SCALING_FACTOR, dtype=dtype)
                        for out in other_inputs]
        if len(other_inputs) == 1:
            other_inputs = other_inputs[0][:, :self.max_length, :]
        else:
            other_inputs = tf.concat(other_inputs, axis=-1)
            other_inputs = other_inputs[:, :self.max_length, :]

        outputs = [tf.keras.utils.pad_sequences(data.loc[:, c], padding='post', dtype=dtype) for c in
                   self.output_columns]
        outputs = [tf.expand_dims(out, axis=-1) for out in outputs]
        outputs = tf.concat(outputs, axis=-1) / tf.constant(self.VELOCITY_SCALING_FACTOR, dtype=dtype)
        outputs = outputs[:, :self.max_length, :]
        outputs = self.relu(outputs)

        shifted_outputs, in_velocity = self.shift_output(outputs, data.loc[:, 'in_velocity'], dtype=dtype)
        if self.training:
            return ((tf.concat([network, other_inputs], axis=-1),
                     shifted_outputs),
                    (outputs,
                     outputs),
                    sample_weights)
        else:
            velocity = tf.keras.utils.pad_sequences(
                data.loc[:, 'gpsVelocity_res10'].apply(lambda x: list(map(list, x))), padding='post', dtype=dtype)
            velocity = velocity[:, :self.max_length, :]
            return ((tf.concat([network, other_inputs], axis=-1),
                     shifted_outputs,
                     in_velocity,
                     velocity),
                    outputs)

    def shift_output(self, y, in_velocity_column, dtype='float32'):
        in_velocity = tf.constant(in_velocity_column.apply(lambda x: x[0]), dtype=dtype,
                                  shape=(y.shape[0], 1, 1)) / tf.constant(self.VELOCITY_SCALING_FACTOR, dtype=dtype)
        io = self.out_velocity_index
        i1 = y.shape[-1] - io - 1
        first_value = tf.concat([tf.zeros([y.shape[0], 1, io], dtype=dtype),
                                 in_velocity,
                                 tf.zeros([y.shape[0], 1, i1], dtype=dtype)],
                                axis=-1)
        return tf.concat([first_value, y[:, :-1, :]], axis=1), in_velocity

    def set_training(self, value):
        self.training = value

    def __len__(self):
        return int(np.ceil(self.parquet_file.metadata.num_rows / self.batch_size))

    def __getitem__(self, item):
        return next(self.iterator)

    def get_parquet_file(self):
        return pq.ParquetFile(self.data_path)

    def make_enc_data_generator(self):
        while True:
            iterator = self.parquet_file.iter_batches(batch_size=self.batch_size, use_threads=True)
            for batch in iterator:
                res = self.preprocess_data_for_encoder(batch)
                if res[0][0].shape[0]==self.batch_size:
                    yield res
    
    def make_enc_data_generator_multiprocessing(self,queue_size=32):
        # not a good implimnetation, it's soo slow
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            while True:
                iterator = self.parquet_file.iter_batches(batch_size=self.batch_size, use_threads=True)            
                futures = set()
                for batch in iterator:
                    # submit each item to the executor for processing
                    futures.add(executor.submit(self.preprocess_data_for_encoder, batch))
                    # yield the processed items as they become available
                    if len(futures)>=queue_size:
                        completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        for task_done in completed:
                            res = task_done.result()
                            if res[0][0].shape[0]==self.batch_size:
                                yield res
                completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
                for task_done in completed:
                    yield task_done.result()
    

    def get_encoder_dataset(self):
        self.training = True
        return tf.data.Dataset.from_generator(self.make_enc_data_generator,
                                              output_signature=(
                                                  (tf.TensorSpec(shape=(self.batch_size, None, len(self.input_columns)), dtype=self.dtype),
                                                   tf.TensorSpec(shape=(self.batch_size, None, len(self.output_columns)), dtype=self.dtype)),
                                                  (tf.TensorSpec(shape=(self.batch_size, None, len(self.output_columns)),  dtype=self.dtype),
                                                   tf.TensorSpec(shape=(self.batch_size, None, len(self.output_columns)),  dtype=self.dtype))
                                              )).shuffle(16)






class GenDataset(Sequence):
    """This class is a tf sequence used when training the model"""

    def __init__(self, data_path, inputs, network_columns, NETWORK_SCALING_FACTOR,
                 VELOCITY_SCALING_FACTOR, batch_size=128, extra_data_path=None):
        super(GenDataset, self).__init__()
              
        self.data_path = data_path
        self.batch_size = batch_size
        self.inputs = inputs
        self.network_columns = network_columns
        self.input_columns_in_network = [network_columns.index(i) for i in inputs if i in network_columns]
        self.input_columns_not_in_network = [i for i in inputs if i not in network_columns]
        self.NETWORK_SCALING_FACTOR = NETWORK_SCALING_FACTOR
        self.VELOCITY_SCALING_FACTOR = VELOCITY_SCALING_FACTOR
        self.extra_data_path = extra_data_path
                
        self.parquet_file = self.get_parquet_file()
        
        if extra_data_path is not None:
            self.extra_data = pd.read_pickle(extra_data_path)
            self.extra_data_batch_size = int(self.batch_size/5)    
            self.extra_data_iterator = self.generator_from_extra_data()
        else:
            self.extra_data_batch_size = 0
        
        self.data_batch = self.batch_size-self.extra_data_batch_size

        self.relu = ReLU()
        self.iterator = self.make_gen_data_generator() 
                                 
        # make asserts
        assert all(item in network_columns for item in inputs[:len(
            self.input_columns_in_network)]), 'network columns should be first in input columns'

    def del_two_succesive_null_values(self,data):
        twoSuccessiveNullValues = lambda x: ((x[1:]+x[:-1])/2<1.5).sum()==0
        #twoSuccessiveNullValues= lambda x: sum([x[i]<l and x[i-1]<l for i in range(1,len_x)])<1           
        mask = data.apply(twoSuccessiveNullValues) 
        return mask
    
    def generator_from_extra_data(self):
        length = len(self.extra_data)
        num = self.extra_data_batch_size  
        columns = [i for i in self.extra_data.columns if i!='velocity']
        while True:
            for im1,ip1 in zip(range(0,length,num),
                               range(num,length,num)):
                data = self.extra_data.iloc[im1:ip1,:]

                sample_weights = np.logical_and(data.min_velocity<8,data.max_velocity>20)
                data.loc[sample_weights,'min_velocity'] = data.loc[sample_weights,'min_velocity']+np.random.rand(sample_weights.sum())*4
                                
                sample_weights = sample_weights.astype(np.float32)*30.0+\
                                np.array(data.max_velocity-data.min_velocity>30).astype(np.float32)*3.0+\
                                np.array(data.max_velocity>100).astype(np.float32)*10.0 + 1.0
                sample_weights = sample_weights.apply(lambda x: [x if (i==0 or i==10) else x*0.7 for i in range(11)])
                                                
                velocity = data.velocity/self.VELOCITY_SCALING_FACTOR
                data = data.loc[:,self.inputs]
                for c in columns:
                    data.loc[:,c] = data.loc[:,c]/self.NETWORK_SCALING_FACTOR[c]                
                
                sample_weights = tf.constant(sample_weights.values.tolist(),dtype='float32')
                
                y = tf.constant(velocity.values.tolist(),dtype='float32')
                y = tf.expand_dims(y, axis=-1)
                x = tf.constant(data.values.tolist(),dtype='float32')
                zeros = tf.zeros((y.shape[0],1,1),dtype='float32')
                y_shifted = tf.concat([zeros,y[:,:-1,:]], axis=1)
                yield  (x,y_shifted),y,sample_weights
                
                
    def preprocess_data_for_encoder(self, data):
        data = data.to_pandas()
        velocity = data.gpsVelocity_res10.explode(ignore_index=True)
        data['in_velocity']  = data.gpsVelocity_res10.apply(lambda x: [i[0] for i in x])
        data['out_velocity'] = data.gpsVelocity_res10.apply(lambda x: [i[-1] for i in x])
        data['max_velocity'] = data.gpsVelocity_res10.apply(lambda x: [max(i) for i in x])
        data['min_velocity'] = data.gpsVelocity_res10.apply(lambda x: [min(i) for i in x])
        
        outputs = pd.DataFrame(columns=self.inputs)
        if len(self.input_columns_in_network):
            for c in self.input_columns_in_network:
                column = self.network_columns[c]
                outputs[column] = data.NetworkFeatures.explode(ignore_index=True).apply(lambda x: x[c])
        
        if len(self.input_columns_not_in_network):
            for c in self.input_columns_not_in_network:
                outputs[c] = data.loc[:,c].explode(ignore_index=True)
        
        #before normalizing, I calculate the sample weight
        sample_weights = np.logical_and(outputs.min_velocity<10,outputs.max_velocity>20)        
        outputs.loc[sample_weights,'min_velocity'] = outputs.loc[sample_weights,'min_velocity']+np.random.rand(sample_weights.sum())*4
        #outputs.loc[:,'max_velocity'] = outputs.loc[:,'max_velocity']-np.random.rand(len(outputs))*5
        
        sample_weights = sample_weights.astype(np.float32)*20.0+\
                        np.array(outputs.max_velocity-outputs.min_velocity>30).astype(np.float32)*3.0+\
                        np.array(outputs.max_velocity>100).astype(np.float32)*10.0 + 1.0
        sample_weights = sample_weights.apply(lambda x: [x if (i==0 or i==10) else x*0.7 for i in range(11)])
        
        for c in outputs.columns:
            outputs.loc[:,c] = outputs.loc[:,c]/self.NETWORK_SCALING_FACTOR[c]
        
        mask = self.del_two_succesive_null_values(velocity)
        velocity = velocity[mask].reset_index(drop=True)
        outputs  = outputs[mask].reset_index(drop=True)
        sample_weights = sample_weights[mask].reset_index(drop=True)
        
        sample_weights = tf.constant(sample_weights.values.tolist(),dtype='float32')
        y = tf.constant(velocity.values.tolist(),dtype='float32')/self.VELOCITY_SCALING_FACTOR
        y = tf.expand_dims(y, axis=-1)
        x = tf.constant(outputs.values.tolist(),dtype='float32')
        zeros = tf.zeros((y.shape[0],1,1),dtype='float32')
        y_shifted = tf.concat([zeros,y[:,:-1,:]], axis=1)
        return (x,y_shifted),y,sample_weights


    def set_training(self, value):
        self.training = value

    def __len__(self):
        return int(np.ceil(self.parquet_file.metadata.num_rows / self.batch_size))

    def __getitem__(self, item):
        return next(self.iterator)

    def get_parquet_file(self):
        return pq.ParquetFile(self.data_path)

    def make_gen_data_generator(self):
        num = self.data_batch
        while True:
            iterator = self.parquet_file.iter_batches(batch_size=1024, use_threads=True)                        
            for batch in iterator:
                res = self.preprocess_data_for_encoder(batch)
                length = res[1].shape[0]
                
                for im1,ip1 in zip(range(0,length,num),
                                   range(num,length,num)):

                    (x,y_shifted),y,sample_weights = ((res[0][0][im1:ip1],res[0][1][im1:ip1]),
                                                       res[1][im1:ip1],
                                                       res[2][im1:ip1])
                    if self.extra_data_path is not None:
                        (x2,y_shifted2),y2,sample_weights2  = next(self.extra_data_iterator)
                        x = tf.concat([x,x2],axis=0)
                        y_shifted = tf.concat([y_shifted,y_shifted2],axis=0)
                        y = tf.concat([y,y2],axis=0)
                        sample_weights = tf.concat([sample_weights,sample_weights2],axis=0)
                                                
                    if x.shape[0]==self.batch_size:
                        # x_m = tf.math.reduce_mean(y,axis=1)
                        yield (x,y_shifted),(y,y),sample_weights
    

    def make_enc_data_generator_multiprocessing(self,queue_size=4):
        num = self.data_batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                iterator = self.parquet_file.iter_batches(batch_size=1024, use_threads=True)            
                futures = set()
                for batch in iterator:
                    # submit each item to the executor for processing
                    futures.add(executor.submit(self.preprocess_data_for_encoder, batch))
                    # yield the processed items as they become available
                    if len(futures)>=queue_size:
                        completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        for task_done in completed:
                            res = task_done.result()
                            length = res[1].shape[0]
                            
                            for im1,ip1 in zip(range(0,length,num),
                                               range(num,length,num)):

                                (x,y_shifted),y,sample_weights = ((res[0][0][im1:ip1],res[0][1][im1:ip1]),
                                                                   res[1][im1:ip1],
                                                                   res[2][im1:ip1])
                                if self.extra_data_path is not None:
                                    (x2,y_shifted2),y2,sample_weights2  = next(self.extra_data_iterator)
                                    x = tf.concat([x,x2],axis=0)
                                    y_shifted = tf.concat([y_shifted,y_shifted2],axis=0)
                                    y = tf.concat([y,y2],axis=0)
                                    sample_weights = tf.concat([sample_weights,sample_weights2],axis=0)
                                                            
                                if x.shape[0]==self.batch_size:
                                    yield (x,y_shifted),(y,y),sample_weights
                            
                            
                            
                completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
                for task_done in completed:
                    yield task_done.result()



    
