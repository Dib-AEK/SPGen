# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:11:18 2022

@author: diba
"""
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Add, LayerNormalization, Multiply,
                                     Concatenate, GaussianNoise, LSTM, GRU, ReLU,
                                     Bidirectional, Reshape, Attention, Minimum, Maximum)
from tensorflow.keras import regularizers, Sequential
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class WeightThresholdCallback(tf.keras.callbacks.Callback):
    # weight pruning
    # iterative pruning
    # fine-grained pruning.
    def __init__(self, threshold=1e-2, batch_len = 500):
        super(WeightThresholdCallback, self).__init__()
        self.max_threshold = threshold
        self.threshold     = threshold  
        self.batch_len = batch_len
        self.batch_counter = 0
            
    def apply_weight_threshold(self):
        for layer in self.model.layers:            
            weights = layer.get_weights()
            if len(weights):
                new_weights = []
                for w in weights:
                    w[tf.math.abs(w) < self.threshold] = 0.0
                    new_weights.append(w)
                layer.set_weights(new_weights)
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch <2:
            self.threshold = 1e-6
        else:
            self.threshold = min(self.threshold*3,self.max_threshold)
    
    def on_epoch_end(self, epoch, logs=None):
        self.apply_weight_threshold()
          
    
    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1            
        if self.batch_counter==self.batch_len:
            self.apply_weight_threshold()
            self.batch_counter = 0


class AddNorm(tf.keras.layers.Layer):
    """Residual connection followed by layer normalization."""

    def __init__(self, name=None, **kwargs):
        super().__init__()
        if name: self._name = name
        self.ln = LayerNormalization(**kwargs)
        self.add = Add()

    def call(self, X):
        return self.ln(self.add(X))


class negloglik(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(negloglik, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)

    def get_config(self):
        config = super(negloglik, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class myLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(myLoss, self).__init__(**kwargs)
        self.mse_loss  = tf.keras.losses.MeanSquaredError()
        #self.msle_loss = tf.keras.losses.MeanSquaredLogarithmicError()
        self.mae_loss  = tf.keras.losses.MeanAbsoluteError() 

    def call(self, y_true, y_pred):
        # compute the mean squared error between y_true and y_pred
        mse  = self.mse_loss(y_true, y_pred)
        #msle = self.msle_loss(y_true, y_pred) 
        # compute the penalty for predicting values below the minimum threshold
        min_true = tf.math.reduce_min(y_true,axis=-2)
        min_pred = tf.math.reduce_min(y_pred,axis=-2)
        diff_min = tf.maximum(min_pred - min_true, 0)
        
        max_true = tf.math.reduce_max(y_true,axis=-2)
        max_pred = tf.math.reduce_max(y_pred,axis=-2)
        diff_max = tf.maximum(max_true - max_pred, 0)
        
        mean_true = tf.math.reduce_mean(y_true,axis=-2)
        mean_pred = tf.math.reduce_mean(y_pred,axis=-2)
        diff_mean = self.mae_loss(y_true,y_pred)
        
        penalty = tf.math.reduce_mean(diff_min+diff_max)+3*diff_mean
        # combine the mse and msle and penalty terms
        loss = mse + 0.3*penalty
        return loss

    def get_config(self):
        config = super(myLoss, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    
################### Sampling layers #############################

class GenSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, resolution=11 , **kwargs):
        super(GenSamplingLayer, self).__init__(**kwargs)
        self.resolution = resolution
    def call(self, x):
        out,k,i = x
        ki = k*3 if (i==0 or i==self.resolution-1) else k
        samples = out.sample(ki)
        prob    = tf.squeeze(out.log_prob(samples),axis=-1)
        idx_max_prob = tf.math.argmax(prob,axis=0)
        samples = tf.transpose(samples,perm=(1,2,3,0))
        out = tf.gather(samples, indices=idx_max_prob,axis=-1,batch_dims=1)
        out = tf.squeeze(out,axis=-1)
        return out
    def get_config(self):
        config = super(GenSamplingLayer, self).get_config()
        config['resolution'] = self.resolution
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EncSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncSamplingLayer, self).__init__(**kwargs)
    def call(self, x):
        out,k = x
        samples = out.sample(k)
        prob    = tf.squeeze(out.prob(samples),axis=-1)
        idx_max_prob = tf.math.argmax(prob,axis=0)
        samples = tf.transpose(samples,perm=(1,2,3,0))
        out = tf.gather(samples, indices=idx_max_prob,axis=-1,batch_dims=1)
        return out
    def get_config(self):
        config = super(EncSamplingLayer, self).get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MinMaxConstraints(tf.keras.layers.Layer):
    def __init__(self, outputs, **kwargs):
        super(MinMaxConstraints, self).__init__(**kwargs)
        self.outputs = outputs
        self.idx_min = outputs.index('min_velocity')
        self.idx_max = outputs.index('max_velocity')
        self.idx_out = outputs.index('out_velocity')
    def call(self, x):
        min_x = tf.gather(x,indices=[self.idx_min],axis=-1)
        out_x = tf.gather(x,indices=[self.idx_out],axis=-1)
        max_x = tf.gather(x,indices=[self.idx_max],axis=-1)
        min_x = tf.math.minimum(tf.math.minimum(min_x, out_x),max_x)
        max_x = tf.math.maximum(tf.math.maximum(min_x, out_x),max_x)
        list_idx = {self.idx_min: min_x,  self.idx_max: max_x, self.idx_out: out_x}
        out = tf.concat([list_idx[i] for i in range(3)],axis=-1)
        return out
    def get_config(self):
        config = super(MinMaxConstraints, self).get_config()
        config['outputs'] = self.outputs
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

################### GENERATOR LAYERS ###############################

class GenEnc(tf.keras.layers.Layer):
    def __init__(self, units,l1=0,l2=0, **kwargs):
        super(GenEnc, self).__init__(**kwargs)
        self.units = units

        regularizer = regularizers.L1L2(l1=l1,l2=l2)

        self.Li = Dense(units, activation='tanh', name='Li', kernel_regularizer= regularizers.L1L2(l1=l1,l2=l2))
        self.L1 = Dense(units, activation='relu', name='L1', kernel_regularizer= regularizers.L1L2(l1=l1,l2=l2))
        self.L2 = Dense(units, activation='relu', name='L2', kernel_regularizer= regularizers.L1L2(l1=l1,l2=l2))
        
        self.noise   = GaussianNoise(0.005,name='noise')
        self.add_li_l1 = AddNorm(name='add_li_l1')
        self.add_l1_l2 = AddNorm(name='add_l1_l2')


        self.state_g1_layer = Dense(units, activation='tanh', name='Lg1', kernel_regularizer=regularizer)
        self.state_g2_layer = Dense(units, activation='tanh', name='Lg2', kernel_regularizer=regularizer)
        self.state_h_layer = Dense(units, activation='tanh', name='Lh', kernel_regularizer=regularizer)
        self.state_c_layer = Dense(units, activation='tanh', name='Lc', kernel_regularizer=regularizer)

    def call(self, x):
        x = self.Li(x)
        xr= self.L1(x)
        x = self.add_li_l1([x,xr])
        x = self.noise(x)
        xr= self.L2(x)
        x2 = self.add_l1_l2([x,xr])        
        
        state_g1 = self.state_g1_layer(x)
        state_g2 = self.state_g2_layer(x)
        state_h  = self.state_h_layer(x2)
        state_c  = self.state_c_layer(x2)
        return (state_g1,state_g2,state_h,state_c)

    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(GenEnc, self).get_config()
        config['units'] = self.units
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GenPredictionBlock(tf.keras.layers.Layer):
    def __init__(self, units, l1=0, l2=0, **kwargs):
        super(GenPredictionBlock, self).__init__(**kwargs)
        self.units = units
   
        self.pre_layer = Dense(4, activation='relu', kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        
        self.lstm1 = GRU(units, return_sequences=True, return_state=True, name='gru1', 
                         kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.lstm2 = GRU(units, return_sequences=True, return_state=True, name='gru2',
                         kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.lstm3 = LSTM(units, return_sequences=True, return_state=True, name='lstm1', 
                          kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.rnn_states = 4
        self.addnorm_rnn1 = AddNorm(name='addnorm1')
        self.addnorm_rnn2 = AddNorm(name='addnorm2')
        # self.concat = Concatenate()

        self.pred_layer = DistLayer(units=units,
                                    l1=l1,l2=l2,
                                    name='pred')


    def call(self, x, states):
        state_g1,state_g2,state_h,state_c = states

        xo = self.pre_layer(x)

        x, state_g1 = self.lstm1(xo, initial_state=state_g1)

        xg, state_g2 = self.lstm2(x, initial_state=state_g2)
        x = self.addnorm_rnn1([x, xg])

        #x = self.concat([x,xo])
        xg, state_h, state_c = self.lstm3(x, initial_state=(state_h, state_c))
        x = self.addnorm_rnn2([x, xg])
        
        x, mean = self.pred_layer((x,xo))
        return (x, mean, (state_g1,state_g2,state_h,state_c))
        
    def get_zeros_states(self, batch_size):
        return tuple([tf.zeros((batch_size, self.units)) for i in range(self.rnn_states)])
    
    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(GenPredictionBlock, self).get_config()
        config['units'] = self.units
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DistLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, l1=0,l2=0, **kwargs):
        super(DistLayer, self).__init__(**kwargs)
        self.units = units

        self.ml = Dense(units, activation='relu', name='ml', kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.sl = Dense(units, activation='relu', name='sl', kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))

        self.mean_layer = Dense(1, activation='relu',kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.std_layer = Dense(1, activation='softplus',kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.skew_layer = Dense(1, activation='softplus',kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2))
        self.density_layer = tfp.layers.DistributionLambda(lambda t: tfd.TwoPieceNormal( loc=t[0], scale=t[1], skewness=t[2]))

        self.add1 = AddNorm()
        self.add2 = AddNorm()
        self.concat = Concatenate()
        self.concat0 = Concatenate()
        # self.min1 = Minimum()
        # self.min2 = Minimum()

    def call(self, x):
        x,xo = x
        x1 = self.concat0([x,xo])
        xm = self.add1([self.ml(x1), x])
        xs = self.add2([self.sl(xm), xm])
        x = self.concat([xm, xs])

        mean = self.mean_layer(x)
        std = self.std_layer(x) * tf.constant(0.05)+ tf.constant(1e-5)
        skew = self.skew_layer(x) +  tf.constant(1e-5)

        return (self.density_layer((mean, std, skew)), mean)

    def get_config(self):
        config = super(DistLayer, self).get_config()
        config['units'] = self.units
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


################### ENCODER LAYERS #################################


class OneParameterLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(OneParameterLayer, self).__init__(**kwargs)
        self.relu = ReLU(max_value=1.0)

    def build(self, input_shape):
        # Define the trainable parameter
        self.my_param = self.add_weight(shape=(1,),
                                        initializer=tf.keras.initializers.Constant(value=0.7),
                                        trainable=True,
                                        name = 'scale_k')
    def call(self, inputs, constant = 0.0):
        # Multiply the input by the trainable parameter
        constant = tf.constant(constant,dtype='float32')
        paramter = self.relu(self.my_param)
        paramter = paramter * (1.0-constant) + (1-paramter)*constant
        return inputs * paramter
    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(OneParameterLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    
class EncEnc(tf.keras.layers.Layer):
    def __init__(self, preprocessing_units, l1=0, l2=0, _dtype='float32', **kwargs):
        super(EncEnc, self).__init__(**kwargs)
        self.preprocessing_units = preprocessing_units
        self._dtype = _dtype

        kernel_cst = tf.keras.constraints.MinMaxNorm(min_value=0, max_value=2, rate=0.5, axis=0)

        d_k = int(preprocessing_units)
        self.Li = Dense(d_k, activation='tanh', name='Li', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), kernel_constraint=kernel_cst, dtype=_dtype)
        self.L1 = Dense(d_k, activation='relu', name='L1', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), kernel_constraint=kernel_cst, dtype=_dtype)
        self.L2 = Dense(d_k, activation='relu', name='L2', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), kernel_constraint=kernel_cst, dtype=_dtype)
        
        self.noise   = GaussianNoise(0.005,name='noise')
                
        self.addnorm_Li_L2 = AddNorm(name='addnorm_Lts_l1')

        self.rnn1 = Bidirectional(LSTM(int(preprocessing_units / 2), return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)), dtype=_dtype)
        self.rnn2 = Bidirectional(LSTM(int(preprocessing_units / 2), return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)), dtype=_dtype)
        self.bi_rnn_add = AddNorm()

    def call(self, x):
        x  = self.Li(x)        
        xl = self.L1(x)        
        xl = self.L2(xl)
        x  = self.addnorm_Li_L2([x,xl])        
        
        xl = self.rnn1(x)
        xl = self.rnn2(xl)
        x  = self.bi_rnn_add([x,xl])
        x  = self.noise(x)
        return x

    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(EncEnc, self).get_config()
        config['preprocessing_units'] = self.preprocessing_units
        config['_dtype'] = self._dtype
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncPredictionBlock(tf.keras.layers.Layer):
    def __init__(self, units, outputs, l1=0, l2=0, _dtype='float32', **kwargs):
        super(EncPredictionBlock, self).__init__(**kwargs)
        self.units = units
        self.outputs = outputs
        self._dtype = _dtype

        self.rnn1 = LSTM(units, return_sequences=True, return_state=True, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), dtype=_dtype)
        self.rnn2 = LSTM(units, return_sequences=True, return_state=True, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), dtype=_dtype)
        self.rnn3 = GRU(units, return_sequences=True, return_state=True, kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2), dtype=_dtype)
        self.rnn_states = 5
        self.rnn_add = [AddNorm(), AddNorm()]        

        self.postL = EncDistLayer(units=units,
                                  regularizer=regularizers.L1L2(l1=l1, l2=l2),
                                  outputs=outputs,
                                  name='sampling_layer',
                                  _dtype=_dtype)
        self.concat = Concatenate(axis=-1)
        #self.concat2 = Concatenate(axis=-1)

    def call(self, x, shifted_y, states=(None, None, None, None, None)):
        (state_h1, state_c1, state_h2, state_c2, state_g) = states

        x2 = self.concat([x, shifted_y])
        x2, state_h1, state_c1 = self.rnn1(x2, initial_state=(state_h1, state_c1) if state_h1 is not None else None)
        x2, state_h2, state_c2 = self.rnn2(x2, initial_state=(state_h2, state_c2) if state_h2 is not None else None)
        x = self.rnn_add[0]([x, x2])
        
        #x2 = self.concat2([x, shifted_y])
        x2, state_g = self.rnn3(x, initial_state=state_g)        
        x = self.rnn_add[1]([x, x2])

        # PostProcessing
        x, m = self.postL((x,shifted_y))
        return x, m, (state_h1, state_c1, state_h2, state_c2, state_g)

    def get_zeros_states(self, batch_size):
        return tuple([tf.zeros((batch_size, self.units)) for i in range(self.rnn_states)])

    def get_config(self):
        # This is overwriting the get_config method in order to get the config of the layer :
        config = super(EncPredictionBlock, self).get_config()
        config['units'] = self.units
        config['outputs'] = self.outputs
        config['_dtype'] = self._dtype
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncDistLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, regularizer=None, outputs=None, _dtype='float32', **kwargs):
        super(EncDistLayer, self).__init__(**kwargs)
        self.units = units
        self.outputs = outputs
        self._dtype = _dtype

        self.ml = Dense(int(units/2), activation='relu', name='ml',
                        kernel_regularizer=regularizer,
                        dtype=_dtype)
        self.sl = Dense(int(units/2), activation='relu', name='sl',
                        kernel_regularizer=regularizer,
                        dtype=_dtype)

        self.mean_layer = Dense(len(outputs), activation='relu', dtype=_dtype)
        self.std_layer = Dense(len(outputs) ** 2, activation='softplus', dtype=_dtype)
        self.reshape = Reshape((-1, len(outputs), len(outputs)))
        self.density_layer = tfp.layers.DistributionLambda( lambda t: tfd.MultivariateNormalTriL(loc=t[0], scale_tril=t[1]) )

        self.add= AddNorm()
        self.concat = Concatenate()
        self.concat2 = Concatenate()
        # self.min1 = Minimum()
        # self.min2 = Minimum()

    def call(self, x):
        x,shifted_y = x
        xm = self.concat2([x,shifted_y])
        xm = self.ml(xm)
        xs = self.sl(xm)
        xs = self.concat([xm, xs])
        x  = self.add([x,xs])

        mean = self.mean_layer(x)
        std = self.std_layer(x) * tf.constant(0.05) + tf.constant(1e-5)
        std = self.reshape(std)

        return (self.density_layer((mean, std)), mean)

    def get_config(self):
        config = super(EncDistLayer, self).get_config()
        config['units'] = self.units
        config['outputs'] = self.outputs
        config['_dtype'] = self._dtype
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



















