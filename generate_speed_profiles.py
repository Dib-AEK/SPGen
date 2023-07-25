# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:44:22 2023

@author: Abdelkader DIB
"""

from SPGen import SPGen
import os
import Functions as F
import tensorflow as tf

data_path:str=os.path.join(os.path.dirname(os.getcwd()),'data','Outputs','Lyon_Ile_de_France_G_data.parquet')

number_of_profiles = 1024*3

out_df_path:str="profiles.pkl"

with tf.device('GPU'):
    # Load the model
    model = SPGen()
    
    # Generate some profiles
    profiles = model.plot_some_profiles(num=number_of_profiles,
                                        data_path=data_path,
                                        how = 'max_from_k',k=3)
# Save the results
profiles.to_pickle(out_df_path)
