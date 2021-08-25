# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:03:12 2021

@author: lilly
"""
import pandas as pd 
import numpy as np 

# Load in data from csv files and define label and image variables
train4 = pd.read_csv('LIDC_Max_Slice_Pixel_Values2.csv')
#test = pd.read_csv('test data')
train_ids4 = train4['InstanceID']
train_images4 = train4.drop(['InstanceID'], axis=1)
#test_ids = test['InstanceID']
#test_images = test.drop(['InstanceID'], axis=1)

# Reshape the data so the pixel vectors turn into arrays, and so the instance id's are also arrays
train_data4 = np.array(train_images4)  # -1? not sure why
train_ids4 = np.array(train_ids4)
#test_data = np.array(test_images).reshape(-1,126,128)
#test_ids = np.array(test_ids)

input_shape = train_data4.shape[1:]