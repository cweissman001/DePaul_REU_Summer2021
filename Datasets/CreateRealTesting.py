# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:35:04 2021

@author: weissmce
"""
import pandas as pd
import numpy

testing50agree = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\50Agree.csv")
testingNOagree = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\NoneAgree.csv")

pixelVals = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\LIDC_Max_Slice_Pixel_ValuesSMALL.csv")

instID = testing50agree["InstanceID"][1]
rows = pixelVals.loc[pixelVals['InstanceID']==instID]
for i in range(1, 522):
    instID = testing50agree["InstanceID"][i]
    
    newRow = (pixelVals.loc[pixelVals['InstanceID']==instID])
    rows = numpy.vstack((rows, newRow))
    
    
    
    
instID = testingNOagree["InstanceID"][1]
rows2 = pixelVals.loc[pixelVals['InstanceID']==instID]
for i in range(1, 69):
    instID = testingNOagree["InstanceID"][i]
    
    newRow = (pixelVals.loc[pixelVals['InstanceID']==instID])
    rows2 = numpy.vstack((rows2, newRow))
    
    
    
numpy.savetxt("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\TestingVals50Agree.csv", rows, fmt= "%10.5f", delimiter=',')

numpy.savetxt("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\TestingValsNOAgree.csv", rows2, fmt= "%10.5f", delimiter=',')