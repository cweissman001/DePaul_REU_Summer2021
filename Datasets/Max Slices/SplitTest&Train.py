# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:19:24 2021

@author: weissmce
"""
import pandas as pd 
import numpy 

df = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\Testing.csv")

df1 = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\Training.csv")

df2 = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\LIDC_Max_Slice_Pixel_ValuesSMALL.csv")




testID = df["InstanceID"]

trainID = df1["InstanceID"]


#df2 = df2.set_index("InstanceID")


#print(df2['InstanceID'][4])
testArr = (df2.loc[df2['InstanceID']==118])
####easiest way to initialize array to the correct shape but need to delete in actual file
for x in testID:
    
    additionTest = (df2.loc[df2['InstanceID']==x])
    testArr = numpy.vstack((testArr, additionTest))
    
    ##arr = numpy.vstack((arr, newArr))

trainArr = (df2.loc[df2['InstanceID']==118])
####easiest way to initialize array to the correct shape but need to delete in actual file
for x in trainID:
    
    additionTrain = (df2.loc[df2['InstanceID']==x])
    trainArr = numpy.vstack((trainArr, additionTrain))
    
  
output_file = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\TestPixel.csv'
#where to save the csv file
numpy.savetxt(output_file, testArr, fmt= "%10.5f", delimiter=',')


output_file = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\TrainPixel.csv'
numpy.savetxt(output_file, trainArr, fmt= "%10.5f", delimiter=',')
    ####write a loop to go through every index and see where that instance id is 
