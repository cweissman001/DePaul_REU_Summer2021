# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:58:47 2021

@author: weissmce
"""

import pandas as pd
import numpy

"""
df = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\AllAgreeCombo.csv")

list1 = df.index
####CREATION OF SEPERATE TESTING AND TRAINING FILES
Spic =[]
NotSpic = []
length = len(list1)
for x in range(length):
    
    rate = df["Binary Rating"][x]
    
    if rate ==1:
        Spic.append(x)
    else:
        NotSpic.append(x)
spicTest = Spic[:25]
spicRest = Spic[25:]

noSpicTest = NotSpic[:124]
noSpicRest = NotSpic[124:]
allRest = spicRest + noSpicRest
testSet = spicTest + noSpicTest


fullArr1 = numpy.asarray(df.iloc[0])
length = len(allRest)
for x in range(length):
    newArr1 = numpy.asarray(df.iloc[allRest[x]])
    fullArr1 = numpy.vstack((newArr1, fullArr1))
    
output_file = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training.csv'
#where to save the csv file
numpy.savetxt(output_file, fullArr1, fmt= '%s', delimiter=',')

    
fullArr = numpy.asarray(df.iloc[0])
length = len(testSet)
for x in range(length):
    #print(testSet[x])
    newArr = numpy.asarray(df.iloc[testSet[x]])
    fullArr = numpy.vstack((newArr, fullArr))
    
output_file1 = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Testing.csv'
numpy.savetxt(output_file1, fullArr, fmt= '%s', delimiter=',')
    
"""

df = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\Training.csv")
#### to create training pairs

list1 = df.index

from itertools import permutations

list2 = list(permutations(list1, 2)) #tuple of all possible permutaions

#positiveSpicPairs = []
positivePairs = []
negativePairs = []

length = len(list2)
print(list2[0])
print(list2[0][0])
 
print(df["Binary Rating"][list2[0][0]])
print(df["InstanceID"][list2[0][0]])

for x in range(length):
    firstInd = list2[x][0]
    secondInd = list2[x][1]
    
    firstRate = df["Binary Rating"][firstInd]
    secondRate = df["Binary Rating"][secondInd]
    
    firstInstanceId = df["InstanceID"][list2[x][0]]
    secondInstanceId = df["InstanceID"][list2[x][1]]
    
    if(firstRate == secondRate and firstRate == 0):
        positivePairs.append((firstInstanceId, secondInstanceId))  
        
    elif(firstRate == secondRate and firstRate == 1):
        positivePairs.append((firstInstanceId, secondInstanceId))  
        
    else:
        negativePairs.append((firstInstanceId, secondInstanceId)) 
        
import random
random.shuffle(positivePairs)
output_file1 = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\PosPairs.csv'
numpy.savetxt(output_file1, positivePairs, fmt= '%s', delimiter=',')

#output_file2 = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\PosNoSpicPairs.csv'
#numpy.savetxt(output_file2, positiveNoSpicPairs, fmt= '%s', delimiter=',')
random.shuffle(negativePairs)
output_file3 = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\NegPairs.csv'
numpy.savetxt(output_file3, negativePairs, fmt= '%s', delimiter=',')