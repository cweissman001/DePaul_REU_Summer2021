# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:32:48 2021

@author: weissmce
"""

import pandas as pd
import cv2
import pickle

dfPos = pd.read_csv('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\posPairs.csv')
dfNeg = pd.read_csv('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\negPairs.csv')
pairs = []
labels = []

#265300
directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\Both\\'
for x in range(35000, 42000):
    print(x)
    negName1 = dfNeg['Img1'][x] #gerts image name
    negName2 = dfNeg['Img2'][x]
    
    posName1 = dfPos['Img1'][x] 
    posName2 = dfPos['Img2'][x]
    
    neg1 = cv2.imread(directory + negName1)
    neg1 = neg1.reshape(-1, 30, 30, 3)
    
    neg2 = cv2.imread(directory + negName2)
    neg2 = neg2.reshape(-1, 30, 30, 3)
    
    pos1 = cv2.imread(directory + posName1)
    pos1 = pos1.reshape(-1, 30, 30, 3)
    
    pos2 = cv2.imread(directory + posName2)
    pos2 = pos2.reshape(-1, 30, 30, 3)
    
    pairs += [(pos1,pos2)]
    pairs += [(neg1, neg2)]
    labels += [1, 0]



pickle_out = open("C:\\Users\\weissmce\\Desktop\\GitRepo\\hvh\\TrainLabels5.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\TrainPairs5.pickle", "wb")
pickle.dump(pairs, pickle_out1)
pickle_out1.close()