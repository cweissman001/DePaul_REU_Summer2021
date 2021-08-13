# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:16:29 2021

@author: weissmce
"""
import pandas as pd
import cv2
import os
import numpy
import pickle


directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\\Both\\'

img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\\Both\\horse1-000.png')
img1 = img1.reshape(-1, 30, 30, 3)
validationPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    img = img.reshape(-1, 30, 30, 3)
    validationPixelVals += img




directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\Both\\'

img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\Both\\horse01-0.png')
img1 = img1.reshape(-1, 30, 30, 3)
trainingPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    img = img.reshape(-1, 30, 30, 3)
    trainingPixelVals += img



pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\ValidationPixel.pickle", "wb")
pickle.dump(validationPixelVals, pickle_out1)
pickle_out1.close()


pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\TrainingPixel.pickle", "wb")
pickle.dump(trainingPixelVals, pickle_out1)
pickle_out1.close()
