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


directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\\horses\\'
count = 0
img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\\horses\\horse1-000.png')
img1 = img1.reshape(-1, 30, 30, 3)
validationPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    img = img.reshape(-1, 30, 30, 3)
    count = count +1
    validationPixelVals += img




directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\humans\\'

img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\validation\\humans\\valhuman01-00.png')
#img1 = img1.reshape(-1, 30, 30, 3)
trainingPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    #img = img.reshape(-1, 30, 30, 3)
    trainingPixelVals += img


"""
pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\ValidationPixel1.pickle", "wb")
pickle.dump(validationPixelVals, pickle_out1)
pickle_out1.close()


pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\ValidationPixel2.pickle", "wb")
pickle.dump(trainingPixelVals, pickle_out1)
pickle_out1.close()
"""