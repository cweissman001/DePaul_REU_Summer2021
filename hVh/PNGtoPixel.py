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

"""
directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\horses\\'
count = 0
img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\horses\\horse01-0.png')
#img1 = img1.reshape(-1, 300, 300, 3)
img1 = img1.flatten()
validationPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    #img = img.reshape(-1, 300, 300, 3)
    arr = img.flatten()
    count = count +1
    validationPixelVals = numpy.vstack((validationPixelVals, arr))
    #validationPixelVals += arr

output_file = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\TrainingPixel0.csv'
#where to save the csv file
numpy.savetxt(output_file, validationPixelVals, fmt= "%10.5f", delimiter=',')
"""

directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\humans\\'

img1 = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\train\\humans\\human01-08.png')
#img1 = img1.reshape(-1, 300, 300, 3)
img1 = img1.flatten()
trainingPixelVals = img1
for filename in os.listdir(directory):
    img = cv2.imread(directory + filename)
    #img = img.reshape(-1, 300, 300, 3)
    arr = img.flatten()
    trainingPixelVals = numpy.vstack((trainingPixelVals, arr))

output_file = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\TrainingPixel2.csv'
#where to save the csv file
numpy.savetxt(output_file, trainingPixelVals, fmt= "%10.5f", delimiter=',')

"""
pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\ValidationPixel1.pickle", "wb")
pickle.dump(validationPixelVals, pickle_out1)
pickle_out1.close()


pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\hvh\\ValidationPixel2.pickle", "wb")
pickle.dump(trainingPixelVals, pickle_out1)
pickle_out1.close()
"""