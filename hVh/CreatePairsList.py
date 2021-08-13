# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:07:48 2021

@author: weissmce
"""
import cv2
import os
import numpy
#img = cv2.imread('C:\\Users\\weissmce\\Desktop\\GitREPO\\horse-or-human\\horse-or-human\\train\horses\\horse01-0.png') 
#img2 = img.reshape(1, 90000, 3)


Negpairs = [];
Pospairs = [];

directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\horse-or-human\\validation\horses\\'
directory1 = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\horse-or-human\\validation\humans\\'

#directory = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\horse-or-human\\train\horses\\'  for training
#directory1 = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\horse-or-human\\horse-or-human\\train\humans\\'
for filename in os.listdir(directory):
    for filename1 in os.listdir(directory1):
        Negpairs += [(filename, filename1)]
      
import random
random.shuffle(Negpairs)
output_file1 = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\valiNegPairs.csv'
numpy.savetxt(output_file1, Negpairs, fmt= '%s', delimiter=',')       
       
        
for filename in os.listdir(directory):
    for filename1 in os.listdir(directory):
        Pospairs += [(filename, filename1)]



for filename in os.listdir(directory1):
    for filename1 in os.listdir(directory1):
        Pospairs += [(filename, filename1)]
    
  
random.shuffle(Pospairs)
output_file1 = 'C:\\Users\\weissmce\\Desktop\\GitREPO\\hVh\\valiPosPairs.csv'
numpy.savetxt(output_file1, Pospairs, fmt= '%s', delimiter=',')  
        
"""            
output_file1 = 'C:\\Users\\weissmce\\GitREPO\\hVh\\posPairs'
numpy.savetxt(output_file1, Pospairs, fmt= '%s', delimiter=',')

output_file1 = 'C:\\Users\\weissmce\\GitREPO\\hVh\\negPairs'
numpy.savetxt(output_file1, Negpairs, fmt= '%s', delimiter=',')
img1 = cv2.imread(directory + filename)
        img1 = img1.reshape(1, 90000, 3)
        img2 = cv2.imread(directory1 + filename1) 
        img2 = img2.reshape(1, 90000, 3)
"""
