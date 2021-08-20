# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:17:42 2021

@author: weissmce
"""

import dicom
import dicom_numpy
import pydicom as dicom1
import matplotlib.pyplot as plt
import numpy
import csv 
import skimage
from skimage.transform import resize
import os.path
from os import path

import pandas as pd

length = 2688
#length = 20
file_path = "C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\nodules_MaxSlice_Subset\\"
#put where the folder of all dicom images are
#file_path = "C:\\Users\\weissmce\\Desktop\\DCM Images\\"
# resize to new size
IMG_PX_SIZE1 = 71
IMG_PX_SIZE2 = 71
#max size of both dimensions
#pixelLabel = numpy.empty((16128), dtype = object)
#length = 16128
#for x in range(length):
    #(x)
    #label = "pixel" + str(x)
    #numpy.append(pixelLabel, label)
    
df = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\LIDC_SMALL_MaxSlicePerNodule_inLineRatings.csv")


arr4 = df["InstanceID"]
ds= dicom1.dcmread("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\nodules_MaxSlice_Subset\\4.dcm")
resized_img1 = resize(ds.pixel_array, (IMG_PX_SIZE1, IMG_PX_SIZE2), anti_aliasing=True)
arr = resized_img1.flatten()
#arr = pixelLabel

for i in arr4:
    print(i)
    #print(arr4[i])
    index = str(i) + ".dcm"
    if path.exists((file_path + index)):
        ds1= dicom1.dcmread(file_path + index)
    #print(ds1)
        resized_img1 = resize(ds1.pixel_array, (IMG_PX_SIZE1, IMG_PX_SIZE2), anti_aliasing=True)
    #plt.imshow(resized_img1)
   
        newArr = resized_img1.flatten()
        numpy.insert(newArr,0,i)
    #print(newArr)
        arr = numpy.vstack((arr, newArr))
    #print(arr)
  

output_file = 'C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\LIDC_Max_Slice_Pixel_ValuesSMALL.csv'
#where to save the csv file
numpy.savetxt(output_file, arr, fmt= "%10.5f", delimiter=',')
print("done")

"""
arr = ds1.pixel_array
arr1 = arr.flatten()
#print(arr1)

ds2 = dicom1.dcmread("2.dcm")
arr2 = ds2.pixel_array
arr2 = arr2.flatten()


#print(arr2.size)
arr1 = numpy.array(arr1)
arr2 = numpy.array(arr2)



#cat = numpy.vstack((arr1, arr2))
# save the dataframe as a csv file
#DF.to_csv("sample.csv")



"""
"""
with open('sample.csv', 'w') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(arr2)
"""

