# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:04:18 2021

@author: weissmce
"""

#go through csv file and only take out the lines where all of the radiologists agree


import numpy as np
import pandas as pd
import csv
   
allSame = []
threeFourths = []
oneHalf = []
twoRatingsorLess = []
noRatings = []
allDifferent = []
aggloSAME = []
agglo75 = []
list2 = []
df = pd.read_csv('C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\Max Slices\\LIDC_SMALL_MaxSlicePerNodule_inLineRatings.csv')

#print(df.InstanceID[1])
length = 2688 #length of the input file
for x in range(length):
   
   sp1 = df["Spiculation_1"][x]
   sp2 = df["Spiculation_2"][x]
   sp3 = df["Spiculation_3"][x]
   sp4 = df["Spiculation_4"][x]
   
   list2.append(sp1)
   list2.append(sp2)
   list2.append(sp3)
   list2.append(sp4)
   zeroCount = list2.count(0)
   
   if(zeroCount == 4):
       noRatings.append(x)
   elif(zeroCount == 3):
       twoRatingsorLess.append(x)
   elif(zeroCount == 2):
       twoRatingsorLess.append(x)
   
   elif (sp1 == sp2 == sp3 == sp4):
       allSame.append(x)
       aggloSAME.append(sp1)
  
   elif ((sp1 == sp2 == sp3) or (sp1 == sp2 == sp4) or (sp1 == sp3 ==sp4) or (sp2 == sp3 == sp4)):
       threeFourths.append(x)
       agglo75.append(round((sp1+sp2+sp3+sp4)/4))
 
   elif(sp1 == sp2 or sp1 ==sp3 or sp1 == sp4 or sp2 == sp3 or sp2 == sp4 or sp3 == sp4):
       oneHalf.append(x)
            
   else:
       allDifferent.append(x)
       
   list2.clear()   

file_path0 = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\TwoRatings.csv"
twoSpic = []
twoSpic.append(df.columns)
for x in twoRatingsorLess:
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    twoSpic.append(arrRow)

np.savetxt(file_path0, twoSpic, fmt='%s', delimiter=',') 

file_path = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\NoRatings.csv"
noRate = []
noRate.append(df.columns)
for x in noRatings:
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    noRate.append(arrRow)

np.savetxt(file_path, noRate, fmt='%s', delimiter=',')    

file_path1 = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\AllAgree.csv"
allsameVal = []
allsameVal.append(df.columns)
for x in allSame:
    aggloVal = aggloSAME[allSame.index(x)]
    df["Agglomeration"][x] = aggloVal
    if(aggloVal == 1):
        df["Binary Rating"][x] = 0
    else:
        df["Binary Rating"][x] = 1
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    allsameVal.append(arrRow)
    
   
np.savetxt(file_path1, allsameVal, fmt='%s', delimiter=',')

file_path2 = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\75agree.csv"
three4ths = []
three4ths.append(df.columns)
for x in threeFourths:
    aggloVal = agglo75[threeFourths.index(x)]
    df["Agglomeration"][x] = aggloVal
    if(aggloVal == 1):
        df["Binary Rating"][x] = 0
    else:
        df["Binary Rating"][x] = 1
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    three4ths.append(arrRow)
    
  
np.savetxt(file_path2, three4ths, fmt='%s', delimiter=',')

file_path3 = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\50agree.csv"
halfAndHalf = []
halfAndHalf.append(df.columns)
for x in oneHalf:
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    halfAndHalf.append(arrRow)
   
 
np.savetxt(file_path3, halfAndHalf, fmt='%s', delimiter=',')

file_path4 = "C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\AgreementFiles\\NoneAgree.csv"
noAgree = []
noAgree.append(df.columns)
for x in allDifferent:
    arrRow = df.iloc[x]
    arrRow = arrRow.to_numpy()
    noAgree.append(arrRow)

    
np.savetxt(file_path4, noAgree, fmt='%s', delimiter=',')

###DO THE SAME FOR THE OTHER FILES
#write to C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\AllAgree.csv       
#for x in range(1,2687):
    #print(df[x])