# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:23:59 2021

@author: weissmce
"""
import pandas as pd
import numpy as np
import pickle

filepath ='C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\'

#train = pd.read_csv(filepath + 'TrainPixel.csv')
train = pd.read_csv(filepath + 'TestPixel.csv')   
    
dfPos = pd.read_csv("C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\Max Slices\\TESTPosPairs.csv")
dfNeg = pd.read_csv("C:\\Users\\weissmce\\Desktop\\GitREPO\Datasets\\Max Slices\\TESTNegPairs.csv")
         
    
##takes all of the pairs and puts the pixels values of the pairs in teh sameline in a csv to allow to load to model
pairs = []
labels = []
instanceID = []
#max lnegth is 222600
print("create pairs running ...")
    
#my computer would only let me run 20,000 before the memory filled up 
for i in range(58320):
    pos0 = dfPos.iloc[i][0]
    #print(pos0)
    pos1 = dfPos.iloc[i][1]
    neg0 = dfNeg.iloc[i][0]
    neg1 = dfNeg.iloc[i][1]
    print(i)
    instanceID += [(pos0, pos1), (neg0, neg1)]
    #instanceID += [neg0, neg1]
    # additionTest = (train.loc[train['InstanceID']==pos0])
    pos0Arr = (train.loc[train['InstanceID']==pos0])
    #print(pos0Arr)
    pos1Arr = (train.loc[train['InstanceID']==pos1])
    neg0Arr = (train.loc[train['InstanceID']==neg0])
       
    neg1Arr = (train.loc[train['InstanceID']==neg1])
         
    pos0Arr = pos0Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
        #print(pos0Arr)
    pos1Arr = pos1Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    neg0Arr = neg0Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    neg1Arr = neg1Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
        
    pos0Arr = pos0Arr.to_numpy().reshape(-1, 71, 71)
    pos1Arr = pos1Arr.to_numpy().reshape(-1, 71, 71)
    neg0Arr = neg0Arr.to_numpy().reshape(-1, 71, 71)
    neg1Arr = neg1Arr.to_numpy().reshape(-1, 71, 71)
        
        
    POS0 = pos0Arr[0]
    POS1 = pos1Arr[0]
    NEG0 = neg0Arr[0]
    NEG1 = neg1Arr[0]
    pairs += [(POS0, POS1)]
    pairs += [(NEG0, NEG1)]
    labels += [1, 0]
    
#filepath ='C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\'

pickle_out = open("C:\\Users\\weissmce\\Desktop\\GitRepo\\Datasets\\TestLabels.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()
pickle_out1 = open("C:\\Users\\weissmce\\Desktop\\GitREPO\\Datasets\\TestPairs.pickle", "wb")
pickle.dump(pairs, pickle_out1)
pickle_out1.close()

#pickle_out2 = open("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestID", "wb")
#pickle.dump(pairs, pickle_out2)
#pickle_out2.close()
#make sure the change the numnber at the end of the name to differentiate between files
#np.save('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestLabel',labels)
#np.save('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestPairs',pairs)
#np.save('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestID',instanceID)


"""        
for i in range(6199):
    pos0 = dfPosTEST.iloc[i][0]
        #print(pos0)
    pos1 = dfPosTEST.iloc[i][1]
    neg0 = dfNegTEST.iloc[i][0]
    neg1 = dfNegTEST.iloc[i][1]
       # additionTest = (train.loc[train['InstanceID']==pos0])
    pos0Arr = (test.loc[test['InstanceID']==pos0])
    pos1Arr = (test.loc[test['InstanceID']==pos1])
    neg0Arr = (test.loc[test['InstanceID']==neg0])  
    neg1Arr = (test.loc[test['InstanceID']==neg1]) 
    
    pos0Arr = pos0Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    pos1Arr = pos1Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    neg0Arr = neg0Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    neg1Arr = neg1Arr.drop(['InstanceID', 'Binary Rating'],  axis=1)
    pos0Arr = pos0Arr.to_numpy().reshape(-1, 126, 128)
      
    print(i)
    pos1Arr = pos1Arr.to_numpy().reshape(-1, 126, 128)
    neg0Arr = neg0Arr.to_numpy().reshape(-1, 126, 128)
    neg1Arr = neg1Arr.to_numpy().reshape(-1, 126, 128)
        
        
    POS0 = pos0Arr[0]
    POS1 = pos1Arr[0]
    NEG0 = neg0Arr[0]
    NEG1 = neg1Arr[0]
    pairs1 += [(POS0, POS1)]
    pairs1 += [(NEG0, NEG1)]
    labels1 += [1, 0]
        
filepath ='C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\'

pd.DataFrame(labels1).to_csv(filepath +"TestLabelPairs.csv")

pd.DataFrame(pairs1).to_csv(filepath +"TestPairs_Pixels.csv")
"""
print("DONE")