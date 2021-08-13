"# REU_SCNN" 
#my model --saved folder for ther final model of the SCNN format used on the LIDC data

#AssiningPairsPickle.py --> a program that takes in dicom pixel values and already created pairs list and creates postive and negative pairs alternating in one file to run through the network as training and validation data

#CreatePairs_forLIDC --> used to create postive and negative pairs (1 == spiculated) and (0==spiculated) for the SCNN. Outputs CSV files

#LIDC_Test -->file used to run the LIDC testing data to output the TSNE and clustering and unceratinty metric

#NormalizePred-Fastest --> final SCNN structure for the fashion mnist data

#SCNN_Model_LIDC ---> file used to train the SCNN on LIDC data
Sortby_Agreement -->py file that will sort the nodules from the LIDC by the number of radilogists agreeing on the spiculation rating
