This folder includes the datasets that were created in 2021 for a summer REU project 

nodules_MaxSlice_subset-----> includes the dicom images of all of the max slice nodule images. 

Dicom_Resize_Pixelval_toArray.py ----> a python script that takes in a folder of Dicom images, resized images to a set size and then turns all of 
the images into a pixel value array and writes them to the output csv file
   ###MAKE SURE TO CHANGE FILE PATHS WHEN USING##
   
LIDC_Max_Slice_Pixel_ValuesSMALL.csv --> the outpout of above script then the nodules_MaxSlice_subset is passed in and with other features deleted


LIDC_all_slices_inlineratings_NODUPS.csv --> all of the in line ratings for all of the slices but not included duplicates
		--10,000 ish slices


CreatePairs.py --> a script that takes in csv file including instance ID and created binary rating(1=spiculated 0 = not spiculated)
  and created a list of all possible pairs within the same class and all possible pairs from different classes

CreateRealTesting --> creating testing file for LIDC data without labels --->outputs a file with just pixel values and ID

nodule_MaxSlice_Subset ---. folder of all of the dicom images for the max slices 

AgglomerationStats -->file showing the breakdown of the agglomeration metric

SimiliarityDist --> Distribution of similarity score output when testing the SCNN model on LIDC

Test Pixel ---> all pixels values for the validation data

TrainPixel   --> all pixel values for testing data

TestingValsALL --> 100% and 75% agree together

TestingValsNOAgree --> cases with no agreement from radiologists

TestingVals50Agree --> 50% agreement from radiologist
