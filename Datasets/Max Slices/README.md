####descripton of Max slices folder

****ALL OF THE FUNCTIONS AND FILES ARE BASED ON THE MAX SLICES ONLY

50Agree --> list of slice where 2 radiologists agree -- no zero agreement

AllAgree -> list of slices with 100% and 75% agreement 
	#Used as the reference truth for this project

LIDC_Max_Slices_SpliculationFeatures --> CSV with the max slice features but only the features needed
for this project --the other were deleted but are still in the original main file

LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings -- exactly what name says


LIDC_SMALL_MaxSlicePerNodule_inLineRatings --> CSV with max slices of only nodules with two or more radiologist ratings


Neg Pairs - List of tuples of all possible negative pairs from LIDC_SMALL

Pos Pairs- List of tuples of all possible positive pairs from LIDC_SMALL
   --Pairs are for the siamese network 

Testing - Set of all the LIDC data being run through the network as testing

Training - Set of all the LIDC data being run through the network as training sets

Two Ratings -- CSV file of all of the nodules with only two ratings from radiologists 