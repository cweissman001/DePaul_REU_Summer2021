Horse Versus Humans dataset location

CreatePairslist --> py file to create pos and neg files from dataset

GetPairPixelVals --> Takes the pair lists and outputs file with the pixel vals of the pairs next to each other

PNGtoPixel --> takes in PNG and outputs the pixel arrays

SCNN_hvh --> network for horses and humans -->final strucutre in paper

hvh_Test --> file that loads in the saved model and runs all of the data through the uncertainty metric

negpPairs --negative training pairs

posPairs --> positive training pairs

TrainingPixel -->Pixels of all training images

TrainLabels -->trinaing file of neg and pos pairs (1,0,1, 0...)

TrainPairs? -->splits of the training data

Validation --> validtion pairs

ValidationLabels --> validation labels for network 

validationPix --> pixel values of all of the validation pngs

valiNegPairs --> validation negtaive pairs

valiPosPairs --> validation positive pairs

my model --> the saved model from the horses versus humans trained network
