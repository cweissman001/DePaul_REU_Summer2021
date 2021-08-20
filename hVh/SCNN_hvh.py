# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:53:07 2021

@author: weissmce
"""


from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np  # linear alg
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, MinMaxScaler
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import pickle

epochs = 10

random.seed(25)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
      uses y_true as Y and y_pred as the Euclidean distance between dissimilar points
    '''

    margin = 1
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

   

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = input
    #maybe more filters than needed
    x = Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape,padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    x = Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape,padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    #34X34
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    #16 x 16
    #another layer would likely not make a difference
    x = Flatten()(x) #take out one of the dense layers
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    

    
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x) #square root of the sum of difference
    return Model(input, x)
   


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    #mean2 = y_pred.mean()
    pred = y_pred.ravel() < 0.5
   # print(y_pred.ravel())
    #print(pred)
    ###can be changed
    #print(y_pred.ravel())
    
    #print(pred)
    #print(pred) # pred = 1 if number falls below the threshold, 0 if it is above
    # if number is closer to 0, then there is less distance, and therefore it is in a similar category
    # if number is closer to 1, then there is more distance, and therefore it is in a different category
          #(makes sense as the values go beyond 1, meaning very far away very unlikely to be in same group)
    #print(y_true)  # 1 = True (same category) & 0 = False (different categories)
    #print(pred == y_true) 
    #print(np.mean(pred == y_true))
    return np.mean(pred == y_true) 


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
"""
filepath ='/content/drive/MyDrive/MedIx REU/Datasets/'

train = pd.read_csv(filepath + 'TrainPixel.csv')
test = pd.read_csv(filepath + 'TestPixel.csv')
train_labels = train["Binary Rating"]

train_ids = train['InstanceID']
train_images = train.drop(['InstanceID', 'Binary Rating'],  axis=1)

test_ids = test['InstanceID']
test_images = test.drop(['InstanceID', "Binary Rating"], axis=1)

# Reshape the data so the pixel vectors turn into arrays, and so the instance id's are also arrays
train_data = np.array(train_images).reshape(-1,71,71,1)  # -1? not sure why
train_ids = np.array(train_ids)
# load and reshape data
train_labels = np.array(train_labels)

test_data = np.array(test_images).reshape(-1, 71, 71,1)
test_ids = np.array(test_ids)

train = train.astype('float32')
test = test.astype('float32')
"""
input_shape = (300, 300, 3)
#train_data.shape[1:]
#train.shape[1:]


#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("loading pairs...")


##calling on postive and negative pairs
#train data
path = "/content/drive/MyDrive/MedIx REU/horseVhumans/"
pickle_in = open(path + "TrainPairs9.pickle", "rb")
trainPixel = pickle.load(pickle_in)
trainPixel = np.asarray(trainPixel)
tr_pairs = trainPixel.reshape(4000,2,300,300,3)

#train labels
pickle_in = open(path + "TrainLabels.pickle", "rb")
tr_y = pickle.load(pickle_in)
tr_y = np.array(tr_y)

#test data
pickle_in = open(path + "Validation.pickle", "rb")
testPixel = pickle.load(pickle_in)
testPixel = np.array(testPixel)
print(testPixel.shape)
te_pairs = testPixel.reshape(4000,2,300,300,3)

#test labels
pickle_in = open(path + "ValidationLabels.pickle", "rb")
te_y = pickle.load(pickle_in)
te_y = np.array(te_y)

print("pairs loaded")

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b]) #dont understand
#Ask rosalyne about 

model = Model([input_a, input_b], distance)  #think we are shaping our model 

model.load_weights("/content/drive/MyDrive/Saved Weights/weights.ckpt")
rms = RMSprop()
#metrics=['binary_accuracy', 'categorical_accuracy']
model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
    #'binary_crossentropy'
    #tfa.losses.TripletSemiHardLoss()
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))


# compute final accuracy on training and test sets
y_pred_tr_before = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
y_pred_tr = (y_pred_tr_before - y_pred_tr_before.min()) / (y_pred_tr_before.max() - y_pred_tr_before.min())


tr_acc = compute_accuracy(tr_y, y_pred_tr)


""" use for training run when validation is included
"""
y_pred_te_before = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
y_pred_te = (y_pred_te_before - y_pred_te_before.min()) / (y_pred_te_before.max() - y_pred_te_before.min())

te_acc = compute_accuracy(te_y, y_pred_te)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
base_network.save("/content/drive/MyDrive/MedIx REU/horseVhumans/my model")
model.save_weights("/content/drive/MyDrive/Saved Weights/weights.ckpt")
