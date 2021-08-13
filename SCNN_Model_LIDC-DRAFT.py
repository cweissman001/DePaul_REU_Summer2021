# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:40:33 2021

@author: weissmce
"""

from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np  # linear alg
#import tensorflow_addons as tfa
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

#num_classes = 10
epochs = 10

random.seed(205)

#physical_devices=tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[],True)


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

   
def create_pairs(train, string, length):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    if string == "Test":
        dfPos = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\TESTNegPairs.csv")
        dfNeg = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\TESTPosPairs.csv")
    
    else:
        dfPos = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\PosPairs.csv")
        dfNeg = pd.read_csv("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Max Slices\\NegPairs.csv")
        
    dfPos = dfPos.sample(frac = 1)
    dfNeg = dfNeg.sample(frac = 1)
    pairs = []
    labels = []
    length = length #based on input from the size of the testing and training sets
    print("create pairs running ...")
    

    for i in range(length):
        pos0 = dfPos.iloc[i][0]
        #print(pos0)
        pos1 = dfPos.iloc[i][1]
        neg0 = dfNeg.iloc[i][0]
        neg1 = dfNeg.iloc[i][1]
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
   
    print("create pairs DONE")
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = input
    x = Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape,padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x= tf.keras.layers.AveragePooling2D()(x)
    
    x = Flatten()(x)
    x = Dense(396, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
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

filepath ='C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\'

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

input_shape = train_data.shape[1:]
#train.shape[1:]


##loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("loading pairs...")

##calling on postive and negative pairs
trainPixel = np.load('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TrainPairs1.npy')
tr_pairs = trainPixel.reshape(40000,2,71,71,1)
tr_y = np.load('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TrainLabel1.npy')


testPixel = np.load('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestPairs.npy')
te_pairs = testPixel.reshape(12398,2,71,71,1)
te_y = np.load('C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Training Pairs CSV\\TestLabel.npy')

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

#model.load_weights("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Saved Weights\\weights.ckpt")
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
mean1 = y_pred_tr_before.mean()
y_pred_tr = (y_pred_tr_before - y_pred_tr_before.min()) / (y_pred_tr_before.max() - y_pred_tr_before.min())

#y_prob = model.predict_proba([tr_pairs[:, 0], tr_pairs[:, 1]]) --doesnt exisit 
#test_scores = model.evaluate(tr_y, y_pred_tr, verbose=2)

tr_acc = compute_accuracy(tr_y, y_pred_tr)


""" use for training run when validation is included
"""
y_pred_te_before = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
y_pred_te = (y_pred_te_before - y_pred_te_before.min()) / (y_pred_te_before.max() - y_pred_te_before.min())



te_acc = compute_accuracy(te_y, y_pred_te)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model.save_weights("C:\\Users\\weissmce\\Desktop\\Multi-LevelSimilarity2021\\MedIx REU\\Datasets\\Saved Weights\\weights.ckpt")
import matplotlib.pyplot as plt

# LOSS Learning curves
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history.history['val_accuracy']) + 1))
plt.plot(epochs, loss_values, color='blue', linewidth=2, label='Training loss')
plt.plot(epochs, val_loss_values, color='blue', linewidth=2, linestyle='--', label='Validation loss')
my_y_ticks = np.arange(0, 0.5, 0.1)
plt.yticks(my_y_ticks)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ACCURACY Learning Curves

history_dict = history.history
loss_values = history_dict['accuracy']
val_loss_values = history_dict['val_accuracy']
epochs = range(1, (len(history.history['accuracy']) + 1))
plt.plot(epochs, loss_values, color='orange', linewidth=2, label='Training Acc')
plt.plot(epochs, val_loss_values, color='orange', linewidth=2, linestyle='--', label='Validation Acc')
my_y_ticks = np.arange(0.7, 1, 0.05)
plt.yticks(my_y_ticks)
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
#used to print out the classification accuracy

y_testArr = pd.Series(y_test).array
##base_network1 = create_base_network(input_shape)
embeddings1 = base_network.predict(x_test)
embeddings = (embeddings1 - embeddings1.min()) / (embeddings1.max() - embeddings1.min())
neigh = KNeighborsClassifier(n_neighbors=2)   #--> it includes itself too
neigh.fit(embeddings, y_testArr)
y_pred_class = neigh.predict(embeddings)
accuracy_test = accuracy_score(y_true=y_testArr, y_pred=y_pred_class)
print(accuracy_test)
   

# Plot the representation learned from the siamese network

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2,random_state=0).fit_transform(embeddings1)


mnist_classes = ['T-shirt/top','Trouser','Pullover','Dress',
                  'Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'
                 ]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

plt.figure(figsize=(10,10))
for i in range(10):
    inds = np.where(y_test==i)[0]
    plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.5, color=colors[i])
plt.legend(mnist_classes)
plt.show()

## Running K mean clusters on this data

# example x = [[ 1.12031365,  5.75806083], [ 1.9263585 ,  4.15243012]]

km = KMeans(n_clusters=7, random_state=0) #init='random', n_init=10, max_iter=300, tol=1e-04, 
km.fit(X_embedded)
"""

"""
# NCA implementation
from sklearn.neighbors import NeighborhoodComponentsAnalysis
nca = NeighborhoodComponentsAnalysis(random_state=42)
nca.fit(new_x_test,y_test)
new_x_test = []
for im in x_test:
    new_im = im.ravel()
    new_x_test.append(new_im)
"""
"""
tSNE analysis
# image number, category, x val, y val
tsne_info = ()
"""
"""
PCA implementation 
from sklearn.decomposition import PCA 
pca = PCA(n_components=4).fit_transform(embeddings)
"""


