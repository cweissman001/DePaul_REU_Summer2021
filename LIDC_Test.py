from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import numpy as np  # linear alg
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import keras
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
import matplotlib.pyplot as plt

#num_classes = 10
epochs = 5

random.seed(205)

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
    #x = Dense(512, activation='relu')(x)
    
    
    
    #overfitting could be due to dense layer number
   
    
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


test = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestingVals50Agree.csv")    #50 agree
#test = pd.read_csv("/content/drive/MyDrive/MedIx REU/Datasets/TestingValsNOAgree.csv") #--no agree

test_ids = test['InstanceID']
test_images = test.drop(['InstanceID'],  axis=1)

test_data = np.array(test_images).reshape(-1,71,71,1) 
test_ids = np.array(test_ids)

test = test.astype('float32')

input_shape = test_data.shape[1:]

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)

#distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(processed_a) #dont understand
#Ask rosalyne about 

model = keras.models.load_model("/content/drive/MyDrive/MedIx REU/Datasets/my model" , compile = True, custom_objects={"contrastive_loss": contrastive_loss})
"""
Model(input_a) #think we are shaping our model 

model.load_weights("/content/drive/MyDrive/MedIx REU/Datasets/FINALweights.ckpt.index")
model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
"""
# compute final accuracy on training and test sets
y_pred_test_before = model.predict(test_data)
y_pred_test = (y_pred_test_before - y_pred_test_before.min()) / (y_pred_test_before.max() - y_pred_test_before.min())


from sklearn.manifold import TSNE

## Get the classification accuracy 
embeddings_raw = base_network.predict(test_data)
embeddings = (embeddings_raw - embeddings_raw.min()) / (embeddings_raw.max() - embeddings_raw.min())

X_embedded = TSNE(n_components=2,random_state=2).fit_transform(embeddings)
og_tsne = plt.figure(figsize=(10,10))
for i in range(len(X_embedded)):
    plt.scatter(X_embedded[i,0],X_embedded[i,1],alpha=0.5)

og_tsne.savefig('tsne')

print(X_embedded)





print("DONE")