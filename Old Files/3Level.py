
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K


num_classes = 10
epochs = 10

random.seed(101)


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
    '''

    margin = 1
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

# 0-'T-shirt/top', 1- 'Trouser', 2-'Pullover', 3-'Dress', 4-'Coat',
# 5-'Sandal', 6-'Shirt', 7-'Sneaker', 8-'Bag', 9-'Ankle boot'

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''

    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]

            if (d == 0 and dn == 6) or (d == 0 and dn == 3):  # tshirt, shirt, dress
                labels += [1, 0.25]
            elif (d == 6 and dn == 4) or (d == 6 and dn == 2) or (d == 6 and dn == 3):  # shirt with coat, pullover, dress
                labels += [1, 0.25]
            elif (d == 4 and dn == 2):  # coat, pullover
                labels += [1, 0.25]
            else:
                labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# load and reshape data
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

train_labels = train['label']
train_images = train.drop(['label'], axis=1)
test = test.drop(['label'], axis=1)

train_data = np.array(train_images).reshape(-1, 28, 28)
train_label = np.array(train_labels)
test_x = np.array(test).reshape(-1, 28, 28)

# the data, split between train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, shuffle=True,
                                                    random_state=1)
print("Train data shape: {}.".format(x_train.shape))
print("Test data shape {}.".format(test_x.shape))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
test_x = test_x / 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]

tr_pairs, tr_y = create_pairs(x_train, digit_indices)
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

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
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()

model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# train
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                    batch_size=128,
                    epochs=epochs,
                    validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred_tr_before = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
y_pred_tr = (y_pred_tr_before - y_pred_tr_before.min()) / (y_pred_tr_before.max() - y_pred_tr_before.min())
tr_acc = compute_accuracy(tr_y, y_pred_tr)
y_pred_te_before = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
y_pred_te = (y_pred_te_before - y_pred_te_before.min()) / (y_pred_te_before.max() - y_pred_te_before.min())
te_acc = compute_accuracy(te_y, y_pred_te)

