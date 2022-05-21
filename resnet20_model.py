import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPool2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

import config
import utils


def Conv_layer1(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    shortcut = x
    
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    shortcut = x
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def Conv_layer2(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    shortcut = x
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', strides = 2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x1 = Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', strides = 2)(shortcut)
    x1 = BatchNormalization()(x1)
    x = Add()([x, x1])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def Conv_layer3(x):
    x = ZeroPadding2D(padding = (1,1))(x)
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', strides = 2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x1 = Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', strides = 2)(shortcut)
    x1 = BatchNormalization()(x1)
    x = Add()([x, x1])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


labels = config.labels
NUM_CLASS = len(labels)

X = []  
Y = []

print('=================================================================')
print('Start Training with ')
print(*labels)
print('=================================================================')

for idx in range(len(labels)):
    data = pd.read_csv(os.path.join('.','data',labels[idx]+'.txt'), index_col=0)
    dataset = data.values
    n_sample = len(dataset)
    for i in range(22, n_sample):
        X.append(np.append(dataset[i - 22:i, :], ([0]*102)).reshape(32,32,3))
        tmp = np.zeros(5)
        tmp[idx] = 1
        Y.append(tmp)

X, y = np.array(X), np.array(Y)

print(X.shape, y.shape)
print(X[0].shape, y[0].shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


inputs = Input(shape = (32, 32, 3), dtype = 'float32')
x = Conv_layer1(inputs)
x = Conv_layer2(x)
x = Conv_layer3(x)
x = GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(5, activation = 'softmax')(x)
resnet20 = tf.keras.models.Model(inputs, outputs)
resnet20.summary()


nadam = tf.keras.optimizers.Nadam(lr = 0.01)
resnet20.compile(optimizer = nadam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
resnet20.fit(X_train, y_train, batch_size = 128, epochs = 20, validation_split = 0.1)


utils.save_model(resnet20)
