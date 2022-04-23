import os

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM

from sklearn.model_selection import train_test_split

import utils
import config

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
    for i in range(config.num_of_timestep, n_sample):
        X.append(dataset[i - config.num_of_timestep:i, :])
        Y.append(idx)
        
X, y = np.array(X), np.array(Y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dense(units=NUM_CLASS, activation='softmax'))

model.summary()

model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


utils.save_model(model)
