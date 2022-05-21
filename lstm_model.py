import os

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.model_selection import train_test_split

import utils
import config

# labels = config.labels
# NUM_CLASS = len(labels)
labels = os.listdir('./data/0516/yellow/label')
# heading, shoot, dribble, trapping + noball, (no man: 생략)
CLASS = {'heading': 0, 'shoot': 1, 'dribble': 2, 'trapping': 3, 'noball': 4}
NUM_CLASS = 5

X = []
Y = []

print('=================================================================')
print('Start Training with ')
print(*labels)
print('=================================================================')

for label in labels:
    data = pd.read_csv(os.path.join(
        '.', 'data', '0516', 'yellow', 'label', label), index_col=0)
    dataset = data.values
    n_sample = len(dataset)
    for i in range(config.num_of_timestep, n_sample):
        X.append(dataset[i - config.num_of_timestep:i, :])
        Y.append(CLASS[label.split('_')[1]])
# for idx in range(len(labels)):
#     data = pd.read_csv(os.path.join(
#         '.', 'data', labels[idx]+'.txt'), index_col=0)
#     dataset = data.values
#     n_sample = len(dataset)
#     for i in range(config.num_of_timestep, n_sample):
#         X.append(dataset[i - config.num_of_timestep:i, :])
#         Y.append(idx)

X, y = np.array(X), np.array(Y)
print(X.shape, y.shape)
print(X[0].shape, y[0].shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=256, return_sequences=True,
          input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dense(units=NUM_CLASS, activation='softmax'))

model.summary()

model.compile(optimizer="adam", metrics=[
              'accuracy'], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))


utils.save_model(model)
