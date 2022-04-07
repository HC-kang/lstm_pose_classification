import os

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM

from sklearn.model_selection import train_test_split

X = []  
Y = []
data_name = ['stop.txt', 'l_h_swing.txt', 'r_h_swing.txt', 'l_f_swing.txt', 'r_f_swing.txt']
num_of_timestep = 10 

def save_model(model):
    model_number = 1
    while True:
        if not os.path.isfile(f'./models/simple_lstm_model_{model_number:0>3d}.h5'):
            model.save(f'./models/simple_lstm_model_{model_number:0>3d}.h5')
            break
        else:
            model_number += 1


for idx in range(len(data_name)):
    data = pd.read_csv(os.path.join('./data/',data_name[idx]), index_col=0)
    dataset = data.values
    n_sample = len(dataset)
    for i in range(num_of_timestep, n_sample):
        X.append(dataset[i - num_of_timestep:i, :])
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
model.add(Dense(units=5, activation='softmax'))

model.summary()

model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))


save_model(model)
