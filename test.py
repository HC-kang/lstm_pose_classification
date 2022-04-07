import pandas as pd
import numpy as np

data = pd.read_csv('./data/stop.txt', index_col=0)
print(data.shape)
data = pd.read_csv('./data/head_swing.txt', index_col=0)
print(data.shape)
data = pd.read_csv('./data/l_f_swing.txt', index_col=0)
print(data.shape)
data = pd.read_csv('./data/r_f_swing.txt', index_col=0)
print(data.shape)
data = pd.read_csv('./data/l_h_swing.txt', index_col=0)
print(data.shape)
data = pd.read_csv('./data/r_h_swing.txt', index_col=0)
print(data.shape)

data

type(data)
type(data.values)
dataset = data.iloc[:, 1:]
dataset


data = pd.DataFrame(data)
data.shape

data.describe()
data
columns = data.columns

for col in columns:
    print(col)
    
data.T.describe()

data.describe()
data.iloc[:20, -20:]

import os

os.listdir('./data')

len(X)
X[0].shape

Y

X, y = np.array(X), np.array(Y)

import pandas as pd
a = [[1,2,3],[1,2,3]]
a = pd.DataFrame(a)
b = [[1,2,3],[1,2,3]]
b = pd.DataFrame(a)
c = [[1,2,3],[1,2,3]]
c = pd.DataFrame(a)

a.head(1)

lists = [a, b, c]

print(*lists)