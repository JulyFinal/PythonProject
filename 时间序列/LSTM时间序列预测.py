import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

data = pd.read_csv('../Data/AirPassengers.csv', usecols=[1], engine='python', skipfooter=3)
data_set = data.values

data_set = data_set.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
data_set = scaler.fit_transform(data_set)

n_steps = 3
n_features = 1

train_data_x = []
train_data_y = []

for i in range(data_set.shape[0] - n_steps):
    train_data_x.append(data_set[i:i + n_steps])
    train_data_y.append(data_set[i + n_steps])

train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)

train_data_x = np.reshape(train_data_x, (train_data_x.shape[0], train_data_x.shape[1], 1))

seed = 77
np.random.seed(seed)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_data_x, train_data_y, epochs=100, batch_size=1, verbose=2)

train_predict = model.predict(train_data_x)
print(train_data_x[-1])
for i in range(1, n_steps + 1):

    if i == 1:
        insert_data = np.array(
            np.concatenate((train_data_y[len(train_data_y) - 2:], train_predict[-i]), axis=None).reshape(
                (1, train_data_x.shape[1], 1)))
        train_predict1 = model.predict(insert_data)
        train_predict = np.append(train_predict, train_predict1)

    elif i == 2:
        insert_data = np.array(
            np.concatenate((train_data_y[-1], train_predict[-i:]), axis=None).reshape((1, train_data_x.shape[1], 1)))
        train_predict1 = model.predict(insert_data)
        train_predict = np.append(train_predict, train_predict1)

    elif i == 3:
        insert_data = np.array(train_predict[-i:]).reshape((1, train_data_x.shape[1], 1))
        train_predict1 = model.predict(insert_data)
        train_predict = np.append(train_predict, train_predict1)

    else:
        insert_data = np.array(train_predict[-i:-i + 3]).reshape((1, train_data_x.shape[1], 1))
        train_predict1 = model.predict(insert_data)
        train_predict = np.append(train_predict, train_predict1)

print(train_predict[-5:])
plt.plot(train_data_y)
plt.plot(train_predict[1:])
plt.show()
