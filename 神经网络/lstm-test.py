from pandas import read_csv
from numpy import concatenate
from datetime import datetime
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# load  data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


dataset = pd.read_csv('../Data/raw.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize  first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('../Data/pollution.csv')

from matplotlib import pyplot

dataset = read_csv('../Data/pollution.csv', header=0, index_col=0)
values = dataset.values
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()


# 本节，我们将调整一个 LSTM 模型以适合此预测问题。
# LSTM 数据准备
# 第一步是为 LSTM 模型准备污染数据集，这涉及将数据集用作监督学习问题以及输入变量归一化。
# 我们将监督学习问题设定为：根据上一个时间段的污染指数和天气条件，预测当前时刻（t）的污染情况。
# 这个表述简单直接，只是为了说明问题。你可以探索的一些替代方案包括：
# 根据过去一天的天气情况和污染状况，预测下一个小时的污染状况。
# 根据过去一天的天气情况和污染状况以及下一个小时的「预期」天气条件，预测下一个小时的污染状况。
# 我们可以使用之前博客中编写的 series_to_supervised（）函数来转换数据集：
# 如何用 Python 将时间序列问题转换为监督学习问题（https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/）
# 首先加载「pollution.csv」数据集。给风速特征打上标注（整型编码）。如果你再深入一点就会发现，整形编码可以进一步进行一位有效编码（one-hot encoding）。
# 接下来，所有特征都被归一化，然后数据集转换成监督学习问题。之后，删除要预测的时刻（t）的天气变量。
# 完整的代码列表如下。

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('pollution1.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# print(reframed.head())

# 运行上例打印转换后的数据集的前 5 行。我们可以看到 8 个输入变量（输入序列）和 1 个输出变量（当前的污染水平）

# 这个数据准备过程很简单，我们可以深入了解更多相关知识，包括：
# 对风速进行一位有效编码
# 用差值和季节性调整使所有序列数据恒定
# 提供超过 1 小时的输入时间步长
# 最后也可能是最重要的一点，在学习序列预测问题时，LSTM 通过时间步进行反向传播。

# 定义和拟合模型
# 在本节中，我们将拟合多变量输入数据的 LSTM 模型。
# 首先，我们必须将准备好的数据集分成训练集和测试集。为了加快此次讲解的模型训练，我们将仅使用第一年的数据来拟合模型，然后用其余 4 年的数据进行评估。
# 下面的示例将数据集分成训练集和测试集，然后将训练集和测试集分别分成输入和输出变量。最后，将输入（X）重构为 LSTM 预期的 3D 格式，即 [样本，时间步，特征]。

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 运行此示例输出训练数据的维度，并通过测试约 9K 小时的数据对输入和输出集合进行训练，约 35K 小时的数据进行测试。

# 我们现在可以定义和拟合 LSTM 模型了。
# 我们将在第一个隐藏层中定义具有 50 个神经元的 LSTM，在输出层中定义 1 个用于预测污染的神经元。输入数据维度将是 1 个具有 8 个特征的时间步长。
# 我们将使用平均绝对误差（MAE）损失函数和高效的随机梯度下降的 Adam 版本。
# 该模型将适用于 50 个 epoch，批大小为 72 的训练。请记住，每个批结束时，Keras 中的 LSTM 的内部状态都将重置，因此内部状态是天数的函数可能有所帮助（试着证明它）。
# 最后，我们通过在 fit（）函数中设置 validation_data 参数来跟踪训练过程中的训练和测试损失，并在运行结束时绘制训练和测试损失图。

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# 评估模型
# 模型拟合后，我们可以预测整个测试数据集。
# 我们将预测与测试数据集相结合，并调整测试数据集的规模。我们还用预期的污染指数来调整测试数据集的规模。
# 通过初始预测值和实际值，我们可以计算模型的误差分数。在这种情况下，我们可以计算出与变量相同的单元误差的均方根误差（RMSE）。

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
# print(inv_y, inv_yhat)
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE:%.3f' % rmse)

# 运行示例首先创建一幅图，显示训练中的训练和测试损失。
# 有趣的是，我们可以看到测试损失低于训练损失。该模型可能过度拟合训练数据。在训练过程中测绘 RMSE 可能会使问题明朗。
# 在每个训练 epoch 结束时输出训练和测试的损失。在运行结束后，输出该模型对测试数据集的最终 RMSE。我们可以看到，该模型取得了不错的 RMSE——3.836，
# 这显著低于用持久模型（persistence model）得到的 RMSE（30）。
