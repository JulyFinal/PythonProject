# -*- encoding: utf-8 -*-
"""
@File    : AutoARIMA.py
@Time    : 2019-04-26 14:10
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""

# import pmdarima as pm
# from pmdarima.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load/split your data
# y = pm.datasets.load_wineind()
# train, test = train_test_split(y, train_size=150)
#
# # Fit your model
# model = pm.auto_arima(train, seasonal=True, m=12)
#
# # make your forecasts
# forecasts = model.predict(test.shape[0])  # predict N steps into the future
#
# # Visualize the forecasts (blue=train, green=forecasts)
# x = np.arange(y.shape[0])
# plt.plot(x[:150], train, c='blue')
# plt.plot(x[150:], forecasts, c='green')
# plt.show()

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pickle

# Load/split your data
y = pm.datasets.load_sunspots()
train, test = train_test_split(y, train_size=2700)

# Define and fit your pipeline
pipeline = Pipeline([
    ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),  # lmbda2 avoids negative values
    ('arima', pm.AutoARIMA(seasonal=True, m=12,
                           suppress_warnings=True,
                           trace=True))
])

pipeline.fit(train)

print(pipeline.summary)
# Serialize your model just like you would in scikit:
with open('model.pkl', 'wb') as pkl:
    pickle.dump(pipeline, pkl)

# Load it and make predictions seamlessly:
with open('model.pkl', 'rb') as pkl:
    mod = pickle.load(pkl)
    print(mod.predict(15))
# [25.20580375 25.05573898 24.4263037  23.56766793 22.67463049 21.82231043
# 21.04061069 20.33693017 19.70906027 19.1509862  18.6555793  18.21577243
# 17.8250318  17.47750614 17.16803394]