# -*- encoding: utf-8 -*-
"""
@File    : Prophet.py
@Time    : 2019-04-12 13:11
@Author  : final
@Email   : julyfinal@outlook.com
@Software: PyCharm
"""


import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
plt.style.use('fivethirtyeight')
register_matplotlib_converters()
df = pd.read_csv('../Data/AirPassengers.csv')

df['Month'] = pd.DatetimeIndex(df['Month'])

df = df.rename(columns={'Month': 'ds',
                        'AirPassengers': 'y'})

# ax = df.set_index('ds').plot(figsize=(12, 8))
# ax.set_ylabel('Monthly Number of Airline Passengers')
# ax.set_xlabel('Date')

# plt.show()

# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.80,weekly_seasonality=True,yearly_seasonality=True)
my_model.fit(df)
future_dates = my_model.make_future_dataframe(periods=0, freq='MS')
forecast = my_model.predict(future_dates)
my_model.plot(forecast,
              uncertainty=True)

my_model.plot_components(forecast)
plt.show()

import numpy as np

from sklearn.metrics import mean_squared_error,mean_absolute_error
s=mean_absolute_error(df['y'][97:],forecast['yhat'][97:])
print(df['y'][97:])
print(s)