# -*- encoding: utf-8 -*-
"""
@File    : Prophet_test.py
@Time    : 2019-04-12 22:10
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
df = pd.read_csv('../Data/吉林省.csv')

df['Month'] = pd.DatetimeIndex(df['Month'])
df = df.rename(columns={'Month': 'ds',
                        'jilin': 'y'})

ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Number of ShouDianLiang')
ax.set_xlabel('Date')

# # set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95, yearly_seasonality=True,weekly_seasonality=True)
my_model.fit(df)

future_dates = my_model.make_future_dataframe(periods=0, freq='MS')
forecast = my_model.predict(future_dates)

# Python
# from fbprophet.diagnostics import cross_validation
# df_cv = cross_validation(my_model,horizon = '48 days')
# print(df_cv)

# my_model.plot(forecast, uncertainty=True)
# my_model.plot_components(forecast)
# plt.show()

import numpy as np

s=np.mean(abs(1-forecast['yhat'][48:]/df['y'][48:]))
print(forecast['yhat'][48:])
print(s)
