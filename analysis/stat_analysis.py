# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:23:20 2020

@author: Stan
"""

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

df = pd.read_excel('.\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns=['item_name', 'id'])

#Check daily sales for randow-walk and autocorrelation
df_daily_total = df.groupby('date_time').sum()
df_daily_total.plot()
r = adfuller(df_daily_total['sales_qtt'])
print("The p-value for the ADF test is ", r[1])
plot_acf(df_daily_total, lags=30, alpha=0.05)
plot_pacf(df_daily_total, lags=30, alpha=0.05)

df_daily_total = df_daily_total.asfreq('D').fillna(0)
mod_ARMA = ARMA(df_daily_total, order=(2,0))
result_ARMA = mod_ARMA.fit()
result_ARMA.plot_predict(start='2020-04-01', end='2020-05-15')

#Check weekly sales for randow-walk and autocorrelation
df_daily_total = df.groupby('date_time').sum()
df_weekly_total = df_daily_total.resample('W').sum().fillna(0)
df_weekly_total.plot()
r = adfuller(df_weekly_total['sales_qtt'])
print("The p-value for the ADF test is ", r[1])
plot_acf(df_weekly_total,
         lags=int(len(df_weekly_total)/2),
         alpha=0.05)
plot_pacf(df_weekly_total,
         lags=int(len(df_weekly_total)/2),
         alpha=0.05)

mod_ARMA_weekly = ARMA(df_weekly_total, order=(2,0))
result_ARMA_weekly = mod_ARMA_weekly.fit()
result_ARMA_weekly.plot_predict(start='2020-03', end='2020-05-24')

#Check weekly differences for randow-walk and autocorrelation
df_weekly_total_diff = df_weekly_total.diff()
df_weekly_total_diff = df_weekly_total_diff.dropna()
df_weekly_total_diff.plot()
r = adfuller(df_weekly_total_diff['sales_qtt'], maxlag=4)
print("The p-value for the ADF test is ", r[1])
plot_acf(df_weekly_total_diff,
         lags=int(len(df_weekly_total_diff)/2),
         alpha=0.05)
plot_pacf(df_weekly_total_diff,
         lags=int(len(df_weekly_total_diff)/2),
         alpha=0.05)

mod_ARMA_weekly_diff = ARMA(df_weekly_total_diff, order=(2,0))
result_ARMA_weekly_diff = mod_ARMA_weekly_diff.fit()
result_ARMA_weekly_diff.plot_predict(start='2020-03', end='2020-07')


