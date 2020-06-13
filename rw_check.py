# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:23:20 2020

@author: Stan
"""

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

df = pd.read_excel(r'C:\workdir\rosabella\project-rosa\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns=['item_name', 'id'])

#Check daily sales for randow-walk and autocorrelation
df_daily_total = df.groupby('date_time').sum()
df_daily_total.plot()
r = adfuller(df_daily_total['sales_qtt'])
print("The p-value for the ADF test is ", r[1])
plot_acf(df_daily_total, lags=30, alpha=0.05)

#Check weekly sales for randow-walk and autocorrelation
df_daily_total = df.groupby('date_time').sum()
df_weekly_total = df_daily_total.resample('W').sum()
df_weekly_total.plot()
r = adfuller(df_weekly_total['sales_qtt'])
print("The p-value for the ADF test is ", r[1])
plot_acf(df_weekly_total,
         lags=int(len(df_weekly_total)/2),
         alpha=0.05)

#Check weekly differences for randow-walk and autocorrelation
df_weekly_total_diff = df_weekly_total.diff()
df_weekly_total_diff = df_weekly_total_diff.dropna()
df_weekly_total_diff.plot()
r = adfuller(df_weekly_total_diff['sales_qtt'], maxlag=4)
print("The p-value for the ADF test is ", r[1])
plot_acf(df_weekly_total_diff,
         lags=int(len(df_weekly_total_diff)/2),
         alpha=0.05)


