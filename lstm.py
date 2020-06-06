# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:23:20 2020

@author: Stan
"""

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

df = pd.read_excel(r'C:\workdir\rosabella\project-rosa\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns=['item_name', 'id'])

df_daily_total = df.groupby('date_time').sum()
df_daily_total.plot()
r = adfuller(df_daily_total['sales_qtt'])
print("The p-value for the ADF test is ", r[1])
plot_acf(df_daily_total, lags=30, alpha=0.05)

df_daily_total_diff = df_daily_total.diff()
df_daily_total_diff.plot()
