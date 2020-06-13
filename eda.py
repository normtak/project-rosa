# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:23:20 2020

@author: Stan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os
os.chdir(r'C:\workdir\rosabella\project-rosa')

sns.set(style='dark')
#sns.set_palette(sns.cubehelix_palette())

df = pd.read_excel(r'C:\workdir\rosabella\project-rosa\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns='item_name')
df['id'] = df['id'].astype('category')
df.set_index('date_time', inplace = True)

#EDA
df.head()
df.info()
df.describe()
df['id'].value_counts()
df['sales_qtt'].plot(kind='hist', cumulative=False)
df['sales_qtt'].plot(kind='hist', cumulative=True)
plt.close()

#ECDF - daily sales
x = np.sort(df['sales_qtt'])
y = np.arange(1, len(x) + 1)/len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Quantity sold (daily)')
_ = plt.ylabel('Cumulative probability')
plt.xticks(range(0, x.max()+1))
plt.margins(0.02)
plt.show()
plt.savefig(r'charts\daily_sales_ecdf.png')
plt.close()

#ECDF - weekly sales
df_weekly = df.groupby('id').resample('W').sum()
df_weekly.reset_index(inplace=True)
df_weekly.set_index('date_time', inplace = True)
df_weekly = df_weekly[df_weekly['sales_qtt'] != 0]

x = np.sort(df_weekly['sales_qtt'])
y = np.arange(1, len(x) + 1)/len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Quantity sold (weekly)')
_ = plt.ylabel('Cumulative probability')
plt.margins(0.02)
plt.show()
plt.savefig(r'charts\weekly_sales_ecdf.png')
plt.close()

#Sales by day of week
df_dow = df.copy()
df_dow['dow'] = df_dow.index.strftime('%a')
df_dow.reset_index(inplace=True)
df_dow = df_dow.groupby('dow').sum()
dows = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df_dow = df_dow.reindex(dows)
df_dow.plot()
plt.savefig(r'charts\sales_by_dow.png')
plt.close()

#Pareto chart
df_top = df.groupby('id').sum()
df_top.sort_values('sales_qtt', ascending=False, inplace=True)
df_top['sales_pct_cum'] = df_top['sales_qtt'].cumsum()/df_top['sales_qtt'].sum()*100
df_top = df_top.reset_index()
df_top['id'] = df_top['id'].astype(str)

fig, ax = plt.subplots()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax.bar(df_top['id'], df_top['sales_qtt'], color="C0")
ax2 = ax.twinx()
ax2.plot(df_top['id'], df_top['sales_pct_cum'], color="C1", marker='.')
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.axhline(y=80, color='r')
ax2.axhline(y=50, color='C1')
plt.show()
plt.savefig(r'charts\pareto.png')
plt.close()


#Aggregated weekly sales decomposition
df_weekly_total = df.resample('W').sum()
weekly_decompose = sm.tsa.seasonal_decompose(df_weekly_total, model='additive', period=4)
fig1 = weekly_decompose.plot()
df_weekly_total.hist(bins=20)

#Aggregated daily sales decomposition
df_daily_total = df.resample('D').sum()
daily_decompose = sm.tsa.seasonal_decompose(df_daily_total, model='additive', period=28)
fig2 = daily_decompose.plot()
df_daily_total.hist(bins=20)



