# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:42:15 2020

@author: Stan
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM

df = pd.read_excel(r'C:\workdir\rosabella\project-rosa\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns=['item_name', 'id'])

#Transform from TS to supervised
df_daily_total = df.groupby('date_time').sum()
df_supervised = df_daily_total.copy()
lags = 28
for inc in range(1,lags+1):
    col_name = 'lag_' + str(inc)
    df_supervised[col_name] = df_supervised['sales_qtt'].shift(inc)
df_supervised = df_supervised.dropna().reset_index()
#Checking adj.R^2
formula = 'sales_qtt ~ lag_1'
ars = np.zeros(lags)
for inc in range (2, lags+1):
    formula = formula + ' + lag_' + str(inc)
    model = smf.ols(formula=formula, data=df_supervised)
    model_fit = model.fit()  
    adj_rsq = model_fit.rsquared_adj
    ars[inc-1] = adj_rsq
best_lag = np.argmax(ars) + 1
print(str(best_lag) + ' : ' + str(ars[best_lag-1]))
df_supervised = df_supervised.loc[:,:'lag_'+str(best_lag)]

# #Transform from diff-TS to supervised
# df_daily_total = df.groupby('date_time').sum()
# df_daily_total_diff = df_daily_total.diff()
# df_daily_total_diff = df_daily_total_diff.dropna()
# df_daily_total_diff = df_daily_total_diff.rename(columns={'sales_qtt':'diff'})
# df_supervised = df_daily_total_diff.copy()
# # lags = len(df_supervised)/2
# lags = 60
# for inc in range(1, lags+1):
#     col_name = 'lag_' + str(inc)
#     df_supervised[col_name] = df_supervised['diff'].shift(inc)
# df_supervised = df_supervised.dropna().reset_index()
# #Checking goodness of fit
# formula = 'diff ~ lag_1'
# ars = []
# for inc in range (2, lags+1):
#     formula = formula + ' + lag_' + str(inc)
#     model = smf.ols(formula=formula, data=df_supervised)
#     model_fit = model.fit()  
#     adj_rsq = model_fit.rsquared_adj
#     ars.append(adj_rsq)


#Splitting and scaling data
df_model = df_supervised.drop(['date_time'], axis=1)
train_set, test_set = df_model[0:-21].values, df_model[-21:].values
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_set)

train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:,1:], train_set_scaled[:,0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:,1:], test_set_scaled[:,0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#LSTM model
model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train,  nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

y_pred = model.predict(X_test, batch_size=1)