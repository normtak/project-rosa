# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:42:15 2020

@author: Stan
"""
import sys
sys.path.append('.\modules')
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import mape

#import Keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

df = pd.read_excel(r'C:\workdir\rosabella\project-rosa\data\sales.xlsx')
df.columns = ['date_time', 'item_name', 'id', 'sales_qtt']
df = df.drop(columns=['item_name', 'id'])

#Transform from TS to supervised
df_daily_total = df.groupby('date_time').sum()
df_weekly_total = df_daily_total.resample('W').sum()
df_supervised = df_weekly_total.copy()
lags = 2
for lag in range(1, lags+1):
    col_name = 'lag_' + str(lag)
    df_supervised[col_name] = df_supervised['sales_qtt'].shift(lag)
df_supervised = df_supervised.dropna()


#Splitting and scaling data
df_model = df_supervised.copy()
test_size = 5
train_set, test_set = df_model.values, df_model[-test_size:].values
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
model.fit(X_train, y_train,  nb_epoch=1000, batch_size=1, verbose=1, shuffle=False)

#Predictions
y_pred = model.predict(X_test, batch_size=1)
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
pred_test_set = []
for index in range(0, len(y_pred)):
    pred_test_set.append(np.concatenate([y_pred[index], X_test[index]], axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#Evaluating
y_test_inv = test_set[:,0:1]
y_pred_inv = pred_test_set_inverted[:,0:1]
print('RMSE: {}'.format(mean_squared_error(y_test_inv, y_pred_inv, squared=False)))
print('MAE: {}'.format(mean_absolute_error(y_test_inv, y_pred_inv)))
print('MAPE: {}'.format(mape.mean_absolute_percentage_error(y_test_inv, y_pred_inv)))
