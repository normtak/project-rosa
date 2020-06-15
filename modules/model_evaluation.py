# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:28:38 2020

@author: Stan
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def evaluate_model(y_true, y_pred):
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print('RMSE: {}'.format(rmse))
    print('MAE: {}'.format(mae))
    print('MAPE: {}'.format(mape))
    

def mean_absolute_percentage_error(y_true, y_pred):
    #Adding epsilon to y_true to avoid division by zero
    y_true = y_true + np.finfo(float).eps
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100