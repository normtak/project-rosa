# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:54:38 2020

@author: Stan
"""

import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    #Adding epsilon to y_true to avoid division by zero
    y_true = y_true + np.finfo(float).eps
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    