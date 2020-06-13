# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:20:56 2020

@author: Stan
"""

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



# #Transform from TS to supervised
# df_daily_total = df.groupby('date_time').sum()
# df_weekly_total = df_daily_total.resample('W').sum()
# df_supervised = df_weekly_total.copy()
# lags = 8
# formula = 'sales_qtt ~'
# ars = np.zeros(lags)
# for inc1 in range(1, lags+1):
#     df_supervised = df_weekly_total.copy()
#     for inc in range(1, inc1+1):
#         col_name = 'lag_' + str(inc)
#         df_supervised[col_name] = df_supervised['sales_qtt'].shift(inc)
#     df_supervised = df_supervised.dropna()
#     #Checking adj.R^2
#     if inc == 1:
#         formula = formula + ' lag_1'
#     else:
#         formula = formula + ' + lag_' + str(inc)
#     model = smf.ols(formula=formula, data=df_supervised)
#     model_fit = model.fit()  
#     adj_rsq = model_fit.rsquared_adj
#     ars[inc-1] = adj_rsq

# best_lag = np.argmax(ars) + 1
# print(str(best_lag) + ' : ' + str(ars[best_lag-1]))
# df_supervised = df_supervised.loc[:,:'lag_'+str(best_lag)]
