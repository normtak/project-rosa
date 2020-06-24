# project rosa

There are two main objectives of this project:
1. Analysis and profiling of historical sales data (done).
2. Forecast future sales using classic statistical methods and/or ML algorithms (in progress).

Sales data was retrieved from the real-wordl small company, which operates in the eco-friendly cosmetics business.
However, data was changed for privacy reasons, but initial distributions were retained.

The structure is as follows.
1. 'data' - contains file with raw data.
2. 'analysis' - contains python modules for EDA and statistical analysis, as well as separate folder with produced charts and PowerBI report.
3. 'models' - contains modules with ML/DL/statistical models which are being tested for forecasting performanc (this part is work-in-progress). 
4. 'modules' - contains Python modules with helper functions, which are used in other modules.


Models.
Implemented: 
1. LSTM - implemented using Keras package.
LSTM performs well for one-step forecasts, but its quality decreases for the next steps into the future. This is mainly due to the fact that LSTMs tend to overfitting in forecasting problems. Thus, other models should be taken into consideration.

In progress:
Classic statistical methods:
1. ARIMA/SARIMA
2. Exponential Smoothing
3. Double Exponential Smoothing (Holt)
4. Triple Exponential Smoothing (Holt-Winters)
ML/DL methods:
1. Linear ML (linear regression, Lasso & Ridge regression)
2. Nonlinear ML (support vector regression)
3. Ensemble ML (random forest)
4. Deep Learning (MLP, CNN)