# project rosa

There are two main objectives of this project:
1. Analysis and profiling of historical sales data (done).
2. Forecast future sales using classical statistical methods and/or ML algorithms (in progress).

Sales data was retrieved from the real-wordl small company, which operates in the eco-friendly cosmetics business.
However, data was changed for privacy reasons, but initial distributions were retained.

The structure is as follows.
1. 'data' - contains file with raw data.
2. 'analysis' - contains python modules for EDA and statistical analysis, as well as separate folder with produced charts and PowerBI report.
3. 'models' - contains modules with ML/DL/statistical models which are being tested for forecasting performance (currently it's only LSTM model, implemented using Keras package).
4. 'modules' - contains Python modules with helper functions, which are used in other modules.