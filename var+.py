#The following code is used to implement the VAR model which will be used for each financial time series forecast (VAR(1)).
#Hereby, the logarithmic returns are being analyzed, as the adjusted closing prices themselves are not stationary (refer to the visualization & model selection file
#for the associated Augmented Dickey-Fuller tests). On the foundation of the log returns, the code generates out-of-sample forecasts of the adusted closing prices
#ten steps into the future. The evaluation of the forecasts (the results are then being compared to the ARIMA and the video prediction neural network) is performed on the basis of
#the MAPE, the RRMSE and the price movement direction metric.

#Import all necessary libraries

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_percentage_error

#Download the dataset
#Use the code below (commented out) if the download should happen directly through Yahoo Finance

'''
ticker_list = ['AAPL', 'ACN', 'ADBE',
              'AMD', 'GOOG', 'MU',
              'PYPL', 'QCOM', 'STX']
stock_prices = yf.download(ticker_list, start = '2023-07-17', end = '2023-12-16', interval = '1d')['Adj Close']
'''

stock_prices = pd.read_csv('https://github.com/sch-wm/time_series/raw/main/yfinance_stock_prices_for_baselines.csv', index_col = 0)
stock_prices.index = pd.to_datetime(stock_prices.index)

#Inspect the data

stock_prices.head(10)
stock_prices.shape

#Preprocess the data (insert missing dates and remove the related nans through linear interpolation)

dates = pd.date_range(start = pd.to_datetime('2023-07-17').tz_localize("GMT+0") , end = pd.to_datetime('2023-12-16').tz_localize("GMT+0"), freq = 'B')
stock_prices = stock_prices.reindex(dates)
stock_prices.interpolate(inplace = True)

#Inspect the data

stock_prices.shape #Dataframe has 110 elements, starts at a Monday, ends at a Friday

#Use ln on the prices and difference to get log returns

log_stock_prices = np.log(stock_prices) #Dataframe has 110 elements, index: 0 - 109, data starts at Monday, ends at Friday
log_returns_stock_prices = log_stock_prices.diff()
log_returns_stock_prices.dropna(inplace = True) #Dataframe has 109 elements after differencing, index: 0 - 108, data starts at Tuesday, ends at Friday

#Define and initialize all parameters, necessary for looping

number_of_assets = 9
timesteps_forecast = 10

#Compute log returns forecasts with VAR (manual differencing is required)

VAR_model_before_fit = VAR(log_returns_stock_prices.iloc[:99, :]) #Data sequence starts at Tu, ends at Fr
VAR_model = VAR_model_before_fit.fit(maxlags = 1)
forecasts_list = VAR_model.forecast(log_returns_stock_prices.values, steps = 10) #Forecasts start at Mo, ends at Fr, results are stored in np.array of shape 10, 9

#Inspect the forecasts

print(forecasts_list, 'Forecasts in list of lists (log returns)')
print(len(forecasts_list), 'timesteps')
print(len(forecasts_list[0]), 'assets per timestep')

#Prepare one list with 10 * 9 elements for forecasts

forecasts_flat = []
for asset in range(number_of_assets):
  for timestep in range(timesteps_forecast):
    forecasts_flat.append(forecasts_list[timestep][asset])

#Inspect the forecasts of the log returns

print(forecasts_flat, 'Forecasts in one list (log returns)')
print(len(forecasts_flat), 'elements in one list')

#Transform the forecasts of the logarithmically-modified adjusted closing prices back into
#forecasts of adjusted closing prices

forecasts_flat_prices = []
index_shift = 0
for asset in range(number_of_assets):
  exponent = log_stock_prices.iloc[99, asset] #Computation starts from ln price of Fr (add up the log returns)
  print(log_stock_prices.iloc[99, asset], 'Last observed ln price - asset', asset)
  print(log_stock_prices.iloc[100:110, asset], 'Test data ln prices - asset', asset)
  for timestep in range(timesteps_forecast):
    exponent = exponent + forecasts_flat[timestep + index_shift]
    print('Forecast ln price', 1 + timestep + asset * timesteps_forecast, '-', exponent, '- asset', asset)
    price = np.exp(exponent)
    print('Forecast price', 1 + timestep + asset * timesteps_forecast, '-', price, '- asset', asset)
    forecasts_flat_prices.append(price)
  index_shift = index_shift + timesteps_forecast

print(forecasts_flat_prices, 'Forecasts of adjusted closing prices')

#Put test data in a list

test_flat = []
for asset in range(number_of_assets):
  for timestep in range(timesteps_forecast):
    test_flat.append(stock_prices.iloc[timestep + 100, asset]) #Appending starts at Mo

#Inspect the list

print(test_flat, 'Test data prices')
stock_prices.tail(10)

#Compute the MAPE

MAPE_list = []
MAPE_assets_averages_list = []
total_errors = number_of_assets * timesteps_forecast
for error in range(total_errors):
  MAPE = np.abs(test_flat[error] - forecasts_flat_prices[error]) / test_flat[error]
  MAPE_list.append(MAPE)
print(MAPE_list, 'MAPE')
index_shift = 0
for asset in range(number_of_assets):
  MAPE_asset_average = mean_absolute_percentage_error(test_flat[index_shift : index_shift + 10], forecasts_flat_prices[index_shift : index_shift + 10])
  MAPE_assets_averages_list.append(MAPE_asset_average)
  index_shift = index_shift + 10
print(MAPE_assets_averages_list, 'average MAPE (over all timesteps/per asset)')

#Compute the RRMSE

RRMSE_assets = []
index_shift = 0
for asset in range(number_of_assets):
  RRMSE_asset_sum = 0.00
  for timestep in range(timesteps_forecast):
    RRMSE_asset_sum = RRMSE_asset_sum + (((forecasts_flat_prices[timestep + index_shift] - test_flat[timestep + index_shift])
                                                     / test_flat[timestep + index_shift]) ** 2)
  RRMSE_asset = np.sqrt(RRMSE_asset_sum / timesteps_forecast)
  RRMSE_assets.append(RRMSE_asset)
  index_shift = index_shift + 10
print(RRMSE_assets, 'Individual RRMSE values (for each asset)')
RRMSE_assets = sum(RRMSE_assets)
RRMSE_assets = RRMSE_assets / number_of_assets
print(RRMSE_assets, 'RRMSE')

#Compute the price movement direction metric

price_movement_direction_assets = []
weight_lambda = 0.60
index_shift = 0
for asset in range(number_of_assets):
  price_movement_direction_asset_sum = 0.00
  if (((forecasts_flat_prices[index_shift] - stock_prices.iloc[99, asset]) / (test_flat[index_shift] - stock_prices.iloc[99, asset])) > 0):
      price_movement_direction_asset_sum = price_movement_direction_asset_sum + (1 * (weight_lambda ** 0))
  for timestep in range(timesteps_forecast - 1):
    if (((forecasts_flat_prices[timestep + 1 + index_shift] - forecasts_flat_prices[timestep + index_shift]) / (test_flat[timestep + 1 + index_shift] - test_flat[timestep + index_shift])) > 0):
      price_movement_direction_asset_sum = price_movement_direction_asset_sum + (1 * (weight_lambda ** (timestep + 1)))
  price_movement_direction_assets.append(price_movement_direction_asset_sum)
  index_shift = index_shift + 10
print(price_movement_direction_assets, 'Individual price movement direction metric values (for each asset)')
price_movement_direction_assets_sum = sum(price_movement_direction_assets)
print(price_movement_direction_assets_sum, 'Price movement direction metric')
