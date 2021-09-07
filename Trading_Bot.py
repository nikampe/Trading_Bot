import pandas as pd
import numpy as np
import datetime as dt
from keras import layers
from keras import models
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from API.krakenapi import CoinAPI
from Portfolio import Portfolio
from Trades import Trades
from DATABASE.db import DB
from Assets import BTC
from Assets import ETH
from Assets import ADA

TRANSACTION_COSTS = 0.015

def MACrossover():
    portfolio = Portfolio().getDBPortfolio()
    portfolio = portfolio.iloc[-1]
    date = dt.datetime.now().strftime('%Y/%m/%d %H:%M:00')
    margin = 1.005
    coins = [BTC, ETH, ADA]
    for coin in coins:
        coin_data = coin.getData()
        coin_cash = portfolio[coin.abbreviation + '_Cash']
        coin_assets = portfolio[coin.abbreviation]
        coin_price = coin_data.loc[(coin_data.shape[0]-1), 'Close']
        # Buy Signal
        if coin_data.loc[(coin_data.shape[0]-1), 'Close'] > coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] < coin_data.loc[(coin_data.shape[0]-2), '5d_SMA'] and coin_cash > 0: # coin_data.loc[i, 'Close']
            pair = coin.alias + coin.countercurrency
            volume = coin_cash / (coin_price * margin)
            # CoinAPI().placeBuyOrder(pair, str(volume))
            type = 'Buy'
            Trades().updateTrades(date, coin.abbreviation, type, coin_price, volume)
            print('Buy', volume, 'of', coin.name, 'at', coin_price, coin.countercurrency)
        # Sell Signal
        elif coin_data.loc[(coin_data.shape[0]-1), 'Close'] < coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] > coin_data.loc[(coin_data.shape[0]-2), '5d_SMA'] and coin_assets > 0:
            pair = coin.alias + coin.countercurrency
            volume = coin_assets
            # CoinAPI().placeSellOrder(pair, str(volume))
            type = 'Sell'
            Trades().updateTrades(date, coin.abbreviation, type, coin_price, volume)
            print('Sell', volume, 'of', coin.name, 'at', coin_price, coin.countercurrency)
        # Hold Signal
        else:
            type = 'Hold'
            print('Hold', coin.name)

def ARMAForecast():
    coins = [BTC, ETH, ADA]
    model_overview = pd.DataFrame()
    for coin in coins:
        returns = coin.getData()['1d_Return']
        n = len(returns)
        # AR- and MA-order Validation
        p_arr = [1,2,3,4,5]
        q_arr = [1,2,3,4,5]
        p_val = [3]
        q_val = [2]
        if len(p_val) == 0 or len(q_val) == 0:
            AR_orders = p_arr
            MA_orders = q_arr
        else:
            AR_orders = p_val
            MA_orders = q_val
        for p in AR_orders:
            for q in MA_orders:
                # ARMA Model Fit
                model = ARIMA(returns, order = (p, 0, q))
                model_fit = model.fit()
                # ARMA Model Forecast
                horizon = 10
                returns_predicted = model_fit.predict()
                returns_forecasted = model_fit.predict(start = n + 1, end = n + horizon - 1)
        # Plot
        plt.figure(figsize = (16,6))
        plt.plot(range(1, n+1), returns, linestyle = ':', marker = 'o', color = 'blue', label = "Returns")
        plt.plot(range(1, n+1), returns_predicted, linestyle = ':', marker = 'o', mfc = 'none', color = 'green', label = "Returns Fitted")
        plt.axvline(x = n, linestyle = ':', color = 'k')
        plt.plot(range(n+1, n+horizon), returns_forecasted, linestyle = ':', marker = 'o', color = 'green', label = "Returns Forecasted")
        plt.title("Returns, Fitted Returns and Forecasted Returns for " + coin.name + " (ARMA)", fontsize = 16)
        plt.legend()
        plt.xlim(n-100, n+horizon+1)
        plt.show()
       
def CNNForecast():
    forecasts = pd.DataFrame()
    coins = [BTC, ETH, ADA]
    for coin in coins:
        prices = coin.getTrainingData()['Close']
        forecasts.loc[len(forecasts), 'Coin'] = coin.abbreviation
        # Rescaling & 1-Lag Shift
        prices_min = min(prices)
        prices_max = max(prices)
        prices = (prices-prices_min)/(prices_max-prices_min)
        prices = pd.Series(prices)
        df_prices = pd.concat([prices.shift(1), prices], axis = 1)
        df_prices.columns = ['price(t-1)', 'price(t)']
        # Train-Test Split
        n = len(prices)
        n_train = int(n * 0.9)
        n_test = n - n_train
        train, test = df_prices.iloc[0:n_train,:], df_prices.iloc[n_train:n,:]
        train = train.dropna(inplace=False)
        p = df_prices.shape[1] - 1
        X_train, y_train = train.iloc[:,0:p], train.iloc[:,p]
        X_test, y_test = test.iloc[:,0:p], test.iloc[:,p]
        # Fully Connected NN - Model Setup (!!!NOT VALIDATED/TUNED!!!)
        p = X_train.shape[1]
        model = models.Sequential()
        model.add(layers.Dense(16, activation = 'relu', input_shape = (p,)))
        model.add(layers.Dense(8, activation = 'relu'))
        model.add(layers.Dense(4, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'linear')) 
        # Fully Connected NN - Model Compiling & Fit (!!!NOT VALIDATED/TUNED!!!)
        model.compile(optimizer = 'adam', loss = 'mse')
        model.fit(
            X_train, 
            y_train,
            epochs = 25,
            batch_size = 128,
            shuffle = False,
            validation_data = (X_test, y_test))
        # Fully Connected NN - Model Prediction & Forecast
        horizon = 10
        prices_forecasted = np.empty(horizon+p)
        prices_forecasted[0:p] = prices[(n-p):n] 
        for t in range(p, horizon+p):
            prices_forecasted[t] = model.predict([prices_forecasted[t-1]])
        prices_forecasted = prices_forecasted[p:(n+p)]
        # Fully Connected NN - Descaling
        prices = prices_min + prices * (prices_max - prices_min)
        prices_predicted = model.predict(df_prices.iloc[:,0:1])
        prices_predicted = prices_min + prices_predicted * (prices_max - prices_min)
        prices_forecasted = prices_min + prices_forecasted * (prices_max - prices_min)
        # Plot
        plt.figure(figsize = (16,6))
        plt.plot(range(1, n+1), prices,linestyle = ':', marker = 'o', color = 'blue', label = "Prices")
        plt.plot(range(1, n+1), prices_predicted, linestyle = ':', marker = 'o', mfc = 'none', color = 'green', label = "Prices Predicted (1-step ahead)")
        plt.axvline(x = n, linestyle = ':', color = 'k')
        plt.plot(range(n+1, n+horizon+1), prices_forecasted, linestyle = ':', marker = 'o', color = 'green', label = "Prices Forecasted")
        plt.title("Prices, Fitted Prices and Forecasted Prices for " + coin.name + " Fully Connected NN", fontsize = 16)
        plt.legend()
        plt.xlim(n-100, n+horizon+1)
        plt.show()
        # Fully Connected NN - Forecast Results
        for i in range(0, len(prices_forecasted)):
            forecasts.loc[len(forecasts) - 1, str(i + 1) + 'd_Forecast'] = prices_forecasted[i]
    print(forecasts)     

def RNNForecast():
    forecasts = pd.DataFrame()
    coins = [BTC, ETH, ADA]
    for coin in coins:
        prices = coin.getTrainingData()['Close']
        forecasts.loc[len(forecasts), 'Coin'] = coin.abbreviation
        # Rescaling & 1-Lag Shift
        prices_min = min(prices)
        prices_max = max(prices)
        prices = (prices-prices_min)/(prices_max-prices_min)
        prices = pd.Series(prices)
        # Data Preparation [batch, timesteps, feature]
        n_timesteps = 14
        features_set = []
        labels = []
        for i in range(n_timesteps, prices.shape[0]):
            features_set.append(prices[i-n_timesteps:i])
            labels.append(prices[i])   
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
        # Train-Test Split
        n = features_set.shape[0]
        n_train = int(n * 0.80)
        n_test = n - n_train
        features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
        labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
        # Recurrent NN - Model Setup (!!!NOT VALIDATED/TUNED!!!)
        n_features = 1
        model = models.Sequential()
        model.add(LSTM(6, return_sequences=False, activation='tanh', input_shape = (n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(layers.Dense(1, activation='linear')) 
        # Recurrent NN - Model Compiling & Fit (!!!NOT VALIDATED/TUNED!!!)
        model.compile(optimizer = 'adam', loss = 'mse')
        model.fit(
            features_set_train, 
            labels_train,
            epochs = 40,
            batch_size = 32,
            validation_data = (features_set_test, labels_test))
        # Fully Connected NN - Descaling
        prices = prices_min + prices * (prices_max - prices_min)
        prices_fitted = model.predict(features_set)
        prices_fitted = prices_min + prices_fitted * (prices_max - prices_min)
        # Recurrent NN - Plot
        n = len(prices)
        window = np.arange(n-500, n)
        plt.figure(figsize=(16,6))
        plt.title("Prices and Fitted Prices for " + coin.name + " (RNN)", fontsize = 16)
        plt.axvline(x = n-1, linestyle = ':', color = 'k')
        plt.plot(window, prices[window], linestyle = ':', marker = 'o', color = 'blue', label = "Prices")
        plt.plot(window, prices_fitted[window-n_timesteps], linestyle = ':', marker = 'o', color = 'red', label = "Prices Predicted (1-step ahead)")
        plt.grid(True)
        plt.legend(loc = 'lower left', fontsize = 16)
        plt.show()

def LSTMForecast():
    forecasts = pd.DataFrame()
    coins = [BTC, ETH, ADA]
    for coin in coins:
        prices = coin.getTrainingData()['Close']
        forecasts.loc[len(forecasts), 'Coin'] = coin.abbreviation
        # Rescaling & 1-Lag Shift
        prices_min = min(prices)
        prices_max = max(prices)
        prices = (prices-prices_min)/(prices_max-prices_min)
        prices = pd.Series(prices)
        # Data Preparation [batch, timesteps, feature]
        n_timesteps = 14
        features_set = []
        labels = []
        for i in range(n_timesteps, prices.shape[0]):
            features_set.append(prices[i-n_timesteps:i])
            labels.append(prices[i])   
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
        # Train-Test Split
        n = features_set.shape[0]
        n_train = int(n * 0.80)
        n_test = n - n_train
        features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
        labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
        # Recurrent NN - Model Setup (!!!NOT VALIDATED/TUNED!!!)
        n_features = 1
        model = models.Sequential()
        model.add(LSTM(16, activation = 'relu', input_shape=(n_timesteps, n_features)))
        model.add(layers.Dense(1, activation = 'linear')) 
        model.summary()
        # Recurrent NN - Model Compiling & Fit (!!!NOT VALIDATED/TUNED!!!)
        model.compile(optimizer = 'adam', loss = 'mse')
        model.fit(
            features_set_train, 
            labels_train,
            epochs = 40,
            batch_size = 32,
            validation_data = (features_set_test, labels_test))
        prices = prices_min + prices * (prices_max - prices_min)
        prices_fitted = model.predict(features_set)
        prices_fitted = prices_min + prices_fitted * (prices_max - prices_min)
        # Recurrent NN - Plot
        n = len(prices)
        window = np.arange(n-500, n)
        plt.figure(figsize=(16,6))
        plt.title("Prices and Fitted Prices for " + coin.name + " (LSTM)", fontsize = 16)
        plt.axvline(x = n-1, linestyle = ':', color = 'k')
        plt.plot(window, prices[window], linestyle = ':', marker = 'o', color = 'blue', label = "Prices")
        plt.plot(window, prices_fitted[window-n_timesteps], linestyle = ':', marker = 'o', color = 'red', label = "Prices Predicted (1-step ahead)")
        plt.grid(True)
        plt.legend(loc = 'lower left', fontsize = 16)
        plt.show()

if __name__ == '__main__':
    CNNForecast()