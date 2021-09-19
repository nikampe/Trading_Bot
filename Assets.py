import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

from DATABASE.db import DB

ROLLING_PERIOD = '30d'
ROLLING_INTERVAL = '5m'
TRAINING_PERIOD = '3y'
TRAINING_INTERVAL = '1d'

MA_PERIODS = [5, 10, 20]

class Coin:
    def __init__(self, name, id, abbreviation, alias, countercurrency, dbname, dbname_train):
        self.name = name
        self.id = id
        self.abbreviation = abbreviation
        self.alias = alias
        self.countercurrency = countercurrency
        self.dbname = dbname
        self.dbname_train = dbname_train
    def ticker(self):
        ticker = self.abbreviation + '-' + self.countercurrency
        return ticker
    def balance(self):
        ticker = self.abbreviation
        positions = ['_Cash', '_Coins', '_Coins_Value']
        balance = []
        for position in positions:
            asset = ticker + position
            balance.append(asset)
        return balance
    def getData(self):
        data = DB().getRequest(table = self.dbname, columns = '*')
        return data
    def getTrainingData(self):
        train_data = DB().getRequest(table = self.dbname_train, columns = '*')
        return train_data
    def returns(self, data, identifier):
        intervals_per_day = 288
        if identifier == 'data':
            data['1d_Return'] = data['Close'].pct_change(periods = intervals_per_day)
            data['5min_Return'] = data['Close'].pct_change(periods = 1)
        if identifier == 'training_data':
            data['Return'] = data['Close'].pct_change(periods = 1)
        return data
    def movingAverages(self, data, periods):
        intervals_per_day = 288
        for period in periods:
            data[str(period) + 'd_SMA'] = data['Close'].rolling(window = period * intervals_per_day).mean()
            data[str(period) + 'd_EMA'] = data['Close'].ewm(span = period * intervals_per_day, adjust = False).mean()
        return data
    def updateData(self):
        data_old = self.getData()
        data_new = yf.Ticker(self.ticker())
        data_new = data_new.history(period = ROLLING_PERIOD, interval = ROLLING_INTERVAL)
        data_new = data_new[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_new[['Open', 'High', 'Low', 'Close', 'Volume']] = data_new[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        data_new['ID'] = self.id
        data_new['Ticker'] = self.ticker()
        data_new.reset_index(inplace = True)
        data_new['Datetime'] = data_new['Datetime'].dt.tz_convert('Europe/Zurich')
        data_new['Datetime'] = data_new['Datetime'].dt.strftime('%Y/%m/%d %H:%M:%S')
        data_new.drop(data_new.tail(1).index, inplace = True)
        data = self.returns(data_new, 'data')
        data = self.movingAverages(data, MA_PERIODS)
        for i in range(0, len(data)):
            if data.loc[i, 'Datetime'] in data_old['Datetime'].to_numpy():
                data.drop(data.loc[i].name, inplace = True)
            else:
                pass
        if data.isnull().values.any():
            print('Error: NaN values in the coin data request for', self.name)
        else:
            DB().postRequest(data, self.dbname)
            print('Data submitted to DB for', self.name)
    def updateTrainingData(self):
        train_data_old = self.getTrainingData()
        train_data_new = yf.Ticker(self.ticker())
        train_data_new = train_data_new.history(period = TRAINING_PERIOD, interval = TRAINING_INTERVAL)
        train_data_new = train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']]
        train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']] = train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        train_data_new.reset_index(inplace = True)
        train_data_new['Date'] = train_data_new['Date'].dt.strftime('%Y/%m/%d')
        train_data_new.drop(train_data_new.tail(1).index, inplace = True)
        train_data_new = self.returns(train_data_new, 'training_data')
        for i in range(0, len(train_data_new)):
            if train_data_new.loc[i, 'Date'] in train_data_old['Date'].to_numpy():
                train_data_new.drop(train_data_new.loc[i].name, inplace = True)
            else:
                pass
        DB().postRequest(train_data_new, self.dbname_train)
        print('Training data submitted to DB for', self.name)
        return train_data_new

BTC = Coin('Bitcoin', 1, 'BTC', 'XBT', 'USD', 'data_BTC', 'train_data_BTC')
ETH = Coin('Ethereum', 2, 'ETH', 'ETH', 'USD', 'data_ETH', 'train_data_ETH')
ADA = Coin('Cardano', 3, 'ADA', 'ADA', 'USD', 'data_ADA', 'train_data_ADA')

coins = [BTC, ETH, ADA]
for coin in coins:
    coin.updateData()
    coin.updateTrainingData()