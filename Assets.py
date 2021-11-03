import os
import numpy as np
import pandas as pd
import datetime as dt

# Data API
import yfinance as yf

# Database
from DATABASE.db import Mongo_DB, MySQL_DB

ROLLING_PERIOD = '50d'
ROLLING_INTERVAL = '5m'
TRAINING_PERIOD = '1mo'
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
    def get5minData(self):
        data = MySQL_DB().getRequest(table = self.dbname, columns = '*')
        return data
    def get5minLimitedData(self, sortColumn, limit):
        limited_data = MySQL_DB().getLimitedRequest(table = self.dbname, columns = '*', sort_column = sortColumn, limit = limit)
        return limited_data
    def get1dData(self):
        train_data = MySQL_DB().getRequest(table = self.dbname_train, columns = '*')
        return train_data
    def get1dLimitedData(self, sortColumn, limit):
        limited_train_data = MySQL_DB().getLimitedRequest(table = self.dbname_train, columns = '*', sort_column = sortColumn, limit = limit)
        return limited_train_data
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
    def update5minHistoricalData(self):
        files = os.listdir(f'./HISTORICAL_DATA/{self.abbreviation}')
        files = sorted(files)
        data = pd.DataFrame()
        for file in files:
            if self.abbreviation != 'ADA':
                data_raw = pd.read_csv(f'./HISTORICAL_DATA/{self.abbreviation}/{file}', sep = ',', header = 0)
                data_raw.drop(['unix', f'Volume{self.abbreviation}'], axis = 1, inplace = True)
                data_raw = data_raw.rename(columns = {'date': 'Datetime', 'symbol': 'Ticker', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'VolumeUSD': 'Volume'})
                data_raw['Datetime'] = pd.to_datetime(data_raw['Datetime'], format = '%Y/%m/%d %H:%M:%S')
                data_raw = data_raw.groupby(pd.Grouper(key = 'Datetime', freq = '5T')).mean()
                data_raw.reset_index(drop = False, inplace = True)
                data_raw['Datetime'] = data_raw['Datetime'].dt.strftime('%Y/%m/%d %H:%M:%S')
                data_raw.set_index('Datetime', inplace = True, drop = True)
                data = pd.concat([data, data_raw])
            elif self.abbreviation == 'ADA':
                data_raw = pd.read_csv(f'./HISTORICAL_DATA/{self.abbreviation}/{file}', header = None)
                data_raw.drop(6, axis = 1, inplace = True)
                data_raw.rename(columns = {0: 'Datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'}, inplace = True)
                data_raw['Datetime'] = [dt.datetime.fromtimestamp(x) for x in data_raw['Datetime']]
                data_raw['Datetime'] = data_raw['Datetime'].dt.strftime('%Y/%m/%d %H:%M:%S')
                data_raw.set_index('Datetime', drop = True, inplace = True)
                data = pd.concat([data, data_raw])
        data['ID'] = self.id
        data['Ticker'] = self.ticker()
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        data.sort_values(by = ['Datetime'], inplace = True)
        data.reset_index(inplace = True)
        data = self.returns(data, 'data')
        data = self.movingAverages(data, MA_PERIODS)
        # MySQL
        data_SQL = data
        num_upload_batches = 3
        try:
            for i in range(0, num_upload_batches):
                MySQL_DB().postRequest(data_SQL.loc[len(data_SQL)*(i/num_upload_batches):len(data_SQL)*((i+1)/num_upload_batches)], self.dbname)
                print(f'Batch #{str(i+1)} of historical {self.abbreviation} data successfully uploaded to MySQL.')
        except:
            print(f'Error in MySQL upload for {self.name}.')
        # MongoDB
        for i in data_SQL.index:
            data_JSON = {
                'Datetime' : data_SQL.loc[i, 'Datetime'],
                'ID' : int(data_SQL.loc[i, 'ID']),
                'Ticker': data_SQL.loc[i, 'Ticker'],
                'OHLCV': {
                    'Open': round(float(data_SQL.loc[i, 'Open']), 4),
                    'High': round(float(data_SQL.loc[i, 'High']), 4),
                    'Low': round(float(data_SQL.loc[i, 'Low']), 4),
                    'Close': round(float(data_SQL.loc[i, 'Close']), 4),
                    'Volume': round(float(data_SQL.loc[i, 'Volume']), 4)
                },
                'Returns': {
                    '5min_Return': round(float(data_SQL.loc[i, '5min_Return']), 4),
                    '1d_Return': round(float(data_SQL.loc[i, '1d_Return']), 4)
                },
                'Averages': {
                    '5d_SMA': round(float(data_SQL.loc[i, '5d_SMA']), 4),
                    '5d_EMA': round(float(data_SQL.loc[i, '5d_EMA']), 4),
                    '10d_SMA': round(float(data_SQL.loc[i, '10d_SMA']), 4),
                    '10d_EMA': round(float(data_SQL.loc[i, '10d_EMA']), 4),
                    '20d_SMA': round(float(data_SQL.loc[i, '20d_SMA']), 4),
                    '20d_EMA': round(float(data_SQL.loc[i, '20d_EMA']), 4)
                }
            }
            try:
                Mongo_DB().postRequest(data_JSON, self.dbname)
                print(f'{self.abbreviation} uploaded for {str(i)} to MongoDB.')
            except:
                print(f'Error in MongoDB upload for {self.abbreviation}.')
    def update5minData(self):
        # data_old = self.getData()
        data_old = self.getLimitedData(sortColumn = "Datetime", limit = 60 * 288)
        data_new = yf.Ticker(self.ticker())
        data_new = data_new.history(period = ROLLING_PERIOD, interval = ROLLING_INTERVAL)
        data_new = data_new[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_new['Volume'] = data_new['Volume'] / 1000
        data_new[['Open', 'High', 'Low', 'Close', 'Volume']] = data_new[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        data_new['ID'] = self.id
        data_new['Ticker'] = self.ticker()
        data_new.reset_index(inplace = True)
        data_new['Datetime'] = data_new['Datetime'].dt.tz_convert('Europe/Zurich')
        data_new['Datetime'] = data_new['Datetime'].dt.strftime('%Y/%m/%d %H:%M:%S')
        data_new.drop(data_new.tail(1).index, inplace = True)
        # MySQL
        data_SQL = self.returns(data_new, 'data')
        data_SQL = self.movingAverages(data_SQL, MA_PERIODS)
        for i in range(0, len(data_SQL)):
            if data_SQL.loc[i, 'Datetime'] in data_old['Datetime'].to_numpy():
                data_SQL.drop(data_SQL.loc[i].name, inplace = True)
            else:
                pass
        if data_SQL.isnull().values.any():
            print('Error: NaN values in the coin data request for', self.abbreviation)
        else:
            try:
                MySQL_DB().postRequest(data_SQL, self.dbname)
                print(f'Data submitted to MySQL for {self.abbreviation}.')
            except:
                print(f'Error in MySQL upload for {self.abbreviation}.')
            # MongoDB
            for i in data_SQL.index:
                data_JSON = {
                    'Datetime' : data_SQL.loc[i, 'Datetime'],
                    'ID' : int(data_SQL.loc[i, 'ID']),
                    'Ticker': data_SQL.loc[i, 'Ticker'],
                    'OHLCV': {
                        'Open': round(float(data_SQL.loc[i, 'Open']), 4),
                        'High': round(float(data_SQL.loc[i, 'High']), 4),
                        'Low': round(float(data_SQL.loc[i, 'Low']), 4),
                        'Close': round(float(data_SQL.loc[i, 'Close']), 4),
                        'Volume': round(float(data_SQL.loc[i, 'Volume']), 4)
                    },
                    'Returns': {
                        '5min_Return': round(float(data_SQL.loc[i, '5min_Return']), 4),
                        '1d_Return': round(float(data_SQL.loc[i, '1d_Return']), 4)
                    },
                    'Averages': {
                        '5d_SMA': round(float(data_SQL.loc[i, '5d_SMA']), 4),
                        '5d_EMA': round(float(data_SQL.loc[i, '5d_EMA']), 4),
                        '10d_SMA': round(float(data_SQL.loc[i, '10d_SMA']), 4),
                        '10d_EMA': round(float(data_SQL.loc[i, '10d_EMA']), 4),
                        '20d_SMA': round(float(data_SQL.loc[i, '20d_SMA']), 4),
                        '20d_EMA': round(float(data_SQL.loc[i, '20d_EMA']), 4)
                    }
                }
                try:
                    Mongo_DB().postRequest(data_JSON, self.dbname)
                    print(f'Data submitted to MongoDB for {self.name}.')
                except:
                    print(f'Error in MongoDB upload for {self.name}.')
    def update1dData(self):
        # train_data_old = self.getTrainingData()
        train_data_old = self.getLimitedTrainingData(sortColumn = "Date", limit = 60)
        train_data_new = yf.Ticker(self.ticker())
        train_data_new = train_data_new.history(period = TRAINING_PERIOD, interval = TRAINING_INTERVAL)
        train_data_new = train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']]
        train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']] = train_data_new[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        train_data_new.reset_index(inplace = True)
        train_data_new['Date'] = train_data_new['Date'].dt.strftime('%Y/%m/%d')
        train_data_new.drop(train_data_new.tail(1).index, inplace = True)
        # MySQL
        train_data_SQL = self.returns(train_data_new, 'training_data')
        for i in range(0, len(train_data_SQL)):
            if train_data_SQL.loc[i, 'Date'] in train_data_old['Date'].to_numpy():
                train_data_SQL.drop(train_data_SQL.loc[i].name, inplace = True)
            else:
                pass
        try:
            MySQL_DB().postRequest(train_data_SQL, self.dbname_train)
            print(f'Training data submitted to MySQL for {self.abbreviation}.')
        except:
            print(f'Error in MySQL upload for {self.abbreviation}.')
        # MongoDB
        for i in train_data_SQL.index:
            train_data_JSON = {
                'Date' : train_data_SQL.loc[i, 'Date'],
                'OHLCV': {
                    'Open': round(float(train_data_SQL.loc[i, 'Open']), 4),
                    'High': round(float(train_data_SQL.loc[i, 'High']), 4),
                    'Low': round(float(train_data_SQL.loc[i, 'Low']), 4),
                    'Close': round(float(train_data_SQL.loc[i, 'Close']), 4),
                    'Volume': round(float(train_data_SQL.loc[i, 'Volume']), 4)
                },
                'Returns': {
                    'Return': round(float(train_data_SQL.loc[i, 'Return']), 4)
                }
            }
            try:
                Mongo_DB().postRequest(train_data_JSON, self.dbname_train)
                print(f'Training data submitted to MongoDB for {self.abbreviation}.')
            except:
                print(f'Error in MongoDB upload for {self.abbreviation}.')

BTC = Coin('Bitcoin', 1, 'BTC', 'XBT', 'USD', 'data_BTC', 'train_data_BTC')
ETH = Coin('Ethereum', 2, 'ETH', 'ETH', 'USD', 'data_ETH', 'train_data_ETH')
ADA = Coin('Cardano', 3, 'ADA', 'ADA', 'USD', 'data_ADA', 'train_data_ADA')

if __name__ == '__main__':
    coins = [BTC, ETH, ADA]
    for coin in coins:
        coin.update5minData()
        coin.update1dData()