import sys, os
import pandas as pd
import numpy as np
import datetime as dt

sys.path.insert(1, os.getcwd())
from Assets import BTC
from Assets import ETH
from Assets import ADA

# from API.krakenapi import CoinAPI
# from Portfolio import Portfolio
# from Trades import Trades
# from DATABASE.db import MySQL_DB, Mongo_DB

BB_PERIOD = 20

class ChartPatternAnalysis:
    def __init__(self, data):
        self.data = data
    def MA_Crossover(self):
        coin_data = self.data
        # Buy Signal
        if coin_data.loc[(coin_data.shape[0]-1), 'Close'] > coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] < coin_data.loc[(coin_data.shape[0]-2), '5d_SMA']:
            signal = 'Buy'
        # Sell Signal
        elif coin_data.loc[(coin_data.shape[0]-1), 'Close'] < coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] > coin_data.loc[(coin_data.shape[0]-2), '5d_SMA']:
            signal = 'Sell'
        # Hold Signal
        else:
            signal = 'Hold'
        return signal
    def Bollinger_Bands(self):
        coin_data = self.data
        bollinger_bands = pd.DataFrame()
        bollinger_bands['Std'] = coin_data['Close'].rolling(window = BB_PERIOD).std() 
        # Upper Bollinger Band
        bollinger_bands['Upper'] = coin_data['20d_SMA'] + (bollinger_bands['Std'] * 2)
        # Lower Bollinger Band
        bollinger_bands['Lower'] = coin_data['20d_SMA'] - (bollinger_bands['Std'] * 2)
        bollinger_bands.drop('Std', axis = 1, inplace = True)
        return bollinger_bands

# class ChartPatternAnalysis:
#     def __init__(self, coin, coins):
#         self.coin = coin
#         self.coins = coins
#     def data(self):
#         data = self.coin.getData()
#         data = data.tail(365)
#         data.reset_index(inplace = True, drop = True)
#         return data
#     def portfolio(self):
#         portfolio = Portfolio().getDBPortfolio()
#         portfolio = portfolio.iloc[-1]
#         return portfolio
#     def MA_Crossover(self):
#         portfolio = self.portfolio()
#         date = dt.datetime.now().strftime('%Y/%m/%d %H:%M:00')
#         margin = 1.005
#         coins = self.coins
#         for coin in coins:
#             coin_data = coin.getData()
#             coin_cash = portfolio[coin.abbreviation + '_Cash']
#             coin_assets = portfolio[coin.abbreviation]
#             coin_price = coin_data.loc[(coin_data.shape[0]-1), 'Close']
#             # Buy Signal
#             if coin_data.loc[(coin_data.shape[0]-1), 'Close'] > coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] < coin_data.loc[(coin_data.shape[0]-2), '5d_SMA'] and coin_cash > 0: # coin_data.loc[i, 'Close']
#                 pair = coin.alias + coin.countercurrency
#                 volume = coin_cash / (coin_price * margin)
#                 # CoinAPI().placeBuyOrder(pair, str(volume))
#                 type = 'Buy'
#                 Trades().updateTrades(date, coin.abbreviation, type, coin_price, volume)
#                 print('Buy', volume, 'of', coin.name, 'at', coin_price, coin.countercurrency)
#             # Sell Signal
#             elif coin_data.loc[(coin_data.shape[0]-1), 'Close'] < coin_data.loc[(coin_data.shape[0]-1), '5d_SMA'] and coin_data.loc[(coin_data.shape[0]-2), 'Close'] > coin_data.loc[(coin_data.shape[0]-2), '5d_SMA'] and coin_assets > 0:
#                 pair = coin.alias + coin.countercurrency
#                 volume = coin_assets
#                 # CoinAPI().placeSellOrder(pair, str(volume))
#                 type = 'Sell'
#                 Trades().updateTrades(date, coin.abbreviation, type, coin_price, volume)
#                 print('Sell', volume, 'of', coin.name, 'at', coin_price, coin.countercurrency)
#             # Hold Signal
#             else:
#                 type = 'Hold'
#                 print('Hold', coin.name)
#     def Bollinger_Bands(self):
#         prices = self.prices()
#         bollinger_bands = pd.DataFrame()
#         bollinger_bands['Std'] = prices['Close'].rolling(window = BB_PERIOD).std() 
#         # Upper Bollinger Band
#         bollinger_bands['Upper'] = prices['SMA'] + (bollinger_bands['Std'] * 2)
#         # Lower Bollinger Band
#         bollinger_bands['Lower'] = prices['SMA'] - (bollinger_bands['Std'] * 2)
#         bollinger_bands.drop('Std', axis = 1, inplace = True)
#         return bollinger_bands
#     def Golden_Cross():
#         pass