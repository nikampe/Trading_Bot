import pandas as pd
import numpy as np
import datetime as dt

from API.krakenapi import CoinAPI
from Portfolio import Portfolio
from Trades import Trades
from DATABASE.db import DB
from MODELS.Models import TimeSeriesModels, MLModels
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

if __name__ == '__main__':
    MLModels(BTC, [BTC, ETH, ADA]).GRUForecast()
    MLModels(BTC, [BTC, ETH, ADA]).LSTMForecast()