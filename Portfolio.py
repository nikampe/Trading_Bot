import pandas as pd
import numpy as np
import datetime as dt

from API.krakenapi import CoinAPI
from DATABASE.db import DB
from Assets import BTC
from Assets import ETH
from Assets import ADA

class Portfolio:
    def __init__(self):
        self.BTC = BTC
        self.ETH = ETH
        self.ADA = ADA
        self.coins = [BTC, ETH, ADA]
    def getKrakenPortfolio(self):
        resp = CoinAPI().getBalance()
        resp = resp['result']
        date = dt.datetime.now().strftime('%Y/%m/%d %H:%M:00')
        portfolio = pd.DataFrame.from_dict(resp, orient = 'index', columns = [date])
        portfolio = portfolio.astype('float64')
        return portfolio
    def detailPortfolio(self):
        portfolio = self.getKrakenPortfolio()
        detail_portfolio = portfolio.T
        detail_portfolio.rename(columns = {'CHF': 'Cash', 'XXBT': 'BTC', 'XETH': 'ETH'}, inplace = True)
        total_value = 0
        for coin in self.coins:
            data = coin.getData()
            amount = detail_portfolio.loc[detail_portfolio.tail(1).index, coin.abbreviation].to_numpy()
            value = data.loc[data.tail(1).index, 'Close'].to_numpy()
            amount = amount[0]
            value = value[0]
            detail_portfolio[coin.abbreviation + '_Value'] = amount * value
            detail_portfolio[coin.abbreviation + '_Cash'] = 0
            detail_portfolio[coin.abbreviation + '_Weight'] = 1/3
            total_value += amount * value
        detail_portfolio['Total_Value'] = detail_portfolio['Cash'] + total_value
        return detail_portfolio
    def updatePortfolio(self):
        detail_portfolio = self.detailPortfolio()
        detail_portfolio.reset_index(inplace = True)
        detail_portfolio.rename(columns = {'index': 'Datetime'}, inplace = True)
        detail_portfolio.drop('Cash', axis = 1, inplace = True)
        DB().postRequest(detail_portfolio, 'coin_portfolio')
    def getDBPortfolio(self):
        self.updatePortfolio()
        portfolio = DB().getRequest(table = 'coin_portfolio', columns = '*')
        return portfolio

Portfolio().updatePortfolio()