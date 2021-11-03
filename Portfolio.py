import pandas as pd
import numpy as np
import datetime as dt

from API.krakenapi import CoinAPI
from DATABASE.db import MySQL_DB, Mongo_DB

class Portfolio:
    def __init__(self, coins):
        self.coins = coins
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
        detail_portfolio_SQL = self.detailPortfolio()
        detail_portfolio_SQL.reset_index(inplace = True)
        detail_portfolio_SQL.rename(columns = {'index': 'Datetime'}, inplace = True)
        detail_portfolio_SQL.drop('Cash', axis = 1, inplace = True)
        try:
            MySQL_DB().postRequest(detail_portfolio_SQL, 'coin_portfolio')
            print(f'Portfolio overview uploaded to MySQL.')
        except:
            print(f'Error in MySQL upload')
        for i in detail_portfolio_SQL.index:
            detail_portfolio_JSON = {
                'Datetime' : detail_portfolio_JSON.loc[i, 'Datetime'],
                'BTC' : detail_portfolio_JSON.loc[i, 'BTC'],
                'BTC_Value' : detail_portfolio_JSON.loc[i, 'BTC_Value'],
                'BTC_Cash' : detail_portfolio_JSON.loc[i, 'BTC_Cash'],
                'BTC_Weight' : detail_portfolio_JSON.loc[i, 'BTC_Weight'],
                'ETH' : detail_portfolio_JSON.loc[i, 'ETH'],
                'ETH_Value' : detail_portfolio_JSON.loc[i, 'ETH_Value'],
                'ETH_Cash' : detail_portfolio_JSON.loc[i, 'ETH_Cash'],
                'ETH_Weight' : detail_portfolio_JSON.loc[i, 'ETH_Weight'],
                'ADA' : detail_portfolio_JSON.loc[i, 'ADA'],
                'ADA_Value' : detail_portfolio_JSON.loc[i, 'ADA_Value'],
                'ADA_Cash' : detail_portfolio_JSON.loc[i, 'ADA_Cash'],
                'ADA_Weight' : detail_portfolio_JSON.loc[i, 'ADA_Weight'],
                'Total_Value': detail_portfolio_JSON.loc[i, 'Total_Value']
            }
            try:
                Mongo_DB().postRequest(detail_portfolio_JSON, 'coin_portfolio')
                print(f'Trade overview uploaded to MongoDB.')
            except:
                print(f'Error in MongoDB upload')
    def getPortfolio(self):
        self.updatePortfolio()
        portfolio = MySQL_DB().getRequest(table = 'coin_portfolio', columns = '*')
        return portfolio