import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

sys.path.insert(1, os.getcwd())
from API.krakenapi import CoinAPI
from Portfolio import Portfolio
from Trades import Trades
from DATABASE.db import MySQL_DB, Mongo_DB
from Assets import BTC
from Assets import ETH
from Assets import ADA
from UTILITIES.plots import plotForecasts, plotPacf

class TimeSeriesForecasting:
    def __init__(self, coin, data):
        self.coin = coin
        self.data = data
    def AR_Model(self):
        coin = self.coin
        data = self.data
        returns = data('Return')
        ar_orders = range(1, 13)
        model_results = pd.DataFrame()
        for p in ar_orders:
            model = ARIMA(returns, order = (p, 0, 0))
            model_fit = model.fit()
            model_results.loc[len(model_results), 'p'] = p
            model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_AR_model = model_results.min(axis = 1, level = 1)
        best_model = ARIMA(returns, order = (best_AR_model['p'], 0, 0))
        best_model_fit = best_model.fit()
        best_model_fit.save(f'/MODELS/AR/{coin.abbreviation}_AR.pkl')
    def ARMA_Model(self):
        coin = self.coin
        data = self.data
        returns = data('Return')
        ar_orders = range(0, 13)
        ma_orders = range(0, 13)
        model_results = pd.DataFrame()
        for p in ar_orders:
            for q in ma_orders:
                model = ARIMA(returns, order = (p, 0, q))
                model_fit = model.fit()
                model_results.loc[len(model_results), 'p'] = p
                model_results.loc[len(model_results)-1, 'q'] = p
                model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_ARMA_model = model_results.min()
        best_model = ARIMA(returns, order = (best_ARMA_model['p'], 0, best_ARMA_model['q']))
        best_model_fit = best_model.fit()
        best_model_fit.save(f'/MODELS/ARMA/{coin.abbreviation}_ARMA.pkl')
    def ARIMA_Model(self):
        coin = self.coin
        data = self.data
        returns = data('Return')
        ar_orders = range(0, 13)
        ma_orders = range(0, 13)
        diff_orders = range(0, 5)
        model_results = pd.DataFrame()
        for p in ar_orders:
            for q in ma_orders:
                for i in diff_orders:
                    model = ARIMA(returns, order = (p, i, q))
                    model_fit = model.fit()
                    model_results.loc[len(model_results), 'p'] = p
                    model_results.loc[len(model_results)-1, 'q'] = p
                    model_results.loc[len(model_results)-1, 'i'] = i
                    model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_ARMA_model = model_results.min()
        best_model = ARIMA(returns, order = (best_ARMA_model['p'], best_ARMA_model['i'], best_ARMA_model['q']))
        best_model_fit = best_model.fit()
        best_model_fit.save(f'/MODELS/ARIMA/{coin.abbreviation}_ARIMA.pkl')
    def SARIMA_Model(self):
        pass
    def Monte_Carlo_Model(self):
        pass

# class TimeSeriesForecasting:
#     def __init__(self, coin, data):
#         self.coin = coin
#         self.data = data
#     def prices(self):
#         prices = self.coin.getTrainingData()['Close']
#         prices = prices.tail(365)
#         prices.reset_index(inplace = True, drop = True)
#         prices = prices.to_numpy()
#         return prices
#     def returns(self):
#         returns = self.coin.getTrainingData()['Return']
#         returns = returns.tail(365)
#         returns.reset_index(inplace = True, drop = True)
#         returns = returns.to_numpy()
#         return returns
#     def AR_Model(self):
#         ar_orders = range(1, 13)
#         model_results = pd.DataFrame()
#         returns = self.returns()
#         for p in ar_orders:
#             model = ARIMA(returns, order = (p, 0, 0))
#             model_fit = model.fit()
#             model_results.loc[len(model_results), 'p'] = p
#             model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
#         best_AR_model = model_results.min(axis = 1, level = 1)
#         best_model = ARIMA(returns, order = (best_AR_model['p'], 0, 0))
#         best_model_fit = best_model.fit()
#         best_model_fit.save('/MODELS/AR/BTC_AR.pkl')
#     def ARMA_Model(self):
#         ar_orders = range(0, 13)
#         ma_orders = range(0, 13)
#         model_results = pd.DataFrame()
#         for p in ar_orders:
#             for q in ma_orders:
#                 model = ARIMA(self.returns(), order = (p, 0, q))
#                 model_fit = model.fit()
#                 model_results.loc[len(model_results), 'p'] = p
#                 model_results.loc[len(model_results)-1, 'q'] = p
#                 model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
#         best_ARMA_model = model_results.min()
#         best_model = ARIMA(self.returns(), order = (best_ARMA_model['p'], 0, best_ARMA_model['q']))
#         best_model_fit = best_model.fit()
#         best_model_fit.save('/MODELS/ARMA/BTC_ARMA.pkl')
#     def ARIMA_Model(self):
#         ar_orders = range(0, 13)
#         ma_orders = range(0, 13)
#         diff_orders = range(0, 5)
#         model_results = pd.DataFrame()
#         for p in ar_orders:
#             for q in ma_orders:
#                 for i in diff_orders:
#                     model = ARIMA(self.returns(), order = (p, i, q))
#                     model_fit = model.fit()
#                     model_results.loc[len(model_results), 'p'] = p
#                     model_results.loc[len(model_results)-1, 'q'] = p
#                     model_results.loc[len(model_results)-1, 'i'] = i
#                     model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
#         best_ARMA_model = model_results.min()
#         best_model = ARIMA(self.returns(), order = (best_ARMA_model['p'], best_ARMA_model['i'], best_ARMA_model['q']))
#         best_model_fit = best_model.fit()
#         best_model_fit.save('/MODELS/ARIMA/BTC_ARIMA.pkl')
#     def SARIMA_Model(self):
#         pass
#     def Monte_Carlo_Model(self):
#         pass