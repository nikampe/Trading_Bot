import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.arima_model import ARIMAResults

from API.krakenapi import CoinAPI
from Portfolio import Portfolio
from Trades import Trades
from Assets import BTC
from Assets import ETH
from Assets import ADA

# Models
from MODELS.Chart_Pattern_Models import ChartPatternAnalysis
from MODELS.Time_Series_Models import TimeSeriesForecasting
from MODELS.ML_Models import MLForecasting
from MODELS.NLP_Models import NLPAnalysis

COINS = [BTC, ETH, ADA]
TRANSACTION_COSTS = {'Buy': 0.015, 'Sell': 0.015, 'Convert': 0.015}
SIGNAL_MARGIN = 0.05
TRADE_MARGIN = 1.005

class Data:
    def __init__(self, coin, portfolio):
        self.coin = coin
        self.portfolio = portfolio
    # Update 5min and 1say Coin Data
    def updateData(self):
        coin = self.coin
        coin.update5minData()
        coin.update1dData()
    # 5min OHLCV + Return + MA Data
    def get5minData(self):
        coin = self.coin
        data = coin.get5minData()
        return data
    # 1day OHLCV + Return Data
    def get1dData(self):
        coin = self.coin
        data = coin.get1dData()
        return data
    # Update DB Coin Portfolio (Kraken API)
    def updatePortfolio(self):
        portfolio = self.portfolio
        portfolio.updatePortfolio()
    # Get Current DB Coin Portfolio
    def getPortfolio(self):
        portfolio = self.portfolio
        current_portfolio = portfolio.getPortfolio()
        return current_portfolio

class Trade_Signals:
    def __init__(self, coin, portfolio, signal_margin, five_min_data, one_day_data):
        self.coin = coin
        self.portfolio = portfolio
        self.signal_margin = signal_margin
        self.five_min_data = five_min_data
        self.one_day_data = one_day_data
    # 5d vs. 20d Moving Average Crossover
    def MA_Signal(self):
        data = self.five_min_data
        MA_signal = ChartPatternAnalysis(data).MA_Crossover()
        if MA_signal == 'Buy':
            signal = 1
        elif MA_signal == 'Sell':
            signal = -1
        else:
            signal = 0
        return signal
    # 10d Bollinger Bands Crossover
    def Bollinger_Signal(self):
        data = self.five_min_data
        data_current = self.five_min_data
        current_price = data_current.loc[(data_current.shape[0]-1), 'Close']
        bollinger_bands = ChartPatternAnalysis(data).Bollinger_Bands()
        if current_price <= bollinger_bands['Lower'][0]:
            signal = 1
        elif current_price >= bollinger_bands['Upper'][0]:
            signal = -1
        else:
            signal = 0
        return signal
    # Autoregressive-Moving Average Return Forecast
    def ARMA_Signal(self):
        coin = self.coin
        data = self.one_day_data
        data_current = self.five_min_data
        current_price = data_current.loc[(data_current.shape[0]-1), 'Close']
        margin = self.signal_margin
        if dt.datetime.now().strftime('%H:%M:00') == '00:05:00':
            TimeSeriesForecasting(coin, data).ARMA_Model()
        ARMA_Model = ARIMAResults.load(f'./MODELS/ARMA/{coin.abbreviation}_ARMA.pkl')
        data_forecasted = ARMA_Model.forecast(steps = 10)
        if data_forecasted > current_price * (1 + margin):
            signal = 1
        elif data_forecasted <= current_price * (1 - margin):
            signal = -1
        else:
            signal = 0
        return signal
        return signal
    # LSTM Neural Network Price Forecast
    def LSTM_Signal(self):
        data = self.one_day_data
        data_current = self.five_min_data
        current_price = data_current.loc[(data_current.shape[0]-1), 'Close']
        margin = self.signal_margin
        data_forecasted = MLForecasting(data).LSTM_Model()
        if data_forecasted > current_price * (1 + margin):
            signal = 1
        elif data_forecasted <= current_price * (1 - margin):
            signal = -1
        else:
            signal = 0
        return signal
    # GRU Neural Network Price Forecast
    def GRU_Signal(self):
        data = self.one_day_data
        data_current = self.five_min_data
        current_price = data_current.loc[(data_current.shape[0]-1), 'Close']
        margin = self.signal_margin
        data_forecasted = MLForecasting(data).GRU_Model()
        if data_forecasted > current_price * (1 + margin):
            signal = 1
        elif data_forecasted <= current_price * (1 - margin):
            signal = -1
        else:
            signal = 0
        return signal
    # News-Based Sentiment Signaling
    def Sentiment_Signal(self):
        data = self.five_min_data
        sentiment_model = NLPAnalysis().Lexicon_Model()
        sentiment_signal = None
        return sentiment_signal
    # Signal Aggregation
    def Signals_Aggregated(self):
        coin = self.coin
        signals = {
            'Asset': coin.abbreviation,
            'ID': coin.id,
            'Signals': {
                'MA Crossover': self.MA_Signal(),
                'Bollinger Bands': self.Bollinger_Signal(),
                'ARMA Forecast': self.ARMA_Signal(),
                'LSTM Forecast': self.LSTM_Signal(),
                'Sentiment': self.Sentiment_Signal()
            }
        }
        print(signals)
        return signals

class Trade_Execution:
    def __init__(self, coin, signals, costs, portfolio):
        self.coin = coin
        self.signals = signals
        self.costs = costs
        self.portfolio = portfolio
    def Execution(self):
        date = dt.datetime.now().strftime('%Y/%m/%d %H:%M:00')
        # Overall Portfolio
        portfolio = self.portfolio.iloc[-1]
        portfolio_cash = 0
        # Coin-specific Portfolio
        coin = self.coin
        coin_data = self.coin.getData()
        coin_cash = portfolio[coin.abbreviation + '_Cash']
        coin_assets = portfolio[coin.abbreviation]
        coin_price = coin_data.loc[(coin_data.shape[0]-1), 'Close']
        # Signals
        signals = self.signals
        signals = list(signals)[2]
        # Signal Aggregation
        if signals == 1:
            signal_agg = 1
        if signals == -1:
            signal_agg = -1
        else:
            signal_agg = 0
        # Buy Execution
        if signal_agg == 1 and portfolio_cash > 0:
            type = 'Buy'
            pair = coin.alias + coin.countercurrency
            volume = coin_cash / (coin_price * TRADE_MARGIN)
            CoinAPI().placeBuyOrder(pair, str(volume))
            Trades().updateTrades(datetime = date, ticker = coin.abbreviation, type = type, price = coin_price, amount = volume)
        # Sell Execution
        elif signal_agg == -1 and coin_assets > 0:
            type = 'Sell'
            pair = coin.alias + coin.countercurrency
            volume = coin_assets
            CoinAPI().placeSellOrder(pair, str(volume))
            Trades().updateTrades(datetime = date, ticker = coin.abbreviation, type = type, price = coin_price, amount = volume)
        # Hold Execution
        elif signal_agg == 0:
            type = 'Hold'
            Trades().updateTrades(datetime = date, ticker = np.nan, type = np.nan, price = coin_price, amount = np.nan)
        else:
            print('Error in Trade Execution.')

def RUN(coins):
    for coin in coins:
        Data(coin).updateData()
        Trade_Execution(coin, Trade_Signals(coin, Portfolio().getPortfolio(coins), SIGNAL_MARGIN, Data(coin).get5minData(), Data(coin).get1dData()).Signals_Aggregated(), TRANSACTION_COSTS, Portfolio().getPortfolio(coins))

if __name__ == '__main__':
    RUN(COINS)