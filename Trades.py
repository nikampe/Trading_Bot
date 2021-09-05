import pandas as pd
import numpy as np
import datetime as dt

from DATABASE.db import DB

class Trades:
    def __init__(self):
        self.dbname = 'trades'
    def getTrades(self):
        trades = DB().getRequest(table = 'trades', columns = '*')
        return trades
    def updateTrades(self, datetime, ticker, type, price, amount):
        data = {'Dateimte': datetime, 'Ticker': ticker, 'Type': type, 'Price': price, 'Amount': amount}
        trades = pd.DataFrame(data = data)
        DB().postRequest(trades, self.dbname)
