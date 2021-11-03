import pandas as pd
import numpy as np
import datetime as dt

from DATABASE.db import MySQL_DB, Mongo_DB

class Trades:
    def __init__(self):
        self.dbname = 'trades'
    def getTrades(self):
        trades = MySQL_DB().getRequest(table = 'trades', columns = '*')
        return trades
    def updateTrades(self, datetime, ticker, type, price, amount):
        data = {'Dateimte': datetime, 'Ticker': ticker, 'Type': type, 'Price': price, 'Amount': amount}
        trades_SQL = pd.DataFrame(data = data)
        try:
            MySQL_DB().postRequest(trades_SQL, self.dbname)
            print(f'Trade overview uploaded to MySQL.')
        except:
            print(f'Error in MySQL upload')
        for i in trades_SQL.index:
            trades_JSON = {
                'Date' : trades_SQL.loc[i, 'Date'],
                'Ticker' : trades_SQL.loc[i, 'Ticker'],
                'Type': trades_SQL.loc[i, 'Type'],
                'Price': trades_SQL.loc[i, 'Price'],
                'Amount': trades_SQL.loc[i, 'Amount']
            }
            try:
                Mongo_DB().postRequest(trades_JSON, self.dbname)
                print(f'Trade overview uploaded to MongoDB.')
            except:
                print(f'Error in MongoDB upload')
