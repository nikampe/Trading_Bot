import re
import sys, os
import pandas as pd
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from newsapi import NewsApiClient

sys.path.insert(1, os.getcwd())
from API.krakenapi import CoinAPI
from Portfolio import Portfolio
from Trades import Trades
from DATABASE.db import MySQL_DB, Mongo_DB
from Assets import BTC
from Assets import ETH
from Assets import ADA
from UTILITIES.plots import plotForecasts, plotPacf

NEWS_API_KEY = "9c0186d680ce46a4937b9091742015d9"

# nltk.downloader.download('vader_lexicon')
# nltk.downloader.download('stopwords')
# nltk.downloader.download('punkt')
# nltk.downloader.download('wordnet')

class NLPAnalysis:
    def __init__(self, api_key, coin):
        self.api_key = api_key
        self.coin = coin
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
    def returns(self):
        returns = self.coin.getTrainingData()[['Date', 'Return']]
        returns['Date'] = pd.to_datetime(returns['Date'])
        returns['Date'] = returns['Date'].dt.strftime('%Y-%m-%d')
        returns.reset_index(inplace = True, drop = True)
        return returns
    def data(self):
        today = dt.datetime.today()
        start_date = (today - timedelta(days = 30)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        newsapi = NewsApiClient(api_key = self.api_key)
        bitcoin_headlines = newsapi.get_everything(q = "bitcoin", language = "en", page_size = 100, sort_by = "relevancy", from_param = start_date, to = end_date)
        return bitcoin_headlines
    def tokenizer(self, text):
        sw_addons = {'also', 'since', 'youve'}
        # Create a list of the words
        sw = set(stopwords.words('english'))
        # Converting to lowercase
        regex = re.compile("[^a-zA-Z ]")
        # Removing punctuation
        re_clean = regex.sub('', text)
        # Removing stop words
        words = word_tokenize(re_clean)
        # Lemmatize Words into root words
        lem = [self.lemmatizer.lemmatize(word) for word in words]
        tokens = [word.lower() for word in lem if word.lower() not in sw.union(sw_addons)]
        return tokens
    def Lexicon_Model(self):
        data = self.data()
        sentiments = []
        # Lexicon-based Model
        for article in data["articles"]:
            try:
                text = article["content"]
                date = article["publishedAt"][:10]
                sentiment = self.analyzer.polarity_scores(text)
                compound = sentiment["compound"]
                pos = sentiment["pos"]
                neu = sentiment["neu"]
                neg = sentiment["neg"]
                sentiments.append({"Text": text, "Date": date, "Compound": compound, "Positive": pos, "Negative": neg, "Neutral": neu})
            except AttributeError:
                pass
        # Creating News-based Data Frame
        df = pd.DataFrame(sentiments)
        df = df[["Date", "Compound", "Negative", "Neutral", "Positive", "Text"]]
        df = df.sort_values(by = ['Date'])
        df.reset_index(drop = True, inplace = True)
        # Tokenizing
        df['Tokens'] = df['Text'].apply(self.tokenizer)
        # Creating Daily Data Frame
        dates = df['Date'].unique()
        df_daily = pd.DataFrame()
        for date in dates:
            df_filtered = df[df['Date'] == date]
            mean_compound = df_filtered['Compound'].mean()
            df_daily.loc[len(df_daily), 'Date'] = date
            df_daily.loc[len(df_daily) - 1, 'Mean_Compound'] = mean_compound
        # Moving Averages of Compound Sentiment
        period = 5
        df_daily['5d_SMA'] = df_daily['Mean_Compound'].rolling(window = period).mean()
        # Plot
        returns = self.returns()
        returns_filtered = returns[returns['Date'].isin(dates)]
        returns_filtered.reset_index(inplace = True, drop = True)
        n = len(df_daily['Mean_Compound'])
        plt.plot(range(0, n), df_daily['Mean_Compound'], color = 'black', label = 'Daily Compounded Sentiment')
        plt.plot(range(0, n), df_daily['5d_SMA'], color = 'red', label = '5-Day Sentiment MA')
        plt.plot(range(0, n-1), returns_filtered['Return'], color = 'green', label = 'Daily BTC Returns')
        plt.axvline(x = n-1, linestyle = ':', color = 'k')
        plt.legend()
        plt.show()
    def ML_Model(self):
        pass

# if __name__ == '__main__':
#     NLPAnalysis(NEWS_API_KEY, BTC).Lexicon_Model()