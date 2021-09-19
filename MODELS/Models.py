import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from keras.models import save_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras_tuner.tuners import BayesianOptimization
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import sys, os

sys.path.insert(1, os.getcwd()) 
from Assets import BTC
from Assets import ETH
from Assets import ADA

def plot_pacf(data):
    plot_pacf(data, lags = 100)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.title('PACF Plot', size = 14)
    plt.grid(True)
    plt.show()

def plot_forecasts(data, data_predicted, data_forecasted, horizon, timesteps):
    m = horizon
    n = len(data)
    plt.plot(range(1, n+1), data, linestyle = ':', marker = 'o', color = 'blue', label = "Prices")
    plt.plot(range(timesteps+1, n+1), data_predicted, linestyle = ':', marker = 'o', mfc = 'none', color = 'green', label = "Prices Predicted (" + str(timesteps) + "-lag dependent)")
    plt.plot(range(n+1, n+m+1), data_forecasted, linestyle = ':', marker = 'o', color = 'green', label = "Prices Forecasted")
    plt.axvline(x = n, linestyle = ':', color = 'k')
    plt.title("Prices, Fitted Prices and Forecasted Prices", fontsize = 16)
    plt.legend()
    plt.xlim(n-365, n+m+1)
    plt.show()

class TimeSeriesModels:
    def __init__(self, coin, coins):
        self.coin = coin
        self.coins = coins
    def prices(self):
        prices = self.coin.getTrainingData()['Close']
        prices = prices.tail(365)
        prices.reset_index(inplace = True, drop = True)
        prices = prices.to_numpy()
        return prices
    def returns(self):
        returns = self.coin.getTrainingData()['Return']
        returns = returns.tail(365)
        returns.reset_index(inplace = True, drop = True)
        returns = returns.to_numpy()
        return returns
    def AR_Model(self):
        ar_orders = range(1, 13)
        model_results = pd.DataFrame()
        for p in ar_orders:
            model = ARIMA(self.returns(), order = (p, 0, 0))
            model_fit = model.fit()
            model_results.loc[len(model_results), 'p'] = p
            model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_AR_model = model_results.min(axis = 1, level = 1)
        best_model = ARIMA(self.returns(), order = (best_AR_model['p'], 0, 0))
        best_model_fit = best_model.fit()
        best_model_fit.save('/MODELS/AR/BTC_AR.pkl')
    def ARMA_Model(self):
        ar_orders = range(0, 13)
        ma_orders = range(0, 13)
        model_results = pd.DataFrame()
        for p in ar_orders:
            for q in ma_orders:
                model = ARIMA(self.returns(), order = (p, 0, q))
                model_fit = model.fit()
                model_results.loc[len(model_results), 'p'] = p
                model_results.loc[len(model_results)-1, 'q'] = p
                model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_ARMA_model = model_results.min()
        best_model = ARIMA(self.returns(), order = (best_ARMA_model['p'], 0, best_ARMA_model['q']))
        best_model_fit = best_model.fit()
        best_model_fit.save('/MODELS/ARMA/BTC_ARMA.pkl')
    def ARIMA_Model(self):
        ar_orders = range(0, 13)
        ma_orders = range(0, 13)
        diff_orders = range(0, 5)
        model_results = pd.DataFrame()
        for p in ar_orders:
            for q in ma_orders:
                for i in diff_orders:
                    model = ARIMA(self.returns(), order = (p, i, q))
                    model_fit = model.fit()
                    model_results.loc[len(model_results), 'p'] = p
                    model_results.loc[len(model_results)-1, 'q'] = p
                    model_results.loc[len(model_results)-1, 'i'] = i
                    model_results.loc[len(model_results)-1, 'BIC'] = model_fit.bic
        best_ARMA_model = model_results.min()
        best_model = ARIMA(self.returns(), order = (best_ARMA_model['p'], best_ARMA_model['i'], best_ARMA_model['q']))
        best_model_fit = best_model.fit()
        best_model_fit.save('/MODELS/ARIMA/BTC_ARIMA.pkl')
    def SARIMA_Model(self):
        pass
    def MonteCarlo_Model(self):
        pass

class MLModels:
    def __init__(self, coin, coins):
        self.coin = coin
        self.coins = coins
    def prices(self):
        prices = self.coin.getTrainingData()['Close']
        prices = prices.tail(365)
        prices.reset_index(inplace = True, drop = True)
        return prices
    def LSTMForecast(self):
        wd = os.getcwd()
        os.chdir(wd + "/MODELS/LSTM")
        # Data
        prices = self.prices()
        # Normalization
        prices_min = min(prices)
        prices_max = max(prices)
        prices = (prices-prices_min)/(prices_max-prices_min)
        # Data preparation
        n_timesteps = 14
        n_features = 1
        features_set = []
        labels = []
        for i in range(n_timesteps, prices.shape[0]):
            features_set.append(prices[i-n_timesteps:i])
            labels.append(prices[i])
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
        # Train-Test Split
        n = features_set.shape[0]
        n_train = int(n * 0.80)
        n_test = n - n_train
        features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
        labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
        # Model
        def model(hp):
            model = models.Sequential()
            model.add(LSTM(hp.Int('input_unit', min_value = 4, max_value = 64, step = 2), return_sequences = True, input_shape = (n_timesteps, n_features)))
            for i in range(hp.Int('n_layers', 1, 6)):
                model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value = 4, max_value = 64, step = 2), return_sequences = True))
            model.add(LSTM(hp.Int('layer_2_neurons', min_value = 4, max_value = 64, step = 2)))
            model.add(Dropout(hp.Float('Dropout_rate', min_value = 0.0, max_value = 0.5, step = 0.05)))
            model.add(Dense(1, activation = hp.Choice('dense_activation', values = ['relu', 'sigmoid', 'tanh', 'linear'], default = 'relu')))
            model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])), metrics = ['mse'])
            return model
        # Hyperparameter Optimization
        class Tuner(BayesianOptimization):
            def run_trial(self, trial, *args, **kwargs):
                kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 512, step = 32)
                kwargs['epochs'] = trial.hyperparameters.Int('epochs', 16, 128, step = 8)
                super(Tuner, self).run_trial(trial, *args, **kwargs)
        tuner = Tuner(model, objective = 'mse', max_trials = 20, executions_per_trial = 1, overwrite = False)
        tuner.search(x = features_set_train, y = labels_train, validation_data = (features_set_test, labels_test))
        # Model Evaluation
        best_model = tuner.get_best_models(num_models = 1)[0]
        save_model(best_model, self.coin.abbreviation + '_LSTM.h5')
        best_model.fit(features_set_train, labels_train)
        # Forecast
        n = len(prices)
        p = features_set.shape[0]
        q = features_set.shape[1]
        m = 50
        prices_train = []
        prices_train = features_set
        prices_forecasted = []
        for t in range(0, m):
            prices_train = prices_train[-1].reshape(1,q,1)
            forecast = best_model.predict(prices_train)
            residuals = prices_train[-1][1:q]
            residuals = np.append(residuals, forecast)
            prices_train = np.concatenate((prices_train, residuals.reshape(1,q,1)), axis = 0)
            prices_forecasted = np.append(prices_forecasted, forecast)
        # Rescaling
        prices = prices_min + prices * (prices_max - prices_min)
        prices_predicted = best_model.predict(features_set)
        prices_predicted = prices_min + prices_predicted * (prices_max - prices_min)
        prices_forecasted = prices_min + prices_forecasted * (prices_max - prices_min)
        # Forecast Plot
        plot_forecasts(prices, prices_predicted, prices_forecasted, m, n_timesteps)
        # Reset Working Directory
        os.chdir(wd)
    def GRUForecast(self):
        wd = os.getcwd()
        os.chdir(wd + "/MODELS/GRU")
        # Data
        prices = self.prices()
        # Normalization
        prices_min = min(prices)
        prices_max = max(prices)
        prices = (prices-prices_min)/(prices_max-prices_min)
        # Data preparation
        n_timesteps = 14
        n_features = 1
        features_set = []
        labels = []
        for i in range(n_timesteps, prices.shape[0]):
            features_set.append(prices[i-n_timesteps:i])
            labels.append(prices[i])
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
        # Train-Test Split
        n = features_set.shape[0]
        n_train = int(n * 0.80)
        n_test = n - n_train
        features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
        labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
        # Model
        # Model
        def model(hp):
            model = models.Sequential()
            model.add(GRU(hp.Int('input_unit', min_value = 4, max_value = 64, step = 2), return_sequences = True, input_shape = (n_timesteps, n_features)))
            for i in range(hp.Int('n_layers', 1, 6)):
                model.add(GRU(hp.Int(f'lstm_{i}_units', min_value = 4, max_value = 64, step = 2), return_sequences = True))
            model.add(GRU(hp.Int('layer_2_neurons', min_value = 4, max_value = 64, step = 2)))
            model.add(Dropout(hp.Float('Dropout_rate', min_value = 0.0, max_value = 0.5, step = 0.05)))
            model.add(Dense(1, activation = hp.Choice('dense_activation', values = ['relu', 'sigmoid', 'tanh', 'linear'], default = 'relu')))
            model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])), metrics = ['mse'])
            return model
        # Hyperparameter Optimization
        class Tuner(BayesianOptimization):
            def run_trial(self, trial, *args, **kwargs):
                kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 512, step = 32)
                kwargs['epochs'] = trial.hyperparameters.Int('epochs', 16, 128, step = 8)
                super(Tuner, self).run_trial(trial, *args, **kwargs)
        tuner = Tuner(model, objective = 'mse', max_trials = 20, executions_per_trial = 1, overwrite = False)
        tuner.search(x = features_set_train, y = labels_train, validation_data = (features_set_test, labels_test))
        # Model Evaluation
        best_model = tuner.get_best_models(num_models = 1)[0]
        save_model(best_model, self.coin.abbreviation + '_GRU.h5')
        best_model.fit(features_set_train, labels_train)
        # Forecast
        n = len(prices)
        p = features_set.shape[0]
        q = features_set.shape[1]
        m = 50
        prices_train = []
        prices_train = features_set
        prices_forecasted = []
        for t in range(0, m):
            prices_train = prices_train[-1].reshape(1,q,1)
            forecast = best_model.predict(prices_train)
            residuals = prices_train[-1][1:q]
            residuals = np.append(residuals, forecast)
            prices_train = np.concatenate((prices_train, residuals.reshape(1,q,1)), axis = 0)
            prices_forecasted = np.append(prices_forecasted, forecast)
        # Rescaling
        prices = prices_min + prices * (prices_max - prices_min)
        prices_predicted = best_model.predict(features_set)
        prices_predicted = prices_min + prices_predicted * (prices_max - prices_min)
        prices_forecasted = prices_min + prices_forecasted * (prices_max - prices_min)
        # Forecast Plot
        plot_forecasts(prices, prices_predicted, prices_forecasted, m, n_timesteps)
        # Reset Working Directory
        os.chdir(wd)

if __name__ == '__main__':
    MLModels(BTC, [BTC, ETH, ADA]).GRUForecasting()