import sys, os
import numpy as np
import tensorflow as tf
from keras import models
from keras.models import save_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras_tuner.tuners import BayesianOptimization

sys.path.insert(1, os.getcwd())
from Assets import BTC
from Assets import ETH
from Assets import ADA

# from API.krakenapi import CoinAPI
# from Portfolio import Portfolio
# from Trades import Trades
# from DATABASE.db import MySQL_DB, Mongo_DB
# from UTILITIES.plots import plotForecasts, plotPacf

class MLForecasting:
    def __init__(self, data):
        self.data = data
    def LSTM_Model(self, data):
        wd = os.getcwd()
        os.chdir(wd + "/MODELS/LSTM")
        # Data
        data = self.data
        data = data.tail(365)
        data.reset_index(inplace = True, drop = True)
        # Normalization
        data_min = min(data)
        data_max = max(data)
        data = (data-data_min)/(data_max-data_min)
        # Data preparation
        n_timesteps = 14
        n_features = 1
        features_set = []
        labels = []
        for i in range(n_timesteps, data.shape[0]):
            features_set.append(data[i-n_timesteps:i])
            labels.append(data[i])
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
        # Train-Test Split
        n = features_set.shape[0]
        n_train = int(n * 0.80)
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
        tuner = Tuner(model, objective = 'mse', max_trials = 5, executions_per_trial = 1, overwrite = True)
        tuner.search(x = features_set_train, y = labels_train, validation_data = (features_set_test, labels_test))
        # Model Evaluation
        best_model = tuner.get_best_models(num_models = 1)[0]
        save_model(best_model, self.coin.abbreviation + '_LSTM.h5')
        best_model.fit(features_set_train, labels_train)
        # Forecast
        n = len(data)
        p = features_set.shape[0]
        q = features_set.shape[1]
        m = 50
        data_train = []
        data_train = features_set
        data_forecasted = []
        for t in range(0, m):
            data_train = data_train[-1].reshape(1,q,1)
            forecast = best_model.predict(data_train)
            residuals = data_train[-1][1:q]
            residuals = np.append(residuals, forecast)
            data_train = np.concatenate((data_train, residuals.reshape(1,q,1)), axis = 0)
            data_forecasted = np.append(data_forecasted, forecast)
        # Rescaling
        data = data_min + data * (data_max - data_min)
        data_predicted = best_model.predict(features_set)
        data_predicted = data_min + data_predicted * (data_max - data_min)
        data_forecasted = data_min + data_forecasted * (data_max - data_min)
        # Reset Working Directory
        os.chdir(wd)
        # Return Forecasted Data
        return data_forecasted
    def GRU_Model(self):
        wd = os.getcwd()
        os.chdir(wd + "/MODELS/GRU")
        # Data
        data = self.data
        data = data.tail(365)
        data.reset_index(inplace = True, drop = True)
        # Normalization
        data_min = min(data)
        data_max = max(data)
        data = (data-data_min)/(data_max-data_min)
        # Data preparation
        n_timesteps = 14
        n_features = 1
        features_set = []
        labels = []
        for i in range(n_timesteps, data.shape[0]):
            features_set.append(data[i-n_timesteps:i])
            labels.append(data[i])
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
        n = len(data)
        p = features_set.shape[0]
        q = features_set.shape[1]
        m = 50
        data_train = []
        data_train = features_set
        data_forecasted = []
        for t in range(0, m):
            data_train = data_train[-1].reshape(1,q,1)
            forecast = best_model.predict(data_train)
            residuals = data_train[-1][1:q]
            residuals = np.append(residuals, forecast)
            data_train = np.concatenate((data_train, residuals.reshape(1,q,1)), axis = 0)
            data_forecasted = np.append(data_forecasted, forecast)
        # Rescaling
        data = data_min + data * (data_max - data_min)
        data_predicted = best_model.predict(features_set)
        data_predicted = data_min + data_predicted * (data_max - data_min)
        data_forecasted = data_min + data_forecasted * (data_max - data_min)
        # Reset Working Directory
        os.chdir(wd)
        # Return Forecasted Data
        return data_forecasted

# class MLForecasting:
#     def __init__(self, coin, coins):
#         self.coin = coin
#         self.coins = coins
#     def prices(self):
#         prices = self.coin.getTrainingData()['Close']
#         prices = prices.tail(365)
#         prices.reset_index(inplace = True, drop = True)
#         return prices
#     def returns(self):
#         returns = self.coin.getTrainingData()['Return']
#         returns = returns.tail(365)
#         returns.reset_index(inplace = True, drop = True)
#         returns = returns.to_numpy()
#         return returns
#     def plot(self):
#         returns = self.returns()
#         prices = self.prices()
#         plotPacf(returns)
#         plotPacf(prices)
#     def LSTM_Model(self):
#         wd = os.getcwd()
#         os.chdir(wd + "/MODELS/LSTM")
#         # Data
#         data = self.prices()
#         # Normalization
#         data_min = min(data)
#         data_max = max(data)
#         data = (data-data_min)/(data_max-data_min)
#         # Data preparation
#         n_timesteps = 14
#         n_features = 1
#         features_set = []
#         labels = []
#         for i in range(n_timesteps, data.shape[0]):
#             features_set.append(data[i-n_timesteps:i])
#             labels.append(data[i])
#         features_set, labels = np.array(features_set), np.array(labels)
#         features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
#         # Train-Test Split
#         n = features_set.shape[0]
#         n_train = int(n * 0.80)
#         features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
#         labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
#         # Model
#         def model(hp):
#             model = models.Sequential()
#             model.add(LSTM(hp.Int('input_unit', min_value = 4, max_value = 64, step = 2), return_sequences = True, input_shape = (n_timesteps, n_features)))
#             for i in range(hp.Int('n_layers', 1, 6)):
#                 model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value = 4, max_value = 64, step = 2), return_sequences = True))
#             model.add(LSTM(hp.Int('layer_2_neurons', min_value = 4, max_value = 64, step = 2)))
#             model.add(Dropout(hp.Float('Dropout_rate', min_value = 0.0, max_value = 0.5, step = 0.05)))
#             model.add(Dense(1, activation = hp.Choice('dense_activation', values = ['relu', 'sigmoid', 'tanh', 'linear'], default = 'relu')))
#             model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])), metrics = ['mse'])
#             return model
#         # Hyperparameter Optimization
#         class Tuner(BayesianOptimization):
#             def run_trial(self, trial, *args, **kwargs):
#                 kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 512, step = 32)
#                 kwargs['epochs'] = trial.hyperparameters.Int('epochs', 16, 128, step = 8)
#                 super(Tuner, self).run_trial(trial, *args, **kwargs)
#         tuner = Tuner(model, objective = 'mse', max_trials = 5, executions_per_trial = 1, overwrite = True)
#         tuner.search(x = features_set_train, y = labels_train, validation_data = (features_set_test, labels_test))
#         # Model Evaluation
#         best_model = tuner.get_best_models(num_models = 1)[0]
#         save_model(best_model, self.coin.abbreviation + '_LSTM.h5')
#         best_model.fit(features_set_train, labels_train)
#         # Forecast
#         n = len(data)
#         p = features_set.shape[0]
#         q = features_set.shape[1]
#         m = 50
#         data_train = []
#         data_train = features_set
#         data_forecasted = []
#         for t in range(0, m):
#             data_train = data_train[-1].reshape(1,q,1)
#             forecast = best_model.predict(data_train)
#             residuals = data_train[-1][1:q]
#             residuals = np.append(residuals, forecast)
#             data_train = np.concatenate((data_train, residuals.reshape(1,q,1)), axis = 0)
#             data_forecasted = np.append(data_forecasted, forecast)
#         # Rescaling
#         data = data_min + data * (data_max - data_min)
#         data_predicted = best_model.predict(features_set)
#         data_predicted = data_min + data_predicted * (data_max - data_min)
#         data_forecasted = data_min + data_forecasted * (data_max - data_min)
#         # Forecast Plot
#         plotForecasts(data, data_predicted, data_forecasted, m, n_timesteps)
#         # Reset Working Directory
#         os.chdir(wd)
#         # Return Forecasted Data
#         return data_forecasted
#     def GRU_Model(self):
#         wd = os.getcwd()
#         os.chdir(wd + "/MODELS/GRU")
#         # Data
#         data = self.prices()
#         # Normalization
#         data_min = min(data)
#         data_max = max(data)
#         data = (data-data_min)/(data_max-data_min)
#         # Data preparation
#         n_timesteps = 14
#         n_features = 1
#         features_set = []
#         labels = []
#         for i in range(n_timesteps, data.shape[0]):
#             features_set.append(data[i-n_timesteps:i])
#             labels.append(data[i])
#         features_set, labels = np.array(features_set), np.array(labels)
#         features_set = np.reshape(features_set,(features_set.shape[0], features_set.shape[1], 1))
#         # Train-Test Split
#         n = features_set.shape[0]
#         n_train = int(n * 0.80)
#         n_test = n - n_train
#         features_set_train, features_set_test =  features_set[0:n_train,:,:], features_set[n_train:n,:,:]
#         labels_train, labels_test =  labels[0:n_train], labels[n_train:n]
#         # Model
#         def model(hp):
#             model = models.Sequential()
#             model.add(GRU(hp.Int('input_unit', min_value = 4, max_value = 64, step = 2), return_sequences = True, input_shape = (n_timesteps, n_features)))
#             for i in range(hp.Int('n_layers', 1, 6)):
#                 model.add(GRU(hp.Int(f'lstm_{i}_units', min_value = 4, max_value = 64, step = 2), return_sequences = True))
#             model.add(GRU(hp.Int('layer_2_neurons', min_value = 4, max_value = 64, step = 2)))
#             model.add(Dropout(hp.Float('Dropout_rate', min_value = 0.0, max_value = 0.5, step = 0.05)))
#             model.add(Dense(1, activation = hp.Choice('dense_activation', values = ['relu', 'sigmoid', 'tanh', 'linear'], default = 'relu')))
#             model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])), metrics = ['mse'])
#             return model
#         # Hyperparameter Optimization
#         class Tuner(BayesianOptimization):
#             def run_trial(self, trial, *args, **kwargs):
#                 kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 512, step = 32)
#                 kwargs['epochs'] = trial.hyperparameters.Int('epochs', 16, 128, step = 8)
#                 super(Tuner, self).run_trial(trial, *args, **kwargs)
#         tuner = Tuner(model, objective = 'mse', max_trials = 20, executions_per_trial = 1, overwrite = False)
#         tuner.search(x = features_set_train, y = labels_train, validation_data = (features_set_test, labels_test))
#         # Model Evaluation
#         best_model = tuner.get_best_models(num_models = 1)[0]
#         save_model(best_model, self.coin.abbreviation + '_GRU.h5')
#         best_model.fit(features_set_train, labels_train)
#         # Forecast
#         n = len(data)
#         p = features_set.shape[0]
#         q = features_set.shape[1]
#         m = 50
#         data_train = []
#         data_train = features_set
#         data_forecasted = []
#         for t in range(0, m):
#             data_train = data_train[-1].reshape(1,q,1)
#             forecast = best_model.predict(data_train)
#             residuals = data_train[-1][1:q]
#             residuals = np.append(residuals, forecast)
#             data_train = np.concatenate((data_train, residuals.reshape(1,q,1)), axis = 0)
#             data_forecasted = np.append(data_forecasted, forecast)
#         # Rescaling
#         data = data_min + data * (data_max - data_min)
#         data_predicted = best_model.predict(features_set)
#         data_predicted = data_min + data_predicted * (data_max - data_min)
#         data_forecasted = data_min + data_forecasted * (data_max - data_min)
#         # Forecast Plot
#         plotForecasts(data, data_predicted, data_forecasted, m, n_timesteps)
#         # Reset Working Directory
#         os.chdir(wd)
#         # Return Forecasted Data
#         return data_forecasted