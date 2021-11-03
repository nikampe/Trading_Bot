import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

def plotPacf(data):
    plot_pacf(data, lags = 100)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.title('PACF Plot', size = 14)
    plt.grid(True)
    plt.show()

def plotForecasts(data, data_predicted, data_forecasted, horizon, timesteps):
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
