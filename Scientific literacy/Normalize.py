import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def minmaxscaler(x, direction):
    if direction == "+" or direction == 1:
        x = np.array(x)
    if direction == "-" or direction == -1:
        x = 1/np.array(x)
    y = (x - x.min())/(x.max()-x.min())
    return y


def minmaxscaler_log(x, direction):                               # 对数型
    if direction == "+" or direction == 1:
        x = np.array(x)
    if direction == "-" or direction == -1:
        x = 1/np.array(x)
    y = (np.log(x) - np.log(x.min())) / (np.log(x.max()) - np.log(x.min()))
    return y


def efficiency_coefficient(x, a, b, direction, method):
    if direction == "+" or direction == 1:
        x = np.array(x)
    if direction == "-" or direction == -1:
        x = 1/np.array(x)
        a = 1/a
        b = 1/b
    if method == "linear":
        y = (x - a) / (b - a)
    if method == "log":
        y = (np.log(x) - np.log(a)) / (np.log(b) - np.log(a))
    return y


def compress(x, method, direction):
    if direction == 1:
        x = np.array(x)
    if direction == -1:
        x = 1/np.array(x)
    if method == "max":
        y = x/x.max()
    if method == "min":
        y = x.min()/x
    if method == "standard":
        y = (x-x.mean())/x.std()
    if method == "L1normalize":
        y = x / x.mean()
    if method == "L2normalize":
        s = x**2
        d = np.sqrt(s.sum())
        y = x/d
    return y

if __name__ == "__main__":
    X = np.linspace(1, 5, 100)
    Y = efficiency_coefficient(X, 2,4,-1,"linear")
    plt.plot(X, Y)
    plt.plot(X, np.zeros(100))
    plt.plot(X, np.ones(100))
    plt.plot(2*np.ones(100), np.linspace(-1,1.5,100))
    plt.plot(4*np.ones(100), np.linspace(-1,1.5,100))
    plt.show()

    '''
    X = np.linspace(1, 5, 100)
    X = X.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    Y = scaler.transform(X)
    plt.plot(X, Y)
    plt.show()
    '''
