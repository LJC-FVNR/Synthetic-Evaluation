import numpy as np
import matplotlib.pyplot as plt
import Normalize as N
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neural_network import MLPRegressor
import mglearn
if __name__ == "__main__":
    air = np.array([300, 0.5, 40, 0.5, 2, 4, 4])
    road = np.array([80, 4, 50, 8, 5, 5, 2])
    rail = np.array([30, 11, 10, 0.002, 3, 1, 5])
    comb = np.array([55, 9, 18, 3, 4, 3, 3])

    trans_cost = np.array([air[0], road[0], rail[0], comb[0]])
    time = np.array([air[1], road[1], rail[1], comb[1]])
    turns = np.array([air[2], road[2], rail[2], comb[2]])
    energy = np.array([air[3], road[3], rail[3], comb[3]])
    reliability = np.array([air[4], road[4], rail[4], comb[4]])
    flexible = np.array([air[5], road[5], rail[5], comb[5]])
    out = np.array([air[6], road[6], rail[6], comb[6]])
    direction = [-1, -1, 1, -1, 1, 1, 1]
    label = ["trans_cost", "time", "turns", "energy", "reliability", "flexible", "ecology"]

    total = np.array([trans_cost, time, turns, energy, reliability, flexible, out])
    for i in range(0, 7):
        total[i] = N.minmaxscaler_log(total[i], direction=direction[i])

    total = total.T

    ycon = np.array([84.72893513, 82.0835094, 75.51052081, 75.30841115])
    X_train = total[0:3]
    y_train = ycon[0:3]
    mlp = MLPRegressor(solver='lbfgs',activation='tanh', random_state=0, hidden_layer_sizes=[10]).fit(X_train, y_train)

