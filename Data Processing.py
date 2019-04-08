import numpy as np
import matplotlib.pyplot as plt
import Normalize as N
from sklearn.decomposition import PCA
import pandas as pd

air = np.array([300, 0.5, 40, 0.5, 2, 4, 4])
road = np.array([80, 4, 50, 8, 5, 5, 2])
rail = np.array([30, 11, 10, 0.002, 3, 1, 5])
comb = np.array([55, 9, 18, 3, 4, 3	, 3])

trans_cost = np.array([air[0], road[0], rail[0], comb[0]])
time = np.array([air[1], road[1], rail[1], comb[1]])
turns = np.array([air[2], road[2], rail[2], comb[2]])
energy = np.array([air[3], road[3], rail[3], comb[3]])
reliability = np.array([air[4], road[4], rail[4], comb[4]])
flexible = np.array([air[5], road[5], rail[5], comb[5]])
out = np.array([air[6], road[6], rail[6], comb[6]])
direction = [-1, -1, 1, -1, 1, 1, 1]
label = ["trans_cost", "time", "turns", "energy", "reliability", "flexible", "ecology"]

total = np.array([trans_cost,time,turns,energy,reliability,flexible,out])

for i in range(0,7):
    total[i] = N.minmaxscaler_log(total[i], direction=direction[i])

pca = PCA(n_components=2)
pca.fit(total.T)

time_pca = pca.transform(total.T)
'''
plt.plot(time_pca[:,0],time_pca[:,1],"o")

for i in range(0, 7):
    plt.text(time_pca[i,0], time_pca[i,1], label[i], family='serif', ha='right', wrap=True)
plt.show()
'''