import pickle
import matplotlib.pyplot as plt
import numpy as np
from Utils import *

def makeGraph(x, y, c):
    plt.scatter(x, y, c=c)

file = open('behaviorSpace', 'rb')
data = pickle.load(file)
file.close()

file = open('modelParams', 'rb')
modelParams = pickle.load(file)
file.close()

for i, score in enumerate(data['scores']):
    if score>0.07:
        del data['seperability'][i]
        del data['generalizability'][i]
        del data['scores'][i]
        del data['mc'][i]
        del modelParams[i]
        
def getGeneralizability(noise, models):
    grs = []
    for model in models:
        grs.append(calculateGeneralization(model['params'], model['creator'], model['testSet'], noise))
    return grs

noises = [0.01, 0.05]
for noise in noises:
    grs = getGeneralizability(noise, modelParams)
    corr = np.corrcoef(grs, data['scores'])
    print("Correlation at noise {}: {}".format(noise, corr[1, 0]))
    plt.figure()
    plt.scatter(grs, data['scores'])
    plt.xlabel("GR")
    plt.ylabel("Error")