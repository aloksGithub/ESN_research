import reservoirpy as rpy
from reservoirpy.datasets import (lorenz, mackey_glass, narma)
from reservoirpy.observables import (rmse, rsquare, nrmse, mse)
import numpy as np
import math
import pandas as pd
from functools import partial
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the parent directory to the sys.path list
sys.path.append(os.path.abspath('../'))
from NAS import NAS
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

rpy.verbosity(0)
output_dim = 1

def nmse(y, o):
    assert len(o) == len(y), "Both arrays must have the same length."
    sigma2 = np.var(y)  # variance of y
    i = len(y)
    error = sum((o[t] - y[t])**2 for t in range(i))
    return error / (i * sigma2)

def mae(y, o):
    assert len(o) == len(y), "Both arrays must have the same length."
    i = len(y)
    error = sum(abs(o[t] - y[t]) for t in range(i))
    return error / i

def mape(y, o):
    assert len(o) == len(y), "Both arrays must have the same length."
    i = len(y)
    error = sum(abs(o[t] - y[t]) / abs(y[t]) for t in range(i))
    return (error / i) * 100

def getData():
    sunspots = pd.read_csv("../datasets/Sunspots.csv")
    data = sunspots.loc[:,"Monthly Mean Total Sunspot Number"]
    data = np.expand_dims(data, axis=1)
    train = data[:2000]
    val = data[2000:2500]
    test = data[2500:]
    trainX = train[:-1]
    trainY = train[1:]
    valX = val[:-1]
    valY = val[1:]
    testX = test[:-1]
    testY = test[1:]
    return trainX, trainY, valX, valY, testX, testY

if __name__ == "__main__":
    trainX, trainY, valX, valY, testX, testY = getData()

    gaParams = {
        "evaluator": partial(NAS.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY),
        "generator": partial(NAS.generateRandomArchitecture, sampleX=trainX[:2], sampleY=trainY[:2]),
        "populationSize": 5,
        "eliteSize": 1,
        "stagnationReset": 5,
        "generations": 10,
        "minimizeFitness": True,
        "logModels": True,
        "seedModels": [],
        "crossoverProbability": 0.7,
        "mutationProbability": 0.2,
        "earlyStop": 0,
        "n_jobs": 5
    }

    nmseErrors = []
    maeErrors = []
    mapeErrors = []
    for i in range(1):
        models, performances, architectures = NAS.runGA(gaParams)
        model = models[0]
        preds = NAS.runModel(model, testX)
        
        performance = nmse(testY, preds)
        print("Performance", nmse(testY, preds), mae(testY, preds), mape(testY, preds))
        nmseErrors.append(nmse(testY, preds))
        maeErrors.append(mae(testY, preds))
        mapeErrors.append(mape(testY, preds))

    print(np.array(nmseErrors).mean(), np.array(nmseErrors).std())
    print(np.array(maeErrors).mean(), np.array(maeErrors).std())
    print(np.array(mapeErrors).mean(), np.array(mapeErrors).std())