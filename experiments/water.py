import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
import time
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS import NAS_refactored
warnings.filterwarnings("ignore")
import traceback
import sys
import pandas as pd
import math
import copy
from reservoirpy.observables import (rmse, rsquare, nrmse, mse)

rpy.verbosity(0)
output_dim = 1

def getData():
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    data = water[:, 0]
    data = np.concatenate([data, water[-1, 1:19]])
    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)
    
    train_in = np.expand_dims(data[0:trainLen], axis=1)
    train_out = np.expand_dims(data[0+1:trainLen+1], axis=1)
    val_in = np.expand_dims(data[trainLen:valLen], axis=1)
    val_out = np.expand_dims(data[trainLen+1:valLen+1], axis=1)
    test = water[valLen:, 18:]
    return train_in, train_out, val_in, val_out, np.expand_dims(data, axis=1), test, valLen

trainX, trainY, valX, valY, data, test, testIndex = getData()

gaParams = {
    "evaluator": partial(NAS_refactored.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY, numEvals=1),
    "generator": partial(NAS_refactored.generateRandomArchitecture, sampleX=trainX[:2000], sampleY=trainY[:2000], validThreshold=1, numVal=200),
    "populationSize": 100,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 25,
    "minimizeFitness": True,
    "logModels": False,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0,
    "n_jobs": 25,
    "saveModels": False,
    "dataset": "water"
}

if __name__ == "__main__":
    for i in range(3, 4):
        error = False
        gaParams["experimentIndex"] = i
        while True:
            try:
                gaResults = NAS_refactored.runGA(gaParams, error)
                model = gaResults["bestModel"]
                preds = []

                for idx in range(testIndex+18, len(data)):
                    modelCopy = copy.deepcopy(model)
                    pred = []
                    out = NAS_refactored.runModel(modelCopy, data[idx-500:idx])[-1]
                    pred.append(out[0])
                    for i in range(17):
                        out = NAS_refactored.runModel(modelCopy, pred[-1])
                        pred.append(out[0])
                    preds.append(pred)
                preds = np.array(preds)
                print("MSE:", mse(test, preds))
                break
            except:
                print(traceback.format_exc())
                error = True
