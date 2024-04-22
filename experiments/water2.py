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
    
    train_in = water[0:trainLen, :18]
    train_out = np.expand_dims(water[0:trainLen, 18], axis=1)
    val_in = water[trainLen:valLen, :18]
    val_out = np.expand_dims( water[trainLen:valLen, 18], axis=1)
    test_in = water[valLen:, :18]
    test_out = water[valLen:, 18:]
    return train_in, train_out, val_in, val_out, test_in, test_out

trainX, trainY, valX, valY, test_in, test_out = getData()

gaParams = {
    "evaluator": partial(NAS_refactored.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY, numEvals=1, memoryLimit=4*1024),
    "generator": partial(NAS_refactored.generateRandomArchitecture, sampleX=trainX[:2000], sampleY=trainY[:2000], validThreshold=10, maxInput=len(trainX), memoryLimit=4*1024, numVal=200),
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
    "outputDim": trainY.shape[-1],
    "memoryLimitPerJob": 4 * 1024,
    "saveModels": False,
    "dataset": "water"
}

if __name__ == "__main__":
    for i in range(5, 6):
        error = False
        gaParams["experimentIndex"] = i
        while True:
            try:
                gaResults = NAS_refactored.runGA(gaParams, error)
                model = gaResults["bestModel"]
                preds = []

                for input in test_in:
                    modelCopy = copy.deepcopy(model)
                    pred = []
                    for i in range(len(input)):
                        out = NAS_refactored.runModel(modelCopy, input)
                        pred.append(out[0])
                        input = np.append(input, out[0])
                        input = input[1:]
                    preds.append(pred)
                preds = np.array(preds)
                preds = np.squeeze(preds)
                print("MSE:", mse(test_out, preds))
                break
            except:
                print(traceback.format_exc())
                error = True
