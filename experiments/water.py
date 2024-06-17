import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import math
from reservoirpy.observables import (mse)

rpy.verbosity(0)

def getDataSingleCol():
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    firstCol = water[:, 0]
    lastRow = water[-1, 1:]
    allData = np.expand_dims(np.concatenate((firstCol, lastRow)), axis=1)
    
    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)
    
    train_in = allData[0:trainLen]
    train_out = allData[0:trainLen]
    val_in = allData[trainLen:valLen]
    val_out = allData[trainLen:valLen]
    test_in = allData[valLen:]
    test_out = allData[valLen:]
    return train_in, train_out, val_in, val_out, test_in, test_out, water

def getData():
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)
    
    train_in = water[0:trainLen, :18]
    train_out = water[0:trainLen, 18:]
    val_in = water[trainLen:valLen, :18]
    val_out = water[trainLen:valLen, 18:]
    test_in = water[valLen:, :18]
    test_out = water[valLen:, 18:]
    return train_in, train_out, val_in, val_out, test_in, test_out

trainX, trainY, valX, valY, testX, testY, allData = getDataSingleCol()

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    
    error = rmse / mean_norm
    if math.isnan(error):
        return np.inf
    else:
        return error
    
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (numerator / denominator)
    if math.isnan(r2):
        return 0
    else:
        return 1 - (numerator / denominator)

if __name__ == "__main__":
    for i in range(1):
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            50,
            50,
            trainY.shape[-1],
            n_jobs=20,
            errorMetrics=[nrmse, r_squared],
            defaultErrors=[np.inf, 0],
            timeout=60,
            numEvals=1,
            saveLocation='backup/water/backup_0.obj',
            memoryLimit=1024
        )
        ga.run()
        model = ga.bestModel
        preds = runModel(model, testX)
        print("NRMSE:", mse(nrmse, preds))
        print("R2:", r_squared(testY, preds))


    # model1 = copy.deepcopy(model)
    # trainLen = trainX.shape[0] + valX.shape[0]
    # preds = []
    # expectedTestX = []
    # for row, i in enumerate(testX[:-36]):
    #     currModel = copy.deepcopy(model1)
    #     currPred = [runModel(currModel, testX[i:i+18])[-1]]
    #     for _ in range(17):
    #         currPred.append(runModel(currModel, currPred[-1])[0])
    #     preds.append(currPred)
    #     expectedTestX.append(testX[i:i+18])
    #     runModel(model1, testX[i])
    # expectedTestX = np.array(expectedTestX)
    # preds = np.array(preds)
    # print(expectedTestX==allData[trainLen:, :18])
