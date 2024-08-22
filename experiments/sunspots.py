import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel, smape, trainModel
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import math
from reservoirpy.observables import (mse)
import pickle

rpy.verbosity(0)

def getData():
    sunspots = pd.read_csv("./datasets/Sunspots.csv")
    data = sunspots.loc[:,"Monthly Mean Total Sunspot Number"].to_numpy()
    data = np.expand_dims(data, axis=1)

    trainLen = 1600
    valLen = 500
    testLen = 1074
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out
trainX, trainY, valX, valY, testX, testY = getData()

def nrmse(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
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
        return r2
    
def readSavedExperiment(path):
    file = open(path, 'rb')
    return pickle.load(file)

def printSavedResults(version):
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        ga = readSavedExperiment('backup_{}/sunspots/backup_{}.obj'.format(version, i))
        model = ga.bestModel
        preds = runModel(model, testX)
        nrmseError = nrmse(testY, preds)
        r2Error = r_squared(testY, preds)
        nrmseErrors.append(nrmseError)
        rSquaredValues.append(r2Error)
    print("Errors:")
    print(nrmseErrors)
    print(rSquaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(rSquaredValues), np.std(rSquaredValues)))

if __name__ == "__main__":
    nrmseErrors = []
    rSquaredValues = []
    for i in [0, 1, 2, 3, 4]:
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            50,
            50,
            trainY.shape[-1],
            n_jobs=10,
            errorMetrics=[nrmse, r_squared],
            defaultErrors=[np.inf, 0],
            timeout=180,
            numEvals=3,
            saveLocation='backup_50/sunspots/backup_{}.obj'.format(i),
            memoryLimit=756,
            isAutoRegressive=False
        )
        gaResults = ga.run()
        model = ga.bestModel
        preds = runModel(model, testX)
        nrmseError = nrmse(testY, preds)
        r2Error = r_squared(testY, preds)
        nrmseErrors.append(nrmseError)
        rSquaredValues.append(r2Error)
        print("Result:", nrmseError, r2Error)
    print("Errors:")
    print(nrmseErrors)
    print(rSquaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(rSquaredValues), np.std(rSquaredValues)))