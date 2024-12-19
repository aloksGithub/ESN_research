import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from utils import getDataSunspots
from NAS.error_metrics import r_squared
from NAS.utils import runModel
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import math
from reservoirpy.observables import (mse)
import pickle
rpy.verbosity(0)

def nrmse(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
    if math.isnan(error):
        return np.inf
    else:
        return error
    
def readSavedExperiment(path):
    file = open(path, 'rb')
    return pickle.load(file)

def printSavedResults(version):
    _, _, valX, _, testX, testY = getDataSunspots()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        ga = readSavedExperiment('backup_{}/sunspots/backup_{}.obj'.format(version, i))
        model = ga.bestModel
        runModel(model, valX)
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
    trainX, trainY, valX, valY, testX, testY = getDataSunspots()
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