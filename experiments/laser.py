import math
import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS import NAS
warnings.filterwarnings("ignore")
import sys
import pickle
import pandas as pd

rpy.verbosity(0)

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    error = rmse/mean_norm
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

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Santafe laser dataset
def getData():
    sunspots = pd.read_csv("./data/santafelaser.csv")
    data = np.array(sunspots)
    data = data.reshape((data.shape[0],1))
    data = data[:3801,:]
    from scipy import stats
    data = stats.zscore(data)

    trainLen = 2000
    valLen = 100
    testLen = 100
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out

def printSavedResults():
    nrmseErrors = []
    r2_squaredValues = []
    for i in range(5):
        ga = readSavedExperiment('backup/laser/backup_{}.obj'.format(i))
        nrmseErrors.append(ga.bestFitness[0])
        r2_squaredValues.append(ga.bestFitness[1])
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

def printOldSavedResults():
    nrmseErrors = []
    r2_squaredValues = []
    total = 0
    r2_total = 0
    for i in range(5):
        data = readSavedExperiment('old_backup/laser/backup_{}.obj'.format(i))
        total+=min(data["allFitnesses"])
        nrmseErrors.append(min(data["allFitnesses"]))
        r2_squaredValues.append(data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))])
        r2_total+=data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))]
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

if __name__ == "__main__":
    trainX, trainY, valX, valY, testX, testY = getData()
    nrmseErrors = []
    r2_squaredValues = []
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
            saveLocation='backup/laser/backup_{}.obj'.format(i),
            memoryLimit=756,
            isAutoRegressive=True
        )
        ga.run()
        nrmseErrors.append(ga.bestFitness[0])
        r2_squaredValues.append(ga.bestFitness[1])
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))
