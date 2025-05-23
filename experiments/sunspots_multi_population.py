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
    return np.sqrt(mseError/variance)

if __name__ == "__main__":
    for i in range(1):
        GAs = [ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            20,
            30,
            trainY.shape[-1],
            n_jobs=10,
            errorMetrics=[nrmse],
            defaultErrors=[np.inf],
            timeout=120,
            numEvals=3,
            saveLocation='backup/sunspots/backup_{}.obj'.format(i),
            memoryLimit=512
        ) for _ in range(5)]
        gaResults = [ga.run() for ga in GAs]
        architectures = []
        for gaResult in gaResults:
            objective = [errors[0] for errors in gaResult["fitnesses"]]
            bestIndex = objective.index(min(objective))
            architectures.append(gaResult["architectures"][bestIndex])

        finalGA = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            20,
            30,
            trainY.shape[-1],
            n_jobs=10,
            seedModels=architectures,
            errorMetrics=[nrmse],
            defaultErrors=[np.inf],
            timeout=120,
            numEvals=3,
            saveLocation='backup/sunspots/backup_{}.obj'.format(i),
            memoryLimit=512
        )

        gaResults = finalGA.run()
        model = gaResults["bestModel"]
        preds = runModel(model, testX)
        print("NRMSE:", nrmse(testY, preds))