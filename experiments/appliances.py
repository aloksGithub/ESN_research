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
from reservoirpy.observables import (mse, rmse)
import pickle

rpy.verbosity(0)

def getData():
    from ucimlrepo import fetch_ucirepo
    appliances_energy_prediction = fetch_ucirepo(id=374)
    X = appliances_energy_prediction.data.features.iloc[:, 1:].to_numpy()
    y = appliances_energy_prediction.data.targets.to_numpy()

    trainLen = int(len(X)*0.65)
    valLen = int(len(X)*0.75)
    
    train_in = X[0:trainLen]
    train_out = y[0:trainLen]
    val_in = X[trainLen:valLen]
    val_out = y[trainLen:valLen]
    test_in = X[valLen:]
    test_out = y[valLen:]
    return train_in, train_out, val_in, val_out, test_in, test_out

trainX, trainY, valX, valY, testX, testY = getData()

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
            n_jobs=10,
            errorMetrics=[rmse],
            defaultErrors=[np.inf, np.inf],
            timeout=240,
            numEvals=3,
            saveLocation='backup/appliances/backup_{}.obj'.format(i),
            memoryLimit=512
        )
        gaResults = ga.run()
        model = gaResults["bestModel"]
        preds = runModel(model, testX)
        print("RMSE:", rmse(testY, preds))