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

trainX, trainY, valX, valY, testX, testY = getData()

if __name__ == "__main__":
    for i in range(1):
        # file = open('backup/electricity/backup_{}.obj'.format(i), 'rb')
        # data = pickle.load(file)
        # model = data["bestModel"]
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            50,
            10,
            trainY.shape[-1],
            n_jobs=5,
            errorMetrics=[mse, smape],
            defaultErrors=[np.inf, np.inf],
            timeout=180,
            numEvals=1,
            saveLocation='backup/electricity/backup_{}.obj'.format(i),
            memoryLimit=1024
        )
        gaResults = ga.run()
        model = gaResults["bestModel"]
        preds = runModel(model, testX)
        print("MSE (1-7):", mse(testY[:, :7], preds[:, :7]))
        print("SMAPE (1-7):", smape(testY[:, :7], preds[:, :7]))
        print("MSE (8-12):", mse(testY[:, 7:13], preds[:, 7:13]))
        print("SMAPE (8-12):", smape(testY[:, 7:13], preds[:, 7:13]))
        print("MSE (13-18):", mse(testY[:, 13:18], preds[:, 13:18]))
        print("SMAPE (13-18):", smape(testY[:, 13:18], preds[:, 13:18]))
        print("MSE (1-18):", mse(testY, preds))
        print("SMAPE (1-18):", smape(testY, preds))