import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel, smape
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import math
from reservoirpy.observables import (mse)

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
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            15,
            50,
            trainY.shape[-1],
            n_jobs=25,
            errorMetrics=[mse, smape],
            defaultErrors=[np.inf, np.inf],
            timeout=4*480,
            numEvals=2,
            saveLocation='backup/electricity/backup_{}.obj'.format(i)
        )
        gaResults = ga.run()
        model = gaResults["bestModel"]
        preds = runModel(model, testX)
        print("MSE:", mse(testY, preds))
        print("SMAPE:", smape(testY, preds))