import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS import NAS_refactored
from NAS.ESN_NAS import ESN_NAS
warnings.filterwarnings("ignore")
import traceback
import sys
import pandas as pd
from reservoirpy.observables import mse

rpy.verbosity(0)
output_dim = 1

def getData():
    df = pd.read_csv("data/electricity.csv", index_col="Unnamed: 0")
    df.index = pd.to_datetime(df['Date'] + " " + df['Time'])
    df = df.sort_index()
    # df = df.iloc[:60000]

    df['Target_active_power'] = df['Global_active_power'].shift(-1)
    df = df.replace("?", np.nan).ffill().dropna()

    T = len(df)
    valid_boundary = 0.6
    test_boundary = 0.8
    valid_index = int(T*valid_boundary)
    test_index = int(T*test_boundary)
    train = df.iloc[:valid_index, 2:].to_numpy().astype(float)
    val = df.iloc[valid_index:test_index, 2:].to_numpy().astype(float)
    test = df.iloc[test_index:, 2:].to_numpy().astype(float)
    trainX = train[:, :-1]
    trainY = train[:, -1:]
    valX = val[:, :-1]
    valY = val[:, -1:]
    testX = test[:, :-1]
    testY = test[:, -1:]
    return trainX, trainY, valX, valY, testX, testY

trainX, trainY, valX, valY, testX, testY = getData()

def nmse(true, pred):
    return mse(true, pred) / 0.0439292598

if __name__ == "__main__":
    for i in range(1):
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            20,
            50,
            trainY.shape[-1],
            n_jobs=3,
            timeout=480,
            errorMetrics=[nmse],
            saveLocation='backup/electricity/backup_{}.obj'.format(i)
        )
        gaResults = ga.run()
        model = gaResults["bestModel"]
        preds = NAS_refactored.runModel(model, testX)
        print("MSE:", mse(testY, preds))
        print("Norm MSE:", nmse(testY, preds))