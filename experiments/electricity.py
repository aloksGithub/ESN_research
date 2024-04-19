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
    # df = df.iloc[:100000]

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

gaParams = {
    "evaluator": partial(NAS_refactored.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY, numEvals=1),
    "generator": partial(NAS_refactored.generateRandomArchitecture, sampleX=trainX[:2000], sampleY=trainY[:2000], validThreshold=10, numVal=200),
    "populationSize": 50,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 20,
    "minimizeFitness": True,
    "logModels": False,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0,
    "n_jobs": 2,
    "saveModels": False,
    "dataset": "electricity"
}

if __name__ == "__main__":
    for i in range(1):
        error = False
        gaParams["experimentIndex"] = i
        while True:
            try:
                gaResults = NAS_refactored.runGA(gaParams, error)
                model = gaResults["bestModel"]
                preds = NAS_refactored.runModel(model, testX)
                print("MSE:", mse(testY, preds))
                break
            except:
                print(traceback.format_exc())
                error = True