import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.error_metrics import nrmse, nrmse_sunspots, r_squared
warnings.filterwarnings("ignore")
from utils import getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots, getDataWater, getDataWaterMultiStep, readSavedExperiment, printSavedResults
from NAS.utils import runModel
from NAS.ESN_BO import ESN_BO
import math

rpy.verbosity(0)
water_steps = 1
baseArchitecture = {'nodes': [{'type': 'Input', 'params': {'input_dim': 1}}, {'type': 'Reservoir', 'params': {'units': 1000, 'lr': 0.9, 'sr': 0.9, 'input_connectivity': 0.25, 'rc_connectivity': 0.25}}, {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 8.0e-05}}], 'edges': [[0, 1], [1, 2]]}

def printSavedAutoregressiveResults(dataset):
    print(f"================================{dataset}================================")
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_bo2/{}/backup_{}.obj'.format(dataset, i))
        mainErrors = [e[0] for e in bo.performances]
        bestErrors = bo.performances[mainErrors.index(min(mainErrors) if bo.minimizeFitness else max(mainErrors))]
        nrmseError = bestErrors[0]
        r2Error = bestErrors[1]
        nrmseErrors.append(nrmseError)
        rSquaredValues.append(r2Error)
        print("Result:", nrmseError, r2Error)
    print("Errors:")
    print(nrmseErrors)
    print(rSquaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(rSquaredValues), np.std(rSquaredValues)))

def printSavedSunspotsResults():
    print("================================Sunspots================================")
    _, _, valX, _, testX, testY = getDataSunspots()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_bo2/sunspots/backup_{}.obj'.format(i))
        model = bo.bestModel
        runModel(model, valX)
        preds = runModel(model, testX)
        nrmseError = nrmse_sunspots(testY, preds)
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

def printSavedWaterResults():
    print("================================water================================")
    _, _, valX, _, testX, testY = getDataWater()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_bo2/{}/backup_{}.obj'.format("water", i))
        model = bo.bestModel
        runModel(model, valX)
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

def printAllResults():
    printSavedAutoregressiveResults("mgs")
    printSavedAutoregressiveResults("laser")
    printSavedAutoregressiveResults("dde")
    printSavedAutoregressiveResults("lorenz")
    printSavedWaterResults()
    printSavedSunspotsResults()

def runBOExperiment(dataset, dataLoader, errorMetrics, isAutoregressive):
    trainX, trainY, valX, valY, testX, testY = dataLoader()
    bo = ESN_BO(
        trainX,
        trainY,
        valX,
        valY,
        2000,
        2000,
        trainY.shape[-1],
        baseArchitecture,
        3,
        3,
        errorMetrics=errorMetrics,
        defaultErrors=[100000, 0],
        minimizeFitness=True,
        timeout=60,
        saveLocation="backup_bo2/{}/backup_{}.obj".format(dataset, sys.argv[1]),
        isAutoRegressive=isAutoregressive
    )
    bo.run()


if __name__ == "__main__":
    # printAllResults()
    runBOExperiment("laser", getDataLaser, [nrmse, r_squared], True)
    runBOExperiment("dde", getDataDDE, [nrmse, r_squared], True)
    runBOExperiment("lorenz", getDataLorenz, [nrmse, r_squared], True)
    runBOExperiment("mgs", getDataMGS, [nrmse, r_squared], True)
    runBOExperiment("sunspots", getDataSunspots, [nrmse_sunspots, r_squared], False)
    runBOExperiment("water", getDataWater, [nrmse, r_squared], False)
