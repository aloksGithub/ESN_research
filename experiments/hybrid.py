import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.error_metrics import nrmse, r_squared
warnings.filterwarnings("ignore")
from utils import getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots, getDataWater, readSavedExperiment, printSavedResults
from NAS.utils import runModel
from NAS.ESN_BO import ESN_BO
from NAS.ESN_NAS import ESN_NAS
from reservoirpy.observables import (mse)
import math

rpy.verbosity(0)

def nrmse_sunspots(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
    if math.isnan(error):
        return np.inf
    else:
        return error

def printSavedAutoregressiveResults(dataset):
    print(f"================================{dataset}================================")
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_hybrid/{}/backup_{}.obj'.format(dataset, i))
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
    _, _, _, _, testX, testY = getDataSunspots()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_hybrid/sunspots/backup_{}.obj'.format(i))
        model = bo.bestModel
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
    _, _, _, _, testX, testY, _ = getDataWater()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_hybrid/water/backup_{}.obj'.format(i))
        model = bo.bestModel
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

def findBestGaArchitecture(ga: ESN_NAS):
    errors = [errors[0] for errors in ga.fitnesses]
    gaBestError = min(errors)
    gaBestErrorIndex = errors.index(gaBestError)
    gaBestModel = ga.architectures[gaBestErrorIndex]
    return gaBestModel, gaBestError

# def findBestArchitecture(gas: list[ESN_NAS]):
#     bestError = np.inf
#     bestArchitecture = None
#     for ga in gas:
#         gaBestModel, gaBestError = findBestGaArchitecture(ga)
#         if gaBestError<bestError:
#             bestError = gaBestError
#             bestArchitecture = gaBestModel
#     return bestArchitecture

def runAutoRegressiveExperiment(dataset, dataLoader):
    trainX, trainY, valX, valY, testX, testY = dataLoader()
    ga = readSavedExperiment('backup_50/{}/backup_{}.obj'.format(dataset, sys.argv[1]))
    baseArchitecture, _ = findBestGaArchitecture(ga)
    ga_bo = ESN_BO(
        trainX,
        trainY,
        valX,
        valY,
        500,
        2000,
        trainY.shape[-1],
        baseArchitecture,
        3,
        3,
        [nrmse, r_squared],
        [np.inf, 0],
        True,
        180,
        "backup_hybrid/{}/backup_{}.obj".format(dataset, sys.argv[1]),
        True
    )
    ga_bo.run()


if __name__ == "__main__":
    # printAllResults()
    runAutoRegressiveExperiment("mgs", getDataMGS)
    runAutoRegressiveExperiment("laser", getDataLaser)
    runAutoRegressiveExperiment("dde", getDataDDE)
    runAutoRegressiveExperiment("lorenz", getDataLorenz)
    
    # Non auto regressive water experiment
    trainX, trainY, valX, valY, testX, testY, _ = getDataWater()
    ga = readSavedExperiment('backup_50/{}/backup_{}.obj'.format('water', sys.argv[1]))
    baseArchitecture, _ = findBestGaArchitecture(ga)
    ga_bo = ESN_BO(
        trainX,
        trainY,
        valX,
        valY,
        500,
        2000,
        trainY.shape[-1],
        baseArchitecture,
        3,
        3,
        [nrmse, r_squared],
        [np.inf, 0],
        True,
        180,
        "backup_hybrid/{}/backup_{}.obj".format('water', sys.argv[1]),
        False
    )
    ga_bo.run()

    # Non auto regressive sunspots experiment
    trainX, trainY, valX, valY, testX, testY = getDataSunspots()
    ga = readSavedExperiment('backup_50/{}/backup_{}.obj'.format('sunspots', sys.argv[1]))
    baseArchitecture, _ = findBestGaArchitecture(ga)
    ga_bo = ESN_BO(
        trainX,
        trainY,
        valX,
        valY,
        500,
        2000,
        trainY.shape[-1],
        baseArchitecture,
        3,
        3,
        [nrmse_sunspots, r_squared],
        [np.inf, 0],
        True,
        180,
        "backup_hybrid/{}/backup_{}.obj".format('sunspots', sys.argv[1]),
        False
    )
    ga_bo.run()
