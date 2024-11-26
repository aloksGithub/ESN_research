import sys
import os
import warnings

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import trainModel, runModel
from NAS.error_metrics import nrmse, r_squared
from NAS.memory_estimator import measure_memory_usage, estimateMemory
from utils import readSavedExperiment, getDataDDE, getDataLaser, getDataLorenz, getDataMGS, getDataSunspots, getDataWater
import time
import numpy as np

def findBestGaArchitecture(ga: ESN_NAS):
    errors = [errors[0] for errors in ga.fitnesses]
    gaBestError = min(errors)
    gaBestErrorIndex = errors.index(gaBestError)
    gaBestModel = ga.architectures[gaBestErrorIndex]
    return gaBestModel, gaBestError

def findBestGasArchitecture(gas: list[ESN_NAS]):
    bestError = np.inf
    bestArchitecture = None
    for ga in gas:
        architecture, error = findBestGaArchitecture(ga)
        if error<bestError:
            bestArchitecture = architecture
    return bestArchitecture

def findBestGa(gas: list[ESN_NAS]):
    bestError = np.inf
    bestGa = None
    for ga in gas:
        _, error = findBestGaArchitecture(ga)
        if error<bestError:
            bestGa = ga
    return bestGa


def nrmse_sunspots(y_true, y_pred):
    return 0

if __name__ == "__main__":
    dataNames = ['water']
    datasets = [getDataWater]
    for i, getDataset in enumerate(datasets):
        trainX, trainY, valX, valY, testX, testY = getDataset()
        print(len(trainX))
        bos = [readSavedExperiment('./backup_bo/{}/backup_{}.obj'.format(dataNames[i], j)) for j in [0, 1, 2, 4]]
        maxMemory = 0
        maxTime = 0
        for bo in bos:
            model = bo.bestModel
            def func():
                trainModel(model, trainX, trainY)
                runModel(model, valX)
            startTime = time.time()
            try:
                accurate = measure_memory_usage(func)
                timeTaken = time.time() - startTime
                if accurate>maxMemory:
                    maxMemory = accurate
                if timeTaken>maxTime:
                    maxTime = timeTaken
            except:
                pass
        print(f'BO_{dataNames[i]}: {maxMemory} MB       {maxTime}s')

        gas = [readSavedExperiment(f'./backup_50/{dataNames[i]}/backup_{j}.obj') for j in [0, 1, 2, 4]]
        maxMemory = 0
        maxTime = 0
        for ga in gas:
            model = ga.bestModel
            def func():
                trainModel(model, trainX, trainY)
                runModel(model, valX)
            startTime = time.time()
            try:
                accurate = measure_memory_usage(func)
                timeTaken = time.time() - startTime
                if accurate>maxMemory:
                    maxMemory = accurate
                if timeTaken>maxTime:
                    maxTime = timeTaken
            except:
                pass
        print(f'ESNAS_{dataNames[i]}: {maxMemory} MB       {maxTime}s')
        
        hybrid = [readSavedExperiment('./backup_hybrid/{}/backup_{}.obj'.format(dataNames[i], j)) for j in [0, 1, 2, 4]]
        maxMemory = 0
        maxTime = 0
        for ga in hybrid:
            model = ga.bestModel
            def func():
                trainModel(model, trainX, trainY)
                runModel(model, valX)
            startTime = time.time()
            try:
                accurate = measure_memory_usage(func)
                timeTaken = time.time() - startTime
                if accurate>maxMemory:
                    maxMemory = accurate
                if timeTaken>maxTime:
                    maxTime = timeTaken
            except:
                pass
        print(f'ESNAS+BO_{dataNames[i]}: {maxMemory} MB       {maxTime}s')