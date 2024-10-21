import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.parallel_processing import executeParallelImproved
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

if __name__ == "__main__":
    gas = [readSavedExperiment(f'./backup_50/sunspots/backup_{i}.obj') for i in range(5)]
    architectures = [findBestGaArchitecture(ga)[0] for ga in gas]


    trainX, trainY, valX, valY, testX, testY = getDataSunspots()
    for ga in gas:
        architecture, _ = findBestGaArchitecture(ga)
        model = ga.bestModel
        def func():
            trainModel(model, trainX, trainY)
            runModel(model, valX)
        approx = estimateMemory(architecture, len(trainX))
        startTime = time.time()
        try:
            accurate = measure_memory_usage(func)
            print(approx, accurate, time.time() - startTime)
        except:
            pass