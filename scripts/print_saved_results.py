import reservoirpy as rpy
rpy.verbosity(0)
import numpy as np
import sys
import os
import pickle

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.utils import runModel
from src.algorithms.ESN_BO import ESN_BO
from src.algorithms.ESN_GA import ESN_GA
from src.datasets import getDataDDE, getDataLaser, getDataLorenz, getDataMGS

# Load fresh test data from the updated dataset functions
DATASET_LOADERS = {
    'mgs': getDataMGS,
    'lorenz': getDataLorenz,
    'dde': getDataDDE,
    'laser': getDataLaser,
}

def readSavedExperiment(path):
    file = open(path, "rb")
    return pickle.load(file)

def printSavedResults(directory, dataset, isAutoregressive=True, isGA=True):
    # Load fresh test data for autoregressive datasets
    testX = None
    testY = None
    if isAutoregressive and dataset in DATASET_LOADERS:
        _, _, _, _, testX, testY = DATASET_LOADERS[dataset]()

    nrmseErrors = []
    r2_squaredValues = []
    times = []
    for i in range(5):
        ga: ESN_GA | ESN_BO = readSavedExperiment("{}/{}/backup_{}.obj".format(directory, dataset, i))
        try:
            if isGA:
                times.append(sum(ga.generationTimes))
            else:
                totalTime = sum(ga.times)
                times.append(totalTime)
        except:
            times.append(0)
        best_model = ga.bestModel
        if isAutoregressive:
            runModel(best_model, ga.experimentData.trainX)
            runModel(best_model, ga.experimentData.valX)
            prevOutput = testX[0]
            preds = []
            for _ in range(len(testX)):
                pred = runModel(best_model, prevOutput)
                prevOutput = pred
                preds.append(pred[0])
            preds = np.array(preds)
            nrmse_error = ga.evalParams.errorMetrics[0](testY, preds)
            r2_error = ga.evalParams.errorMetrics[1](testY, preds)
        else:
            runModel(best_model, ga.experimentData.valX)
            preds = runModel(best_model, ga.experimentData.testX)
            nrmse_error = ga.evalParams.errorMetrics[0](ga.experimentData.testY, preds)
            r2_error = ga.evalParams.errorMetrics[1](ga.experimentData.testY, preds)

        nrmseErrors.append(nrmse_error)
        r2_squaredValues.append(r2_error)
    print("==============================================================")
    print("{} Errors:".format(dataset))
    print("NRMSE:", nrmseErrors)
    print("R2:", r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))
    print("Times:")
    print(times)
    print("Average time: {} ({})".format(np.average(times), np.std(times)))

if __name__ == "__main__":
    print("==============================ESNAS==============================")
    printSavedResults('results/esnas', 'mgs')
    printSavedResults('results/esnas', 'lorenz')
    printSavedResults('results/esnas', 'dde')
    printSavedResults('results/esnas', 'laser')
    printSavedResults('results/esnas', 'sunspots', isAutoregressive=False)
    printSavedResults('results/esnas', 'water', isAutoregressive=False)
    print("==============================GA==============================")
    printSavedResults('results/ga', 'mgs')
    printSavedResults('results/ga', 'lorenz')
    printSavedResults('results/ga', 'dde')
    printSavedResults('results/ga', 'laser')
    printSavedResults('results/ga', 'sunspots', isAutoregressive=False)
    printSavedResults('results/ga', 'water', isAutoregressive=False)
    print("==============================BO==============================")
    printSavedResults('results/bo', 'mgs', isGA=False)
    printSavedResults('results/bo', 'lorenz', isGA=False)
    printSavedResults('results/bo', 'dde', isGA=False)
    printSavedResults('results/bo', 'laser', isGA=False)
    printSavedResults('results/bo', 'sunspots', isAutoregressive=False, isGA=False)
    printSavedResults('results/bo', 'water', isAutoregressive=False, isGA=False)

