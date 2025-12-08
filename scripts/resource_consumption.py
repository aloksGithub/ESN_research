import sys
import os
import warnings

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.algorithms.ESN_BO import ESN_BO
from src.algorithms.ESN_GA import ESN_GA
from src.utils import evaluateArchitecture, trainModel, runModel
from src.error_metrics import nrmse, r_squared
from src.memory_estimator import measure_memory_usage
import time
import numpy as np
import pickle
import reservoirpy
reservoirpy.verbosity(0)
warnings.filterwarnings("ignore")

def readSavedExperiment(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def findBestGaArchitecture(ga: ESN_GA):
    errors = [errors[0] for errors in ga.fitnesses]
    gaBestError = min(errors)
    gaBestErrorIndex = errors.index(gaBestError)
    gaBestModel = ga.architectures[gaBestErrorIndex]
    return gaBestModel, gaBestError

def findBestGasArchitecture(gas: list[ESN_GA]):
    bestError = np.inf
    bestArchitecture = None
    for ga in gas:
        architecture, error = findBestGaArchitecture(ga)
        if error<bestError:
            bestArchitecture = architecture
    return bestArchitecture

def findBestGa(gas: list[ESN_GA]):
    bestError = np.inf
    bestGa = None
    for ga in gas:
        _, error = findBestGaArchitecture(ga)
        if error<bestError:
            bestGa = ga
    return bestGa

# Stub function to avoid import errors
def nrmse_sunspots(y_true, y_pred):
    return 0

if __name__ == "__main__":
    dataNames = ['dde', 'laser', 'lorenz', 'mgs', 'sunspots', 'water']
    for dataName in dataNames:
        gas: list[ESN_GA] = [readSavedExperiment(f'./results/esnas/{dataName}/backup_{j}.obj') for j in range(5)]
        memories = []
        times = []
        for ga in gas:
            model = ga.bestModel
            best_architecture, _ = findBestGaArchitecture(ga)
            def func():
                evaluateArchitecture(
                    best_architecture,
                    ga.experimentData.trainX,
                    ga.experimentData.trainY,
                    ga.experimentData.valX,
                    ga.experimentData.valY,
                    1,
                    ga.evalParams.errorMetrics,
                    ga.evalParams.defaultErrors,
                    ga.evalParams.isAutoRegressive
                )
            startTime = time.time()
            accurate = measure_memory_usage(func)
            timeTaken = time.time() - startTime
            memories.append(accurate)
            times.append(timeTaken)
        memories = np.array(memories)
        times = np.array(times)
        print("==========================", dataName, "==========================")
        print("Average memory: ", np.mean(memories), np.std(memories))
        print("Average time: ", np.mean(times), np.std(times))