import sys
import os
import warnings
from scipy.special import comb
import tracemalloc
import copy

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.algorithms.ESN_BO import ESN_BO
from src.algorithms.ESN_GA import ESN_GA
from src.utils import trainModel, runModel, constructModel
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

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure the peak memory usage of a function.

    Parameters:
        func (callable): The function to measure.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        float: The peak memory usage in MB during the function's execution.
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_MB = peak / (1024 * 1024)
    return peak_memory_MB, result

def measure_time(
    individual,
    trainX,
    trainY,
    valX,
    valY,
    isAutoRegressive,
):
    """Standalone function executed in a worker process.

    Returns ``(individual, bestErrors, bestModel_or_None)``.  The model
    object is returned *only if it can be pickled*; otherwise ``None`` is
    sent back.
    """

    startTime = time.time()

    model = constructModel(individual)

    model = trainModel(model, trainX, trainY)
    trainTime = time.time() - startTime
    startTime = time.time()

    model_copy = copy.deepcopy(model)

    if isAutoRegressive:
        prevOutput = valX[0]
        preds = []
        for _ in range(len(valX)):
            pred = runModel(model, prevOutput)
            prevOutput = pred
            preds.append(pred[0])
        preds = np.array(preds)
    else:
        preds = runModel(model, valX)
    
    runTime = time.time() - startTime
    
    return trainTime, runTime


if __name__ == "__main__":
    dataNames = ['dde', 'laser', 'lorenz', 'mgs', 'sunspots', 'water']
    for dataName in dataNames:
        gas: list[ESN_GA] = [readSavedExperiment(f'./results/esnas/{dataName}/backup_{j}.obj') for j in range(5)]
        memories = []
        times = []

        train_times = []
        run_times = []
        for ga in gas:
            model = ga.bestModel
            best_architecture, _ = findBestGaArchitecture(ga)
            def func():
                return measure_time(
                    best_architecture,
                    ga.experimentData.trainX,
                    ga.experimentData.trainY,
                    ga.experimentData.valX,
                    ga.experimentData.valY,
                    ga.evalParams.isAutoRegressive,
                )
            accurate, result = measure_memory_usage(func)
            trainTime, runTime = result
            memories.append(accurate)
            train_times.append(trainTime)
            run_times.append(runTime)
            times.append(trainTime + runTime)
        memories = np.array(memories)
        train_times = np.array(train_times)
        run_times = np.array(run_times)
        times = np.array(times)
        print("==========================", dataName, "==========================")
        print("Average memory: ", np.mean(memories), np.std(memories))
        print("Average train time: ", np.mean(train_times), np.std(train_times))
        print("Average run time: ", np.mean(run_times), np.std(run_times))
        print("Average total time: ", np.mean(times), np.std(times))