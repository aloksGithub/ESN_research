import math
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.utils import evaluateArchitecture, estimateMemory, isValidArchitecture
import time
import matplotlib.pyplot as plt

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

def checkTrainTime(architecture, numInputs):
    start = time.time()
    evaluateArchitecture(architecture, trainX[:numInputs], trainY[:numInputs], valX[:numInputs], valY[:numInputs])
    timeTaken = time.time() - start
    return timeTaken


def wait():
    time.sleep(20)

if __name__ == "__main__":
    try:
        parallel = Parallel(n_jobs=5, timeout=3, require='sharedmem')
        parallel(delayed(wait)() for i in range(5))
    except multiprocessing.context.TimeoutError:
        print("Task failed successfully")
        pass
    # runtimes = []
    # architecture = {'nodes': [{'type': 'Input', 'params': {'input_dim': 18}}, {'type': 'IPReservoir', 'params': {'units': 1516, 'lr': 0.3794160763282813, 'sr': 0.9412495273142261, 'mu': 0.09315428782964497, 'sigma': 1.9996893018324113, 'learning_rate': 0.007115438623711385, 'input_connectivity': 0.19270834464023434, 'rc_connectivity': 0.4139009070019663, 'fb_connectivity': 0.4407973219740993}}, {'type': 'LMS', 'params': {'output_dim': 18, 'alpha': 0.3507005013535671}}, {'type': 'NVAR', 'params': {'delay': 5, 'order': 2, 'strides': 3}}, {'type': 'Ridge', 'params': {'output_dim': 18, 'ridge': 6.563310358305086e-06}}, {'type': 'IPReservoir', 'params': {'units': 68, 'lr': 0.8788496628894418, 'sr': 0.660379110102254, 'mu': 0.16625820180299722, 'sigma': 0.5806266716741275, 'learning_rate': 0.008367985016205044, 'input_connectivity': 0.16643347500247355, 'rc_connectivity': 0.136383107016403, 'fb_connectivity': 0.11601073175108537}}, {'type': 'Ridge', 'params': {'output_dim': 18, 'ridge': 2.2092030120388196e-05}}], 'edges': [[0, 1], [1, 2], [2, 3], [1, 4], [2, 5], [3, 6], [4, 6], [5, 6]]}


    # print(isValidArchitecture(architecture, 4000, 1024, 480))
    # start = time.time()
    # perf = evaluateArchitecture(architecture, trainX, trainY, valX, valY)
    # print(time.time() - start)
    # print(perf)
    # for i in range(10, 50, 10):
    #     timeTaken = checkTrainTime(architecture, i)
    #     print(timeTaken)
    #     runtimes.append(timeTaken)
    # plt.plot(runtimes)
    # plt.show()