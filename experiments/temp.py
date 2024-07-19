import multiprocessing
import time
from joblib import Parallel, delayed
import pickle
import sys
import os
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
import numpy as np
import math

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    
    error = rmse / mean_norm
    if math.isnan(error):
        return np.inf
    else:
        return error
    
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (numerator / denominator)
    if math.isnan(r2):
        return 0
    else:
        return 1 - (numerator / denominator)

def square(val):
    time.sleep(val)
    print("Done")
    return val*val

def worker(queue, func, args, n_jobs):
    dataStore = [None] * len(args)
    queue.put(dataStore)
    def funcWrapper(jobIndex, *args):
        result = func(*args)
        dataStore[jobIndex] = result
        while not queue.empty():
            queue.get() # Call get on queue to empty it
        queue.put(dataStore)
        
    parallel = Parallel(n_jobs=n_jobs, require='sharedmem')
    parallel(delayed(funcWrapper)(i, *args[i]) for i in range(len(args)))

def parallelProcessing(func, args, n_jobs, timeout):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(queue, func, args, n_jobs))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join()
    result = queue.get_nowait()
    return result

if __name__ == "__main__":
    # results1 = parallelProcessing(square, [(2,), (3,), (8,)], 3, 6)
    # print(results1)
    # results2 = parallelProcessing(square, [(4,), (15,), (6,)], 3, 11)
    # print(results2)
    
    file = open("./backup/sunspots/backup_0.obj", 'rb')
    ga = pickle.load(file)
    print(ga.population)
