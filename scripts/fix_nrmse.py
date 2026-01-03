import numpy as np
import math
import sys
import os
import pickle
import reservoirpy as rpy
from sklearn.metrics import r2_score

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.algorithms.ESN_BO import ESN_BO
from src.algorithms.ESN_GA import ESN_GA
from src.error_metrics import nrmse, r_squared

rpy.verbosity(0)

def old_nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    error = rmse/mean_norm
    if math.isnan(error):
        return 100000
    else:
        return error

def convert_nrmse(old_nrmse_value, y_true):
    y_true = np.array(y_true)
    
    # If 1D, the values are identical (factor is 1)
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        return old_nrmse_value

    D = y_true.shape[1]
    
    # 1. Scalar mean (what the old function used)
    scalar_mean = np.mean(y_true)
    
    # 2. Vector mean norm (what the new function uses)
    vector_mean = np.mean(y_true, axis=0)
    vector_mean_norm = np.linalg.norm(vector_mean)
    
    if vector_mean_norm == 0:
        return 0 # Or handle appropriately
        
    # 3. Calculate conversion factor
    factor = (np.sqrt(D) * np.abs(scalar_mean)) / vector_mean_norm
    
    return old_nrmse_value * factor

def readSavedExperiment(path):
    file = open(path, "rb")
    return pickle.load(file)

def fix_error_function(directory, dataset, isGA=True):
    for i in range(5):
        path = "{}/{}/backup_{}.obj".format(directory, dataset, i)
        ga: ESN_GA | ESN_BO = readSavedExperiment(path)
        errors = [errors[0] for errors in ga.fitnesses]
        print(min(errors) == min(errors[:4000]))
        # print(len(ga.architectures), len(ga.fitnesses), ga.modelGenerationIndices)
        # y_true = ga.experimentData.valY

        # if isGA:
        #     print(np.array(ga.fitnesses).shape)
        #     corrected_fitnesses = [[convert_nrmse(fitness[0], y_true), fitness[1]] for fitness in ga.fitnesses]
        #     ga.fitnesses = corrected_fitnesses
        # else:
        #     print(np.array(ga.performances).shape)
        #     corrected_fitnesses = [[convert_nrmse(fitness[0], y_true), fitness[1]] for fitness in ga.performances]
        #     ga.performances = corrected_fitnesses
        # ga.evalParams.errorMetrics = [nrmse, r_squared]
        
        # file = open(path, "wb")
        # pickle.dump(ga, file)


if __name__ == "__main__":
    print("==============================ESNAS==============================")
    # fix_error_function('results/esnas', 'mgs')
    # fix_error_function('results/esnas', 'lorenz')
    # fix_error_function('results/esnas', 'dde')
    # fix_error_function('results/esnas', 'laser')
    fix_error_function('results/esnas', 'sunspots')
    fix_error_function('results/esnas', 'water')

    print("==============================GA==============================")
    # fix_error_function('results/ga', 'mgs')
    # fix_error_function('results/ga', 'lorenz')
    # fix_error_function('results/ga', 'dde')
    # fix_error_function('results/ga', 'laser')
    fix_error_function('results/ga', 'sunspots')
    fix_error_function('results/ga', 'water')
    # print("==============================BO==============================")
    # fix_error_function('results/bo', 'lorenz', isGA=False)
    # fix_error_function('results/bo', 'dde', isGA=False)

