import math
import numpy as np
from reservoirpy.observables import (mse)

def nrmse_sunspots(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
    if math.isnan(error):
        return np.inf
    else:
        return error

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    error = rmse/mean_norm
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
        return r2
