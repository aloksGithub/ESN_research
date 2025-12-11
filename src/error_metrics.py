import math
import numpy as np
from reservoirpy.observables import (mse)
from sklearn.metrics import r2_score

def nrmse_sunspots(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
    if math.isnan(error):
        return 100000
    else:
        return error

def r_squared(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred, multioutput="variance_weighted")
    except:
        return 0

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        
    norm_diff = np.linalg.norm(y_true - y_pred, axis=1)
    mean_sq_error = np.mean(norm_diff**2) 
    rmse = np.sqrt(mean_sq_error)
    
    y_mean = np.mean(y_true, axis=0)
    mean_norm = np.linalg.norm(y_mean)
    
    if mean_norm == 0:
        return 100000
        
    return rmse / mean_norm
