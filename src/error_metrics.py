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

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    diff = y_true - y_pred
    mean_sq_error = np.mean(np.sum(diff**2, axis=1))
    mean_norm = np.linalg.norm(np.mean(y_true, axis=0))
        
    rmse = np.sqrt(mean_sq_error)
    
    if mean_norm == 0:
        return 100000
        
    error = rmse / mean_norm
    return error
    
def r_squared(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred, multioutput="variance_weighted")
    except:
        return 0
