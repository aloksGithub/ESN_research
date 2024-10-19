import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.error_metrics import r_squared
from NAS.utils import runModel
warnings.filterwarnings("ignore")
import math
from reservoirpy.observables import (mse)
from utils import getDataSunspots, readSavedExperiment
from NAS.ESN_BO import ESN_BO

rpy.verbosity(0)

def nrmse(y_true, y_pred):
    mseError = mse(y_true, y_pred)
    variance = np.asarray(y_true).var()
    error = np.sqrt(mseError/variance)
    if math.isnan(error):
        return np.inf
    else:
        return error

def printSavedBoResults():
    _, _, _, _, testX, testY = getDataSunspots()
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_bo/sunspots/backup_{}.obj'.format(i))
        model = bo.bestModel
        preds = runModel(model, testX)
        nrmseError = nrmse(testY, preds)
        r2Error = r_squared(testY, preds)
        nrmseErrors.append(nrmseError)
        rSquaredValues.append(r2Error)
        print("Result:", nrmseError, r2Error)
    print("Errors:")
    print(nrmseErrors)
    print(rSquaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(rSquaredValues), np.std(rSquaredValues)))

if __name__ == "__main__":
    trainX, trainY, valX, valY, testX, testY = getDataSunspots()
    baseArchitecture = {'nodes': [{'type': 'Input', 'params': {'input_dim': 1}}, {'type': 'Reservoir', 'params': {'units': 1000, 'lr': 0.9, 'sr': 0.9, 'input_connectivity': 0.25, 'rc_connectivity': 0.25}}, {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 8.0e-05}}], 'edges': [[0, 1], [1, 2]]}
    
    ga_bo = ESN_BO(
        trainX,
        trainY,
        valX,
        valY,
        500,
        2000,
        trainY.shape[-1],
        baseArchitecture,
        3,
        3,
        [nrmse, r_squared],
        [np.inf, 0],
        True,
        180,
        "backup_bo/sunspots/backup_{}.obj".format(sys.argv[1]),
        False
    )
    errors, model = ga_bo.run()
    preds = runModel(model, testX)
    nrmseError = nrmse(testY, preds)
    r2Error = r_squared(testY, preds)
    print("Result:", nrmseError, r2Error)
