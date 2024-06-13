import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel, smape, trainModel
warnings.filterwarnings("ignore")
import sys
import pandas as pd
import math
from reservoirpy.observables import (mse)
from reservoirpy.nodes import Reservoir, Ridge, IPReservoir, FORCE, LMS, RLS, Input, NVAR
import copy
from bayes_opt import BayesianOptimization
import pickle
from bayes_opt import BayesianOptimization

rpy.verbosity(0)

def getDataSingleCol():
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    firstCol = water[:, 0]
    lastRow = water[-1, 1:]
    allData = np.expand_dims(np.concatenate((firstCol, lastRow)), axis=1)
    
    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)
    
    train_in = allData[0:trainLen]
    train_out = allData[0:trainLen]
    val_in = allData[trainLen:valLen]
    val_out = allData[trainLen:valLen]
    test_in = allData[valLen:]
    test_out = allData[valLen:]
    return train_in, train_out, val_in, val_out, test_in, test_out, water

trainX, trainY, valX, valY, testX, testY, allData = getDataSingleCol()

def black_box_vanilla(**params):
    ridge = params['ridge']
    del params['ridge']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    readout = Ridge(output_dim=1, ridge=ridge)
    model = reservoir>>readout
    model.fit(trainX, trainY, warmup=50)
    preds = model.run(valX)
    error = mse(valY, preds)
    return -error

# def black_box_ip(**params):
#     ridge = params['ridge']
#     del params['ridge']
#     params['units'] = int(params['units'])
#     reservoir = IPReservoir(**params)
#     readout = Ridge(output_dim=1, ridge=ridge)
#     model = reservoir>>readout
#     model.fit(trainX, trainY, warmup=50)
#     preds = model.run(valX)
#     error = mse(valY, preds)
#     return error

# def black_box_nvar(**params):
#     readout = Ridge(output_dim=10, ridge=params['ridge'])
#     del params['ridge']
#     nvar = NVAR(delay=int(params['delay']), order=2, strides=int(params['strides']))
#     model = nvar >> readout
#     model.fit(trainX, trainY, warmup=50)
#     preds = model.run(valX)
#     error = mse(valY, preds)
#     return error

# def black_box_rls(**params):
#     readout = RLS(output_dim=10, alpha=params['alpha'])
#     del params['alpha']
#     params['units'] = int(params['units'])
#     reservoir = Reservoir(**params)
#     model = reservoir >> readout
#     model.fit(trainX, trainY, warmup=50)
#     preds = model.run(valX)
#     error = mse(valY, preds)
#     return error

# def black_box_lms(**params):
#     readout = RLS(output_dim=10, alpha=params['alpha'])
#     del params['alpha']
#     params['units'] = int(params['units'])
#     reservoir = Reservoir(**params)
#     model = reservoir >> readout
#     model.fit(trainX, trainY, warmup=50)
#     preds = model.run(valX)
#     error = mse(valY, preds)
#     return error

if __name__ == "__main__":
    pbounds = {'units': (10, 3000), 'lr': (0.2, 1), 'sr': (0.5, 2), 'input_connectivity': (0.05, 0.5), 'rc_connectivity': (0.05, 0.5), 'fb_connectivity': (0.05, 0.5), 'ridge': (0, 0.0001)}

    optimizer = BayesianOptimization(
        f=black_box_vanilla,
        pbounds=pbounds,
        random_state=1,
        
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    params = optimizer.max["params"]
    ridge = params['ridge']
    del params['ridge']
    params['units'] = int(params['units'])
    reservoir = Reservoir(**params)
    readout = Ridge(output_dim=1, ridge=ridge)
    model = reservoir>>readout
    model.fit(trainX, trainY, warmup=50)
    model.run(valX)
    preds = model.run(testX)
    print("MSE1:", mse(testY, preds))
    print("SMAPE1:", smape(testY, preds))