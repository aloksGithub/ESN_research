import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
rpy.verbosity(0)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.algorithms.types import EvalParams, ExperimentData
from src.error_metrics import nrmse, nrmse_sunspots, r_squared
from src.datasets import getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots, getDataWater
from src.algorithms.ESN_BO import ESN_BO

baseArchitecture = {'nodes': [{'type': 'Input', 'params': {'input_dim': 1}}, {'type': 'Reservoir', 'params': {'units': 1000, 'lr': 0.9, 'sr': 0.9, 'input_connectivity': 0.25, 'rc_connectivity': 0.25}}, {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 8.0e-05}}], 'edges': [[0, 1], [1, 2]]}

def runBOExperiment(dataset, dataLoader, errorMetrics, isAutoregressive, save_folder='results/bo'):
    trainX, trainY, valX, valY, testX, testY = dataLoader()
    experimentData = ExperimentData(trainX, trainY, valX, valY, testX, testY)
    evalParams = EvalParams(
        numEvals=3,
        errorMetrics=errorMetrics,
        defaultErrors=[100000, 0],
        minimizeFitness=True,
        timeout=60,
        memoryLimit=0, # Note: memory limit is not used in ESN_BO
        isAutoRegressive=isAutoregressive
    )
    baseArchitecture['nodes'][0]['params']['input_dim'] = trainX.shape[1]
    baseArchitecture['nodes'][-1]['params']['output_dim'] = trainX.shape[1]
    bo = ESN_BO(
        experimentData,
        evalParams,
        n_rand=2000,
        iterations=2000,
        seedModel=baseArchitecture,
        n_jobs=3,
        saveLocation="{}/{}/backup_{}.obj".format(save_folder, dataset, sys.argv[1]),
    )
    bo.run()

if __name__ == "__main__":
    runBOExperiment("laser", getDataLaser, [nrmse, r_squared], True)
    runBOExperiment("dde", getDataDDE, [nrmse, r_squared], True)
    runBOExperiment("lorenz", getDataLorenz, [nrmse, r_squared], True)
    runBOExperiment("mgs", getDataMGS, [nrmse, r_squared], True)
    runBOExperiment("sunspots", getDataSunspots, [nrmse_sunspots, r_squared], False)
    runBOExperiment("water", getDataWater, [nrmse, r_squared], False)
