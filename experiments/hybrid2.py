import reservoirpy as rpy
import numpy as np
import sys
import os


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS import EvalParams, ExperimentData, GAParams, ModelParams
from NAS.utils import runModel
from NAS.ESN_NAS2 import ESN_NAS2
from NAS.error_metrics import nrmse, nrmse_sunspots, r_squared
from utils import getDataMGS, getDataDDE, getDataLaser, getDataLorenz, getDataSunspots, getDataWater, printSavedResults, printSavedResultsAutoRegressive
import warnings
warnings.filterwarnings("ignore")
rpy.verbosity(0)

def runExperiment(dataset, dataLoader, errorMetrics, isAutoregressive, earlyStop=None, save_folder='hybrid2'):
    trainX, trainY, valX, valY, testX, testY = dataLoader()
    experimentData = ExperimentData(trainX, trainY, valX, valY, testX, testY)
    evalParams = EvalParams(
        numEvals=3,
        errorMetrics=errorMetrics,
        defaultErrors=[100000, 0],
        timeout=60,
        memoryLimit=756,
        minimizeFitness=True,
        isAutoRegressive=isAutoregressive,
    )
    gaParams = GAParams(
        generations=20,
        populationSize=40,
        crossoverProbability=0.7,
        mutationProbability=0.2,
        eliteSize=1,
        stagnationReset=5,
        earlyStop=earlyStop,
    )
    modelParams = ModelParams(
        num_nodes_range=(1, 2),
    )
    nrmseErrors = []
    r2_squaredValues = []
    print(f'========================Starting GA for dataset {dataset}========================')
    for i in range(5):
        ga = ESN_NAS2(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            n_jobs=20,
            saveLocation='{}/{}/backup_{}.obj'.format(save_folder, dataset, i),
            bo_init=0,
            bo_iter=5
        )
        ga.run()
        if isAutoregressive:
            nrmseErrors.append(ga.bestFitness[0])
            r2_squaredValues.append(ga.bestFitness[1])
        else:
            model = ga.bestModel
            runModel(model, valX)
            preds = runModel(model, testX)
            nrmseError = ga.evalParams.errorMetrics[0](testY, preds)
            r2Error = ga.evalParams.errorMetrics[1](testY, preds)
            nrmseErrors.append(nrmseError)
            r2_squaredValues.append(r2Error)
    print(f'========================Performance for dataset {dataset}========================')
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

def printAllSavedResults(save_folder='hybrid2'):
    printSavedResults(save_folder, 'mgs')
    printSavedResults(save_folder, 'lorenz')
    printSavedResults(save_folder, 'dde')
    printSavedResults(save_folder, 'laser')
    printSavedResultsAutoRegressive(save_folder, 'sunspots', getDataSunspots)
    printSavedResultsAutoRegressive(save_folder, 'water', getDataWater)

if __name__ == "__main__":
    # runExperiment('lorenz', getDataLorenz, [nrmse, r_squared], True, 0.001)
    # runExperiment('mgs', getDataMGS, [nrmse, r_squared], True, 0.02)
    # runExperiment('dde', getDataDDE, [nrmse, r_squared], True, 0.0003)
    # runExperiment('laser', getDataLaser, [nrmse, r_squared], True, 1.1)
    # runExperiment('sunspots', getDataSunspots, [nrmse_sunspots, r_squared], False, None)
    runExperiment('water', getDataWater, [nrmse, r_squared], False, None, 'hybrid2_small_models')

