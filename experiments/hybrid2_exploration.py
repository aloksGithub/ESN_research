import reservoirpy as rpy
import numpy as np
import sys
import os


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.utils import runModel
from NAS.ESN_NAS2 import ESN_NAS2
from NAS.error_metrics import nrmse, nrmse_sunspots, r_squared
from utils import getDataMGS, getDataDDE, getDataLaser, getDataLorenz, getDataSunspots, getDataWater, printSavedResults, printSavedResultsAutoRegressive
import warnings
warnings.filterwarnings("ignore")
rpy.verbosity(0)

def runExperiment(dataset, dataLoader, errorMetrics, isAutoregressive):
    trainX, trainY, valX, valY, testX, testY = dataLoader()
    nrmseErrors = []
    r2_squaredValues = []
    print(f'========================Starting GA for dataset {dataset}========================')
    for i in range(5):
        ga = ESN_NAS2(
            trainX,
            trainY,
            valX,
            valY,
            20,
            40,
            trainY.shape[-1],
            n_jobs=20,
            errorMetrics=errorMetrics,
            defaultErrors=[100000, 0],
            timeout=60,
            numEvals=3,
            saveLocation='hybrid2_exploration/{}/backup_{}.obj'.format(dataset, i),
            memoryLimit=756,
            isAutoRegressive=isAutoregressive,
            bo_init=3,
            bo_iter=2
        )
        ga.run()
        if isAutoregressive:
            nrmseErrors.append(ga.bestFitness[0])
            r2_squaredValues.append(ga.bestFitness[1])
        else:
            model = ga.bestModel
            runModel(model, valX)
            preds = runModel(model, testX)
            nrmseError = ga.errorMetrics[0](testY, preds)
            r2Error = ga.errorMetrics[1](testY, preds)
            nrmseErrors.append(nrmseError)
            r2_squaredValues.append(r2Error)
    print(f'========================Performance for dataset {dataset}========================')
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

def printAllSavedResults():
    printSavedResults('hybrid2_exploration_long', 'mgs')
    printSavedResults('hybrid2_exploration_long', 'lorenz')
    printSavedResults('hybrid2_exploration_long', 'dde')
    printSavedResults('hybrid2_exploration_long', 'laser')
    printSavedResultsAutoRegressive('hybrid2_exploration_long', 'sunspots', getDataSunspots)
    printSavedResultsAutoRegressive('hybrid2_exploration_long', 'water', getDataWater)

if __name__ == "__main__":
    runExperiment('mgs', getDataMGS, [nrmse, r_squared], True)
    runExperiment('lorenz', getDataLorenz, [nrmse, r_squared], True)
    runExperiment('dde', getDataDDE, [nrmse, r_squared], True)
    runExperiment('laser', getDataLaser, [nrmse, r_squared], True)
    runExperiment('sunspots', getDataSunspots, [nrmse_sunspots, r_squared], False)
    runExperiment('water', getDataWater, [nrmse, r_squared], False)

