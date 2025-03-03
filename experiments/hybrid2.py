import reservoirpy as rpy
import numpy as np
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.ESN_NAS2 import ESN_NAS2
from NAS.error_metrics import nrmse, r_squared
from utils import getDataMGS, getDataDDE, getDataLorenz, printSavedResults
import warnings
warnings.filterwarnings("ignore")
rpy.verbosity(0)

if __name__ == "__main__":
    trainX, trainY, valX, valY, testX, testY = getDataDDE()
    nrmseErrors = []
    r2_squaredValues = []
    for i in [0, 1, 2, 3, 4]:
        ga = ESN_NAS2(
            trainX,
            trainY,
            valX,
            valY,
            10,
            20,
            trainY.shape[-1],
            n_jobs=20,
            errorMetrics=[nrmse, r_squared],
            defaultErrors=[100000, 0],
            timeout=60,
            numEvals=3,
            saveLocation='temp/dde/backup_{}.obj'.format(i),
            memoryLimit=756,
            isAutoRegressive=True,
            bo_init=10,
            bo_iter=10
        )
        ga.run()
        nrmseErrors.append(ga.bestFitness[0])
        r2_squaredValues.append(ga.bestFitness[1])
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

