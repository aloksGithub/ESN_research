import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.error_metrics import nrmse, r_squared
warnings.filterwarnings("ignore")
from utils import getDataLaser, printSavedBoResults
from NAS.ESN_BO import ESN_BO
rpy.verbosity(0)

if __name__ == "__main__":
    # printSavedBoResults("laser")
    trainX, trainY, valX, valY, testX, testY = getDataLaser()
    baseArchitecture = {'nodes': [{'type': 'Input', 'params': {'input_dim': trainX.shape[-1]}}, {'type': 'Reservoir', 'params': {'units': 1000, 'lr': 0.9, 'sr': 0.9, 'input_connectivity': 0.25, 'rc_connectivity': 0.25}}, {'type': 'Ridge', 'params': {'output_dim': trainY.shape[-1], 'ridge': 8.0e-05}}], 'edges': [[0, 1], [1, 2]]}
        
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
        "backup_bo/laser/backup_{}.obj".format(sys.argv[1]),
        True
    )
    errors, model = ga_bo.run()
    print("Result:", errors)
