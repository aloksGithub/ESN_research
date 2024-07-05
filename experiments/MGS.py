import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS import NAS
warnings.filterwarnings("ignore")
import sys
import pickle

rpy.verbosity(0)
output_dim = 1

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    
    return rmse / mean_norm
    
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / denominator)

def readSavedExperiment(path):
    file = open(path, 'rb')
    return pickle.load(file)

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Mackey glass dataset
def getData():
    data = np.load('./data/MG17.npy')
    data = data.reshape((data.shape[0],1))
    data = data[:3801,:]
    from scipy import stats
    data = stats.zscore(data)
    data.shape

    trainLen = 2000
    valLen = 286
    testLen = 286
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out

trainX, trainY, valX, valY, testX, testY = getData()

if __name__ == "__main__":
    numExperiments = 2
    nrmseErrors = []
    r2_squaredValues = []
    for i in range(numExperiments):
        ga = ESN_NAS(
            trainX,
            trainY,
            valX,
            valY,
            50,
            50,
            trainY.shape[-1],
            n_jobs=10,
            errorMetrics=[nrmse, r_squared],
            defaultErrors=[np.inf, 0],
            timeout=60,
            numEvals=3,
            saveLocation='backup/mgs/backup_{}.obj'.format(i),
            memoryLimit=756,
            isAutoRegressive=True
        )
        ga.run()
        nrmseErrors.append(ga.bestFitness[0])
        r2_squaredValues.append(ga.bestFitness[1])
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))
    
    # nrmseErrors = []
    # r2Errors = []
    # for i in range(5):
    #     error = False
    #     gaParams["experimentIndex"] = i
    #     while True:
    #         try:
    #             nrmse, r2 = NAS.runGA(gaParams, error)
    #             r2Errors.append(r2)
    #             nrmseErrors.append(nrmse)
    #             break
    #         except:
    #             print(traceback.format_exc())
    #             error = True
    # print(np.array(nrmseErrors).mean(), np.array(nrmseErrors).std())
    # print(np.array(r2Errors).mean(), np.array(r2Errors).std())
    # total = 0
    # r2_total = 0
    # for i in range(5):
    #     file = open('backup/{}/backup_{}.obj'.format(gaParams["dataset"], i), 'rb')
    #     data = pickle.load(file)
    #     fitnesses = data["allFitnesses"]
    #     minFitnesses = []
    #     total+=min(data["allFitnesses"])
    #     r2_total+=data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))]
    # print(total/5, r2_total/5)
