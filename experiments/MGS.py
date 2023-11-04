import reservoirpy as rpy
import numpy as np
from sklearn.metrics import r2_score
import math
from functools import partial
import sys
import os
import warnings
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS import NAS
warnings.filterwarnings("ignore")

rpy.verbosity(0)
output_dim = 1


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Mackey glass dataset

def getData():
    data = np.load('./data/MG17.npy')
    data = data.reshape((data.shape[0],1))
    data = data[:2801,:]
    from scipy import stats
    data = stats.zscore(data)
    data.shape

    trainLen = 1800
    valLen = 300
    testLen = 286
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out

trainX, trainY, valX, valY, testX, testY = getData()

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    return rmse / mean_norm

gaParams = {
    "evaluator": partial(NAS.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY),
    "generator": partial(NAS.generateRandomArchitecture, sampleX=trainX[:100], sampleY=trainY[:100]),
    "populationSize": 24,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 5,
    "minimizeFitness": True,
    "logModels": True,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0,
    "n_jobs": 24
}

if __name__ == "__main__":
    nrmseErrors = []
    r2Errors = []
    for i in range(1):
        models, performances, architectures = NAS.runGA(gaParams)
        model = NAS.Ensemble(models[:5])
        startInput = testX[0]
        prevOutput = testX[0]
        preds = []
        for j in range(len(testX)):
            pred = NAS.runModel(model, prevOutput)
            prevOutput = pred
            preds.append(pred[0])
        preds = np.array(preds)
        performance_r2 = r2_score(testY, preds)
        print("Performance", performance_r2, nrmse(testY, preds))
        nrmseErrors.append(nrmse(testY, preds))
        r2Errors.append(performance_r2)

    print(np.array(nrmseErrors).mean(), np.array(nrmseErrors).std())
    print(np.array(r2Errors).mean(), np.array(r2Errors).std())