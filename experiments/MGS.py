import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
import time
from reservoirpy.observables import (rmse, rsquare, nrmse, mse)
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

def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / denominator)

gaParams = {
    "evaluator": partial(NAS.evaluateArchitecture2, trainX=trainX, trainY=trainY, valX=valX, valY=valY),
    "generator": partial(NAS.generateRandomArchitecture, sampleX=trainX[:2000], sampleY=trainY[:2000]),
    "populationSize": 100,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 100,
    "minimizeFitness": True,
    "logModels": False,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0.01,
    "n_jobs": 25
}

if __name__ == "__main__":
    nrmseErrors = []
    r2Errors = []
    for i in range(20):
        models, performances, architectures = NAS.runGA(gaParams)
        print("Finsihed", min(performances))
        # allPreds = []
        # for architecture in architectures[:5]:
        #     for _ in range(10):
        #         model = NAS.constructModel(architecture)
        #         model = NAS.trainModel(model, np.concatenate([trainX, valX[:-20]]), np.concatenate([trainY, valY[:-20]]))
        #         prevOutput = valX[-20]
        #         preds = []
        #         for j in range(20+len(testX)):
        #             pred = NAS.runModel(model, prevOutput)
        #             prevOutput = pred
        #             preds.append(pred[0])
        #         preds = np.array(preds)
        #         allPreds.append(preds)
        # valErrors = []
        # for pred in allPreds:
        #     valError = nrmse(valY[-20:], pred[:20], 'mean')
        #     error = nrmse(testY, pred[-len(testY):], 'mean')
        #     valErrors.append(valError)
        # bestPred = allPreds[valErrors.index(min(valErrors))]
        # print(performances[0:5], min(valErrors), nrmse(testY, bestPred[-len(testY):], 'mean'), r_squared(testY, bestPred[-len(testY):]))
        # if min(valErrors)>0.0003:
        #     continue
        # nrmseErrors.append(nrmse(testY, bestPred[-len(testY):]))
        # r2Errors.append(r_squared(testY, bestPred[-len(testY):]))

    # print(np.array(nrmseErrors).mean(), np.array(nrmseErrors).std())
    # print(np.array(r2Errors).mean(), np.array(r2Errors).std())