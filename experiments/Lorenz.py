import reservoirpy as rpy
import numpy as np
from functools import partial
import sys
import os
import warnings
import time
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
    data = np.load('./data/Lorenz_normed_2801.npy')

    trainLen = 1957
    valLen = 399
    testLen = 444
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

def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / denominator)

gaParams = {
    "evaluator": partial(NAS.evaluateArchitecture, trainX=trainX, trainY=trainY, valX=valX, valY=valY),
    "generator": partial(NAS.generateRandomArchitecture, sampleX=trainX, sampleY=trainY),
    "populationSize": 15,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 5,
    "minimizeFitness": True,
    "logModels": False,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0,
    "n_jobs": 5
}

if __name__ == "__main__":
    nrmseErrors = []
    r2Errors = []
    for i in range(5):
        startTime = time.time()
        models, performances, architectures = NAS.runGA(gaParams)
        model = NAS.Ensemble([NAS.constructModel(architectures[0]) for _ in range(5)])
        model.train(np.concatenate([trainX, valX]), np.concatenate([trainY, valY]))
        NAS.runModel(model, valX[-100:])
        prevOutput = testX[0]
        preds = []
        for j in range(len(testX)):
            pred = NAS.runModel(model, prevOutput)
            prevOutput = pred
            preds.append(pred[0])
        preds = np.array(preds)
        performance_r2 = r_squared(testY, preds)
        print("Performance", performance_r2, nrmse(testY, preds), "Time taken", time.time() - startTime)
        nrmseErrors.append(nrmse(testY, preds))
        r2Errors.append(performance_r2)

    print(np.array(nrmseErrors).mean(), np.array(nrmseErrors).std())
    print(np.array(r2Errors).mean(), np.array(r2Errors).std())