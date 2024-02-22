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
import traceback
import sys
import pickle
import matplotlib.pyplot as plt

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

gaParams = {
    "evaluator": partial(NAS.evaluateArchitecture2, trainX=trainX, trainY=trainY, valX=valX, valY=valY),
    "generator": partial(NAS.generateRandomArchitecture, sampleX=trainX[:2000], sampleY=trainY[:2000], validThreshold=1, numVal=200),
    "populationSize": 100,
    "eliteSize": 1,
    "stagnationReset": 5,
    "generations": 50,
    "minimizeFitness": True,
    "logModels": False,
    "seedModels": [],
    "crossoverProbability": 0.7,
    "mutationProbability": 0.2,
    "earlyStop": 0,
    "n_jobs": 25,
    "saveModels": True,
    "dataset": "mgs"
}

if __name__ == "__main__":
    nrmseErrors = []
    r2Errors = []
    for i in range(5):
        error = False
        gaParams["experimentIndex"] = i
        while True:
            try:
                nrmse, r2 = NAS.runGA(gaParams, error)
                r2Errors.append(r2)
                nrmseErrors.append(nrmse)
                break
            except:
                print(traceback.format_exc())
                error = True
    print(np.array(nrmseErrors).mean(), np.array(nrmseErrors).std())
    print(np.array(r2Errors).mean(), np.array(r2Errors).std())
    # convergenceLines = []
    # total = 0
    # for i in range(5):
    #     file = open('backup/{}/backup_{}.obj'.format(gaParams["dataset"], i), 'rb')
    #     data = pickle.load(file)
    #     fitnesses = data["allFitnesses"]
    #     minFitnesses = []
    #     total+=min(data["allFitnesses"])
    #     for i in range(data["params"]["populationSize"], len(fitnesses), data["params"]["populationSize"]):
    #         minFitnesses.append(min(fitnesses[i-data["params"]["populationSize"]:i]))
    #     convergenceLines.append(minFitnesses)
