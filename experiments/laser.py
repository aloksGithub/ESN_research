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
import pandas as pd
warnings.filterwarnings("ignore")
import traceback
import sys
import pickle

rpy.verbosity(0)
output_dim = 1

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Santafe laser dataset

def getData():
    sunspots = pd.read_csv("./data/santafelaser.csv")
    data = np.array(sunspots)
    data = data.reshape((data.shape[0],1))
    data = data[:3801,:]
    from scipy import stats
    data = stats.zscore(data)

    trainLen = 2000
    valLen = 100
    testLen = 100
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
    "generator": partial(NAS.generateRandomArchitecture, sampleX=np.concatenate([trainX[:2000], valX]), sampleY=np.concatenate([trainY[:2000], valY]), validThreshold=100, numVal=100),
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
    "saveModels": False,
    "dataset": "laser"
}

if __name__ == "__main__":
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
    total = 0
    r2_total = 0
    for i in range(5):
        file = open('backup/{}/backup_{}.obj'.format(gaParams["dataset"], i), 'rb')
        data = pickle.load(file)
        fitnesses = data["allFitnesses"]
        minFitnesses = []
        total+=min(data["allFitnesses"])
        r2_total+=data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))]
    print(total/5, r2_total/5)
    