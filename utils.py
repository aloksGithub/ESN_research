import pandas as pd
import numpy as np
import pickle
import math
from NAS.ESN_NAS import ESN_NAS
from NAS.utils import runModel


def readSavedExperiment(path):
    file = open(path, "rb")
    return pickle.load(file)


def printSavedResults(directory, dataset, isAutoregressive=True):
    nrmseErrors = []
    r2_squaredValues = []
    times = []
    for i in range(5):
        ga: ESN_NAS = readSavedExperiment("{}/{}/backup_{}.obj".format(directory, dataset, i))
        try:
            times.append(sum(ga.generationTimes))
        except:
            times.append(0)
        best_model = ga.bestModel
        if isAutoregressive:
            runModel(best_model, ga.experimentData.trainX)
            prevOutput = ga.experimentData.valX[0]
            preds = []
            for _ in range(len(ga.experimentData.valX)):
                pred = runModel(best_model, prevOutput)
                prevOutput = pred
                preds.append(pred[0])
            preds = np.array(preds)
            nrmse_error = ga.evalParams.errorMetrics[0](ga.experimentData.valY, preds)
            r2_error = ga.evalParams.errorMetrics[1](ga.experimentData.valY, preds)
        else:
            runModel(best_model, ga.experimentData.valX)
            preds = runModel(best_model, ga.experimentData.testX)
            nrmse_error = ga.evalParams.errorMetrics[0](ga.experimentData.testY, preds)
            r2_error = ga.evalParams.errorMetrics[1](ga.experimentData.testY, preds)

        nrmseErrors.append(nrmse_error)
        r2_squaredValues.append(r2_error)
    print("==============================================================")
    print("{} Errors:".format(dataset))
    print("NRMSE:", nrmseErrors)
    print("R2:", r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))
    print("Times:")
    print(times)
    print("Average time: {} ({})".format(np.average(times), np.std(times)))

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Mackey glass dataset
def getDataMGS():
    data = np.load("./data/MG17.npy")
    data = data.reshape((data.shape[0], 1))
    data = data[:2801, :]
    from scipy import stats

    data = stats.zscore(data)
    data.shape

    trainLen = 2300
    valLen = 286
    testLen = 0
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = data[trainLen + valLen : trainLen + valLen + testLen]
    test_out = data[trainLen + valLen + 1 : trainLen + valLen + testLen + 1]
    return train_in, train_out, val_in, val_out, test_in, test_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Santafe laser dataset
def getDataLaser():
    sunspots = pd.read_csv("./data/santafelaser.csv")
    data = np.array(sunspots)
    data = data.reshape((data.shape[0], 1))
    data = data[:2801, :]
    from scipy import stats

    data = stats.zscore(data)

    trainLen = 2300
    valLen = 100
    testLen = 0
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = data[trainLen + valLen : trainLen + valLen + testLen]
    test_out = data[trainLen + valLen + 1 : trainLen + valLen + testLen + 1]
    return train_in, train_out, val_in, val_out, test_in, test_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Neutral Normed DDE dataset
def getDataDDE():
    data = np.load("./data/Neutral_normed_2801.npy")

    trainLen = 2300
    valLen = 500
    testLen = 0
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = data[trainLen + valLen : trainLen + valLen + testLen]
    test_out = data[trainLen + valLen + 1 : trainLen + valLen + testLen + 1]
    return train_in, train_out, val_in, val_out, test_in, test_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Lorenz dataset
def getDataLorenz():
    data = np.load("./data/Lorenz_normed_2801.npy")

    trainLen = 2300
    valLen = 444
    testLen = 0
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = data[trainLen + valLen : trainLen + valLen + testLen]
    test_out = data[trainLen + valLen + 1 : trainLen + valLen + testLen + 1]
    return train_in, train_out, val_in, val_out, test_in, test_out


def getDataSunspots():
    sunspots = pd.read_csv("./data/Sunspots.csv")
    data = sunspots.loc[:, "Monthly Mean Total Sunspot Number"].to_numpy()
    data = np.expand_dims(data, axis=1)

    trainLen = 1600
    valLen = 500
    testLen = 1074
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = data[trainLen + valLen : trainLen + valLen + testLen]
    test_out = data[trainLen + valLen + 1 : trainLen + valLen + testLen + 1]
    return train_in, train_out, val_in, val_out, test_in, test_out


def getDataWater():
    return getDataWaterMultiStep(1)


def getDataWaterMultiStep(n: int):
    water = pd.read_csv("./data/Water.csv").to_numpy()
    firstCol = water[:, 0]
    lastRow = water[-1, 1:]
    data = np.expand_dims(np.concatenate((firstCol, lastRow)), axis=1)

    trainLen = math.floor(len(water) * 0.5)
    valLen = math.floor(len(water) * 0.7)

    train_in = data[0:trainLen]
    train_out = data[0 + n : trainLen + n]
    val_in = data[trainLen:valLen]
    val_out = data[trainLen + n : valLen + n]
    test_in = data[valLen : len(data) - n]
    test_out = data[valLen + n :]
    return train_in, train_out, val_in, val_out, test_in, test_out
