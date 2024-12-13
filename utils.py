import pandas as pd
import numpy as np
import pickle
import math

def readSavedExperiment(path):
    file = open(path, 'rb')
    return pickle.load(file)

def printSavedResults(directory, dataset):
    nrmseErrors = []
    r2_squaredValues = []
    for i in range(5):
        ga = readSavedExperiment('{}/{}/backup_{}.obj'.format(directory, dataset, i))
        nrmseErrors.append(ga.bestFitness[0])
        r2_squaredValues.append(ga.bestFitness[1])
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))

def printSavedBoResults(dataset):
    nrmseErrors = []
    rSquaredValues = []
    for i in range(5):
        bo = readSavedExperiment('backup_bo/{}/backup_{}.obj'.format(dataset, i))
        mainErrors = [e[0] for e in bo.performances]
        bestErrors = bo.performances[mainErrors.index(min(mainErrors) if bo.minimizeFitness else max(mainErrors))]
        nrmseError = bestErrors[0]
        r2Error = bestErrors[1]
        nrmseErrors.append(nrmseError)
        rSquaredValues.append(r2Error)
        print("Result:", nrmseError, r2Error)
    print("Errors:")
    print(nrmseErrors)
    print(rSquaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(rSquaredValues), np.std(rSquaredValues)))

def printOldSavedResults(dataset):
    nrmseErrors = []
    r2_squaredValues = []
    total = 0
    r2_total = 0
    for i in range(5):
        data = readSavedExperiment('old_backup/{}/backup_{}.obj'.format(dataset, i))
        total+=min(data["allFitnesses"])
        nrmseErrors.append(min(data["allFitnesses"]))
        r2_squaredValues.append(data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))])
        r2_total+=data["fitnesses2"][data["allFitnesses"].index(min(data["allFitnesses"]))]
    print("Errors:")
    print(nrmseErrors)
    print(r2_squaredValues)
    print("Averaged errors:")
    print("NRMSE: {} ({})".format(np.average(nrmseErrors), np.std(nrmseErrors)))
    print("R2: {} ({})".format(np.average(r2_squaredValues), np.std(r2_squaredValues)))


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Mackey glass dataset
def getDataMGS():
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

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Santafe laser dataset
def getDataLaser():
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

# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Neutral Normed DDE dataset
def getDataDDE():
    data = np.load('./data/Neutral_normed_2801.npy')
    from scipy import stats
    data = stats.zscore(data)
    data.shape

    trainLen = 2000
    valLen = 500
    testLen = 500
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Lorenz dataset
def getDataLorenz():
    data = np.load('./data/Lorenz_normed_2801.npy')
    from scipy import stats
    data = stats.zscore(data)
    data.shape

    trainLen = 2000
    valLen = 444
    testLen = 444
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out

def getDataSunspots():
    sunspots = pd.read_csv("./datasets/Sunspots.csv")
    data = sunspots.loc[:,"Monthly Mean Total Sunspot Number"].to_numpy()
    data = np.expand_dims(data, axis=1)

    trainLen = 1600
    valLen = 500
    testLen = 1074
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:trainLen+valLen]
    val_out = data[trainLen+1:trainLen+valLen+1]
    test_in = data[trainLen+valLen:trainLen+valLen+testLen]
    test_out = data[trainLen+valLen+1:trainLen+valLen+testLen+1]
    return train_in, train_out, val_in, val_out, test_in, test_out

def getDataWater():
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    firstCol = water[:, 0]
    lastRow = water[-1, 1:]
    data = np.expand_dims(np.concatenate((firstCol, lastRow)), axis=1)

    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)

    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    val_in = data[trainLen:valLen]
    val_out = data[trainLen+1:valLen+1]
    test_in = data[valLen:len(data)-1]
    test_out = data[valLen+1:]
    return train_in, train_out, val_in, val_out, test_in, test_out

def getDataWaterMultiStep(n: int):
    water = pd.read_csv("./datasets/Water.csv").to_numpy()
    firstCol = water[:, 0]
    lastRow = water[-1, 1:]
    data = np.expand_dims(np.concatenate((firstCol, lastRow)), axis=1)

    trainLen = math.floor(len(water)*0.5)
    valLen = math.floor(len(water)*0.7)

    train_in = data[0:trainLen]
    train_out = data[0+n:trainLen+n]
    val_in = data[trainLen:valLen]
    val_out = data[trainLen+n:valLen+n]
    test_in = data[valLen:len(data)-n]
    test_out = data[valLen+n:]
    return train_in, train_out, val_in, val_out, test_in, test_out