import pandas as pd
import numpy as np
import math

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
