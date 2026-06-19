import pandas as pd
import numpy as np
import math

NUM_TEST = 5


def _contiguous_warmup(val_in, val_out, test_in, test_out):
    """Build warmup lists for contiguous test sets.

    For contiguous layouts the warmup for test[0] is the validation set,
    and the warmup for test[i] (i>0) is the preceding test set.
    """
    warmup_in = [val_in] + [test_in[i] for i in range(len(test_in) - 1)]
    warmup_out = [val_out] + [test_out[i] for i in range(len(test_out) - 1)]
    return warmup_in, warmup_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Mackey glass dataset
def getDataMGS():
    raw = np.load("./data/MG17.npy")
    raw = raw.reshape((raw.shape[0], 1))

    # Compute normalization stats from the original 2801 window
    orig_window = raw[:2801, :]
    norm_mean = orig_window.mean()
    norm_std = orig_window.std()

    # Normalize the full raw data with the original window's stats
    data = (raw - norm_mean) / norm_std

    trainLen = 2300
    valLen = 286
    testLen = 286
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = []
    test_out = []
    for i in range(NUM_TEST):
        test_in.append(data[trainLen + valLen + (i*testLen) : trainLen + valLen + testLen + (i*testLen)])
        test_out.append(data[trainLen + valLen + 1 + (i*testLen) : trainLen + valLen + testLen + 1 + (i*testLen)])
    warmup_in, warmup_out = _contiguous_warmup(val_in, val_out, test_in, test_out)
    return train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Santafe laser dataset
def getDataLaser():
    raw = pd.read_csv("./data/santafelaser.csv")
    raw = np.array(raw)
    raw = raw.reshape((raw.shape[0], 1))

    # Compute normalization stats from the original 2801 window
    orig_window = raw[:2801, :]
    norm_mean = orig_window.mean()
    norm_std = orig_window.std()

    # Normalize the full raw data with the original window's stats
    data = (raw - norm_mean) / norm_std

    trainLen = 2300
    valLen = 100
    testLen = 100
    warmupLen = 100
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]

    # Non-contiguous test sets chosen to avoid collapse/regime-transition
    # regions while maintaining distributional similarity to training data.
    test_starts = [3043, 3777, 5050, 8025, 8495]

    test_in = []
    test_out = []
    warmup_in = []
    warmup_out = []
    for t in test_starts:
        test_in.append(data[t : t + testLen])
        test_out.append(data[t + 1 : t + testLen + 1])
        warmup_in.append(data[t - warmupLen : t])
        warmup_out.append(data[t - warmupLen + 1 : t + 1])
    return train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Neutral Normed DDE dataset
def getDataDDE():
    data = np.load("./data/Neutral_normed_2801.npy")
    ext = np.load("./data/Neutral_normed_extended.npy")

    trainLen = 2300
    valLen = 500
    testLen = 500
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = []
    test_out = []
    for i in range(NUM_TEST):
        test_in.append(ext[trainLen + valLen + (i*testLen) : trainLen + valLen + testLen + (i*testLen)])
        test_out.append(ext[trainLen + valLen + 1 + (i*testLen) : trainLen + valLen + testLen + 1 + (i*testLen)])
    warmup_in, warmup_out = _contiguous_warmup(val_in, val_out, test_in, test_out)
    return train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out


# https://www.sciencedirect.com/science/article/pii/S0925231222014291
# Parameterizing echo state networks for multi-step time series prediction
# Lorenz dataset
def getDataLorenz():
    data = np.load("./data/Lorenz_normed_2801.npy")
    ext = np.load("./data/Lorenz_normed_extended.npy")

    trainLen = 2300
    valLen = 444
    testLen = 444
    train_in = data[0:trainLen]
    train_out = data[0 + 1 : trainLen + 1]
    val_in = data[trainLen : trainLen + valLen]
    val_out = data[trainLen + 1 : trainLen + valLen + 1]
    test_in = []
    test_out = []
    for i in range(NUM_TEST):
        test_in.append(ext[trainLen + valLen + (i*testLen) : trainLen + valLen + testLen + (i*testLen)])
        test_out.append(ext[trainLen + valLen + 1 + (i*testLen) : trainLen + valLen + testLen + 1 + (i*testLen)])
    warmup_in, warmup_out = _contiguous_warmup(val_in, val_out, test_in, test_out)
    return train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out


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
