"""
CPU-based similarity and entropy estimation for GE-DESN.
Adapted from LocalEntropy.py in the original repository.
"""
import numpy as np


def SMEstimater(X_train, X_num=50, T_num=3000, method=0):
    """Compute pairwise neuron similarity matrix.

    Args:
        X_train: ndarray of shape (N_neurons, T_timesteps), neuron activation states
        X_num: number of neurons (rows in X_train)
        T_num: number of timesteps (columns in X_train)
        method: similarity method (0 = inverse Euclidean distance)

    Returns:
        Upper-triangular similarity matrix of shape (N_neurons, N_neurons)
    """
    N_num, Tnum = X_train.shape
    result = np.zeros((N_num, N_num))
    if method == 0:
        for i in range(N_num):
            for j in range(i + 1, N_num):
                dis = (X_train[i, :] - X_train[j, :]) ** 2
                dif = 1 / (1 + np.sum(dis))
                result[i, j] = dif
    return result


def EntropyEstimater(X_train, X_num=50, T_num=3000):
    """Placeholder entropy estimator (CPU version returns constant).

    The GPU version computes approximate Shannon entropy via histograms.
    This CPU fallback returns a constant as in the original LocalEntropy.py.
    """
    return 0.25
