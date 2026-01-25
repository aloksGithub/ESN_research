import traceback
from sklearn.metrics import mean_squared_error
import numpy as np
from bayes_opt import BayesianOptimization
from Utils import *
from Models import *

#==============================================================================#

def getData(N, Lag=0):
    """
    Description:    Build a NARMA10 Dataset
    Usage:          builtNARMA10(N, Lag=0)
    Input:
        N       = number of values
        Lag     = delay of values
    Output:
        u       = input values
        y       = function values
    """

    while True:
        # generate input
        u = 0.5 * np.random.uniform(low=0.0, high=1.0, size=(N+1000))

        # generate output arrays
        y_base = np.zeros(shape=(N+1000))
        y = np.zeros(shape=(N, Lag+1))

        # calculate intermediate output
        for i in range(10, N+1000):
            # implementation of a tenth order system model
            y_base[i] = 0.3 * y_base[i-1] + 0.05 * y_base[i-1] * \
                np.sum(y_base[i-10:i]) + 1.5 * u[i-1] * u[i-10] + 0.1

        # throw away the first 1000 values (random number), since we need
        # to allow the system to "warum up"
        u = u[1000:]

        # for the default case (Lag = 0), throw away the first 1000 values as
        # well, so we have pairs of u and y values.
        # for Lag > 0,  take earlier values (because of the lag), to be more
        # precise, "value(Lag) earlier values"; do that for every value
        # from 0 to Lag
        for curr_Lag in range(0, Lag+1):
            y[:, curr_Lag] = y_base[1000 - curr_Lag : len(y_base)-curr_Lag]

        # if all values of y are finite, return them with the corresponding
        # inputs
        if np.isfinite(y).all():
            return (u, y)

        # otherwise, try again. You random numbers were "unlucky"
        else:
            pass
            
def getMultipleData(n1, n2):
    x = []
    y = []
    for i in range(n1):
        xi, yi = getData(n2)
        x.append(np.expand_dims(xi, 1))
        y.append(yi)
    return np.array(x), np.array(y)

def trainModel(x, y, params, modelCreator):
    esn = modelCreator(**params)
    esn.fit(x, y)
    return esn

def nmse(x, y):
    numerator = 0
    denominator = 0
    for index in range(len(x)):
        numerator+=np.linalg.norm(x[index]-y[index])**2
    for index in range(len(x)):
        denominator+=np.linalg.norm(y[index])**2
    return numerator/denominator

def testModel(x, y, esn):
    preds = esn.run(x)
    # print(mean_squared_error(np.squeeze(np.array(preds)), np.squeeze(np.array(y))), np.squeeze(np.array(preds)).shape)
    return nmse(np.squeeze(np.array(preds)), np.squeeze(np.array(y)))
    # return mean_squared_error(np.squeeze(np.array(preds)), np.squeeze(np.array(y)))

def optimize(pbounds, modelCreator, reservoirCreator):
    modelData = []
    
    def black_box_function(**params):
        try:
            x, y = getMultipleData(1, 2800)
            esn = trainModel(x, y, params, modelCreator)
            error = testModel(x, y, esn)
            score = 1/error
            if math.isnan(score):
                return 0.0001
            return score
        except:
            print(traceback.format_exc())
            return 0.0001

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
    )
    optimizer.maximize(
        init_points=1,
        n_iter=1,
    )
    # num = 0
    # for score, params in zip(optimizer._space._target, optimizer._space._params):
    #     num+=1
    #     print(num)
    #     params = optimizer._space.array_to_params(params)
    #     x, y = getMultipleData(200, 1000)
    #     separability = calculateSeparabilityForData(params, reservoirCreator, x)
    #     generalizability = calculateGeneralizationForData(params, reservoirCreator, x)
    #     # mc = memoryCapacity(params, reservoirCreator)
    #     modelData.append({'params':params, 'error': 1/score, 'separability': separability, 'generalizability': generalizability, 'mc': 'mc'})
    x, y = getMultipleData(1, 5600)
    esn = trainModel(x[:,:2800], y[:,:2800], optimizer.max['params'], modelCreator)
    error = testModel(x[:,2800:], y[:,2800:], esn)
    print("RESULT: ", error)
    
    return modelData, error
