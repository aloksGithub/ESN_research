'''

MANUAL ITERATION OVER PARAMETERS AS DESCRIPED IN PAPER:
    "Parameterizing echo state networks for multi-step time series prediction"
https://doi.org/10.1016/j.neucom.2022.11.044
USABLE FOR \Theta = \{ $(d(W^r)+N^r)$, $\rho$, $\gamma$, $\beta$ \}


Original Code:
    @online{ReservoirPy,
	Date = {08/09/2019},
	Title = {ReservoirPy},
	Url = {https://github.com/neuronalX/reservoirpy},
	}
Commit: c18f3b62bd788d79f1ead16a20684c6531c2540e

Original author and contribution:
    @author: Xavier HINAUT
    xavier.hinaut #/at\# inria.fr
    Copyright Xavier Hinaut 2018

    "I would like to thank Mantas Lukosevicius for his code that was used as inspiration     for this code:
    http://minds.jacobs-university.de/mantas/code"

@incollection{Trouvain2020,
  doi = {10.1007/978-3-030-61616-8_40},
  url = {https://doi.org/10.1007/978-3-030-61616-8_40},
  year = {2020},
  publisher = {Springer International Publishing},
  pages = {494--505},
  author = {Nathan Trouvain and Luca Pedrelli and Thanh Trung Dinh and Xavier Hinaut},
  title = {{ReservoirPy}: An Efficient and User-Friendly Library to Design Echo State Networks},
  booktitle = {Artificial Neural Networks and Machine Learning {\textendash} {ICANN} 2020}
}

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import ESN_Torch as ESN
import tracemalloc
from scipy.stats import uniform
from sklearn.metrics import r2_score
import sys

'''
Global Stuff for the Network and Attractor
'''

def nrmse_func(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    diff = y_true - y_pred
    mean_sq_error = np.mean(np.sum(diff**2, axis=1))
    mean_norm = np.linalg.norm(np.mean(y_true, axis=0))
        
    rmse = np.sqrt(mean_sq_error)
    
    if mean_norm == 0:
        return 100000
        
    error = rmse / mean_norm
    return error
    
def r_squared(y_true, y_pred):
    try:
        return r2_score(y_true, y_pred, multioutput="variance_weighted")
    except:
        return 0


def pooling(W, pool):
    #poolingby reducing by 2**pool
    #print('Wshape: ', W.shape)
    W_orig = W
    W_new = np.zeros((int(W.shape[0]/pool), int(W.shape[1]/pool)))
    for i in range(W_new.shape[0]):
        for j in range(W_new.shape[1]):
            W_new[i,j] = np.average(W[i*(pool):(i+1)*(pool), j*(pool):(j+1)*(pool)])
    return W_new

def WinPooling(win, pool):
    #poolingby reducing by 2**pool

    W_orig = win
    W_new = np.zeros((int(win.shape[0]/pool), int(win.shape[1])))
    for i in range(W_new.shape[0]):
        for j in range(W_new.shape[1]):
            W_new[i,j] = np.average(win[i*(pool):(i+1)*(pool), j])
    return W_new

def network(params, dim, train_in, train_out, test_in, maskW):
    #print(params)
    initLen = int(params[8])

    n_inputs = dim
    input_bias = True # add a constant input to 1
    n_outputs = dim
    n_reservoir = int(params[0]) # number of recurrent units
    leak_rate = params[1] # leaking rate (=1/time_constant_of_neurons)
    spectral_radius = params[2] #1.25 # Scaling of recurrent matrix
    input_scaling = params[6] # Scaling of input matrix
    proba_non_zero_connec_W = params[3] # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
    proba_non_zero_connec_Win = params[4] # Sparsity of input matrix
    proba_non_zero_connec_Wfb = 1. # Sparsity of feedback matrix
    regularization_coef =  params[5] #None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression
    # out_func_activation = lambda x: x
    N = n_reservoir#100
    dim_inp = n_inputs #26
    #Zufallsmatrizen mit zuf. Topologie
    seed = params[9]


    np.random.seed(int(seed))
    W = np.asarray((uniform.rvs(size=(8192*2,8192*2))))
    W = pooling(W, int(8192*2/N))
    Win = np.asarray((uniform.rvs(size=(8192*2,dim+1))))
    Win = WinPooling(Win, int(8192*2/N))
    #maskWuse = pooling(maskW, int(8192*2/N))
    maskWuse = np.asarray((uniform.rvs(size=(N,N))))

    maskWuse = pooling(maskW, int(8192*2/N))
    #maskWuse = so(maskWuse, proba_non_zero_connec_W)

    idx = np.flatnonzero(maskWuse)
    Nso = np.count_nonzero(maskWuse!=0) - int(round(proba_non_zero_connec_W*maskWuse.size))
    if Nso > 0:
        np.put(maskWuse,np.random.choice(idx,size=Nso,replace=False),0)
    W[maskWuse == 0] = 0
    negMask = np.asarray((uniform.rvs(size=(8192*2,8192*2))))
    negMask = pooling(negMask, int(8192*2/N))
    W[negMask > .5] *= -1
    neginMask = np.asarray((uniform.rvs(size=(8192*2,dim+1))))
    neginMask = WinPooling(neginMask, int(8192*2/N))
    Win[neginMask > .5] *= -1
    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    #TODO: check if this operation is quicker: max(abs(linalg.eig(W)[0])) #from scipy import linalg
    # rescale them to reach the requested spectral radius:
    
    Win = Win*input_scaling
    if original_spectral_radius != 0:
        W = W * (spectral_radius / original_spectral_radius)
    reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None)
    if reservoir == 0:
        return None

    internal_trained, Wout = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=initLen, verbose=False)
    ###
    #for testing multiple beta only, import ESN_TorchBeta and use:
    #
    #Wout = ESN.WoutCalc(regularization_coef)
    ###
    output_pred, internal_pred = reservoir.run(inputs=[test_in,], reset_state=False)

    return output_pred[0]


S_T = 2000

def getDataLaser():
    global dim
    dim = 1
    laser = pd.read_csv("./Samples/santafelaser.csv")
    data = np.array(laser)
    data = data.reshape((data.shape[0],1))
    data = data[:2801,:]
    from scipy import stats
    data = stats.zscore(data)
    params = [1024, 0.41, 0.906, 0.84, 1, 8.1e-7, 1, S_T, 300, 0, 'spawW', 0.84, 100]
    return data, params

def getDataDDE():
    global dim
    dim = 6
    data = np.load('Samples/Neutral_normed_2801.npy')
    params = [256, 0.84, 0.995, 0.91, 1, 8.3e-7, 1, S_T, 300, 0, 'sparW', 0.91, 500]
    return data, params

def getDataLorenz():
    global dim
    dim = 3
    data = np.load('Samples/Lorenz_normed_2801.npy')
    params = [2048, 0.88, 0, 0, 1, 1.1e-6, 1, S_T, 300, 0, 'sparW', 0, 444]
    return data, params

def getDataMGS():
    global dim
    dim = 1
    data = np.load('Samples/MG17.npy')
    data = data.reshape((data.shape[0],1))
    data = data[:2801,:]
    from scipy import stats
    data = stats.zscore(data)
    params = [2048, 0.68, 1.406, 0.44, 1, 6e-7, 1, S_T, 300, 0, 'sparW', 0.44, 286]
    return data, params



 

if __name__ == '__main__':
    tracemalloc.start()
    datasets = [getDataDDE, getDataLorenz, getDataMGS, getDataLaser]
    data, best_params = datasets[int(sys.argv[1])]()

    #Read 100 fixed seeds for reproducibility
    seedSet = np.genfromtxt('seeds.csv')

    global WinInit, maskWin, maskW
    WinInit = np.random.rand(8192*2,dim+1) - 0.5
    maskWin = np.random.rand(8192*2,WinInit.shape[1])
    maskW = np.random.rand(8192*2,8192*2) #create a mask Uniform[0;1]

    # Sequential evaluation across seeds
    results = []
    for s in seedSet:
        p = list(best_params)
        p[9] = int(s)
        r = network(p)
        print(r[0], r[2])
        results.append(r)

    # results: list of [nrmse, mse, r2, intMAE]
    results = np.asarray(results)
    best_r2 = float(np.max(results[:,2]))
    best_nrmse = float(np.min(results[:,0]))
    med_r2 = float(np.median(results[:,2]))
    med_nrmse = float(np.median(results[:,0]))

    print(f"Best R2 over {len(results)} seeds: {best_r2}")
    print(f"Median R2: {med_r2}")
    print(f"Best NRMSE: {best_nrmse}")
    print(f"Median NRMSE: {med_nrmse}")
