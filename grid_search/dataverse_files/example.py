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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import ESN_Torch as ESN
import tracemalloc
from scipy.stats import uniform
from sklearn.metrics import r2_score

def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""

    # Set the seed
    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print( "!!! WARNING !!!: Seed was not set correctly.")
        print( "!!! Seed that we tried to use: "+str(seed))
        print( "!!! Error message: "+str(e))
        seed = None
    #print( "Seed used for random values:", seed)
    return seed

'''
Global Stuff for the Network and Attractor
'''
stds = np.zeros(1)
means = np.zeros(1)
np.random.seed(0)
dim = 1
WinInit = np.random.rand(8192*2,dim+1) - 0.5
maskWin = np.random.rand(8192*2,WinInit.shape[1])
maskW = np.random.rand(8192*2,8192*2) #create a mask Uniform[0;1]



seedSet = 2371135578#2000#42#2371135578#7 #None #42 default; 50 Vgl. Lit.

#set_seed(seed) #random.seed(seed)

data = np.zeros((1,40000))

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

def so(a, spar):
    idx = np.flatnonzero(a)
    Nso = np.count_nonzero(a!=0) - int(round(spar*a.size))
    np.put(a,np.random.choice(idx,size=Nso,replace=False),0)
    return a

def network(params):
    #print(params)
    initLen = int(params[8])
    trainLen = int(params[7]) + initLen
    testLen = 500#150



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
    Win = WinPooling(WinInit,int(8192*2/N))
    maskWinuse = WinPooling(maskWin,int(8192*2/N))

    
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
        mse = 1e121
        maxStep = -1
        return (testLen-maxSteps)
    train_in = data[0:trainLen]
    train_out = data[0+1:trainLen+1]
    test_in = data[trainLen:trainLen+testLen]
    test_out = data[trainLen+1:trainLen+testLen+1]

    internal_trained, Wout = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=initLen, verbose=False)
    ###
    #for testing multiple beta only, import ESN_TorchBeta and use:
    #
    #Wout = ESN.WoutCalc(regularization_coef)
    ###
    output_pred, internal_pred = reservoir.run(inputs=[test_in,], reset_state=False)
    errorLen = len(test_out[:]) #testLen #2000

    addition = params[10]
    value = params[11]
    np.save('MG2021/%sWout%s_with%s_at%s.npy' % (N, addition, value, seed), Wout)
    #np.save('MG/train/intTrain%s_with%s_at%s.npy' % (addition, value, seed), internal_trained[0])
    np.save('MG2021/pred/%sintPred%s_with%s_at%s.npy' % (N, addition, value, seed), internal_pred[0])
    #np.save('MG/pred/intPtred%s_with%s_at%s.npy' % (addition, value, seed), internal_pred[0])
    internal = np.dot(Wout, np.vstack((np.ones((1,trainLen-initLen)), internal_trained[0].T)))
    #print('internal: ', internal.shape)
    intMAE = np.mean(np.abs(internal.T-train_out[initLen:,:]))

    mse = np.mean((test_out[:] - output_pred[0])**2) 
    r2 = r2_score(test_out[:], output_pred[0])

    mae = np.mean((np.abs((test_out[:]-output_pred[0]))))
    return [mse, r2, intMAE]


if __name__ == '__main__':
    tracemalloc.start()
    dim = 1
    data = np.load('Samples/MG17.npy')
    data = data.reshape((data.shape[0],1))
    data = data[:2801,:]
    from scipy import stats
    data = stats.zscore(data)
    print( "data dimensions", data.shape)
    
    #hyperparameters
    gamma = 1

    rho = .995

    N_n = 256#1024

    sparW = 1
    sparWin = 1

    beta = 1e-6

    sigma = 1

    S_I = 300
    S_T = 2000

    params = [100, 1, .9, 1, 1, 1e-6, 1, 2000, 300, 128, 10]
    
    cluster = 15
    seedNum = 100
    ret = np.zeros((3, cluster+1,seedNum))

    #Read 100 fixed seeds for reproducibility
    seedSet = np.genfromtxt('seeds.csv')
    
    
    ####
    # N_n tested for 256, ..., 2048
    ####
    N_n = 2048
    
    ####
    # Select Hyper-parameter
    ####

    spar = np.linspace(0.1,1,10)
    #N_n = [256, 512, 1024, 2048]
    #rho= np.asarray([0, .25, .5, .75, 1, 1.25, 1.5])
    #gamma = np.asarray([0, .25, .5, .75, 1])
    #beta = np.logspace(-10,.3,10)
    cluster = spar.shape[0]
    ret = np.zeros((3, cluster, seedNum))

    for j in range(cluster):
        for i in range(seedNum):
        #print(i)
        #seedSet = bad_seeds[i]
            print(i, ' ', seedSet[i])
            np.random.seed(int(seedSet[i]))

            params = [N_n, gamma, rho, spar[j], sparWin, beta, sigma, S_T, S_I, seedSet[i], 'sparW', spar[j]]

            ret[:,j, i] = network(params)
        np.save('MG2021/sparW_2048_over_seeds%s.npy' % (spar[j]), ret[:,j,:])
    means = np.zeros((cluster))
    for i in range(cluster):
        means[i] = np.median(ret[1,i,:])
    print('Means: ', means)
    np.save('MG2021/2048_sparW.npy', ret[0])
    np.save('MG2021/means/2048_sparW.npy', means)

    sparW = spar[np.argmax(means)]
    
    
