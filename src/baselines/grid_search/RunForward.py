import numpy as np
import ESN_Torch_Further as ESN
from scipy.stats import uniform
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = 6
#import torch
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

add = 300
proba_non_zero_connec_W = 1
rho = .995
spectral_radius = rho
gamma = 1
dim = 6
neurons = 256
N = neurons
seedSet = np.genfromtxt('seeds.csv') 
system = 'NeutralRho'
value = 1.0
param = 'sparW'
#brekn()
def run(W, Win, gamma, seeds, dim):
    print('param')
    wout = np.load('%s/%sWout%s_with%s_at%s.npy' %(system, neurons, param, value, seeds))
    x = np.load('%s/pred/%sintPred%s_with%s_at%s.npy' % (system, neurons, param, value, seeds))
    #print(x.shape)
    test_in = np.dot(wout, np.hstack((np.ones((150,1)),x)).T  ).T
    #print(test_in.shape)
    test_in = test_in[-1,:].reshape((1,dim))
    #print(test_in.shape)
    test_in = np.zeros((300,dim))+test_in
    input_bias = True
    reservoir = ESN.ESN(lr=gamma, W=W, Win=Win, input_bias=input_bias, ridge=1e-6, Wfb=None, fbfunc=None)
    output_pred, internal_pred = reservoir.run(inputs=[test_in,], xin=x[-1,:], wout=wout,  reset_state=False)
    np.save('%s/pred/%sintPred%s_add%s_at%s.npy' % (system, neurons, param, add, seeds), internal_pred[0])
    out = output_pred[0]
    
       
    print('R2: ', r2_score(test[2451:2451+300], out))
    return r2_score(test[2451:2451+300], out)
    
def loop(dim):
    for i in range(seedSet.shape[0]):
        seed = seedSet[i]
        print('seed: ', seed)
        np.random.seed(int(seed))
        W = np.asarray((uniform.rvs(size=(8192*2,8192*2))))
        W = pooling(W, int(8192*2/N))
        Win = np.asarray((uniform.rvs(size=(8192*2,dim+1))))
        Win = WinPooling(Win, int(8192*2/N))
        maskWuse = np.asarray((uniform.rvs(size=(N,N))))

        maskWuse = pooling(maskW, int(8192*2/N))

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
        Win = Win*1#input_scaling
        if original_spectral_radius != 0:
            W = W * (spectral_radius / original_spectral_radius)


        r2[i] = run(W, Win, gamma, seed, dim)
np.random.seed(0)
maskW = np.random.rand(8192*2,8192*2)
r2 = np.zeros((100,1))


####################################
### Initialization of Parameters ###
####################################
'''
Neutral 1
'''

add = 300
proba_non_zero_connec_W = 1
rho = .995
spectral_radius = rho
gamma = 1
dim = 6
neurons = 256
N = neurons
seedSet = np.genfromtxt('seeds.csv') 
system = 'NeutralRho'
value = 1.0
param = 'sparW'
test = np.load('Samples/Neutral_normed_full.npy')
print(test.shape)

loop(dim)
print('mean: ', np.mean(r2))
print('median: ', np.median(r2))


