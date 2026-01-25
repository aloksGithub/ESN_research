import math
import numpy as np 
import Evaluation
import random
from DataSet import UniHHIMUGestures
from torch.utils.data import DataLoader
from reservoirpy.nodes import Ridge
import matplotlib.pyplot as plt
import time
import statistics
import sklearn
import pickle


totalGestureNames = ['left','right','forward','backward','bounce up','bounce down','turn left','turn right','shake lr','shake ud', \
                     'tap 1','tap 2','tap 3','tap 4','tap 5','tap 6','no gesture']
gestureNames = []
usedGestures = [0,1,2,3,4,5,6,7,8,9]
for i in usedGestures:
    gestureNames.append(totalGestureNames[i])
gestureNames.append('no gesture')
learnTreshold = False


def runningAverage(inputData, width):
    inputData = np.atleast_2d(inputData)
    target = np.zeros((inputData.shape))
    for i in range(width,len(inputData-width)):
            target[i,:] = np.mean(inputData[i-width:i+width,:],0)
    return target


def createData(inputFiles, testFiles, dataDir='dataSets/', learnThreshold=False):
    trainset = UniHHIMUGestures(dataDir=dataDir, 
                                train=True, 
                                inputFiles=inputFiles,
                                testFiles=testFiles,
                                useNormalized=2, 
                                learnTreshold=learnThreshold,
                                shuffle=True,
                               )

    testset = UniHHIMUGestures(dataDir=dataDir, 
                               train=False, 
                               inputFiles=inputFiles,
                               testFiles=testFiles,
                               useNormalized=2, 
                               learnTreshold=learnThreshold,
                               shuffle=True
                              
                              )

    trainloader = DataLoader(trainset, batch_size=1,
                            shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=1,
                            shuffle=True, num_workers=1)
    return trainset, testset, trainloader, testloader

def getData(loader):
    x = []
    y = []
    for inputs, targets in loader:
        x.append(inputs[0])
        y.append(targets[0])
    x = np.concatenate(x)
    y = np.concatenate(y)
    return (x, y)



# Behavoir Space Utilities
def rollingWindow(data, windowSize=50):
    rollingData = []
    for i in range(windowSize, data.shape[0]):
        rollingData.append(data[i-windowSize:i])
    return np.array(rollingData[:len(rollingData)//2])

def sampleCollector(x, y):
    samplesX = [[]]
    samplesY = [[]]
    prevLabel = np.array([0]*10)
    for index, sample in enumerate(y):
        if (sample==prevLabel).all():
            samplesX[-1].append(x[index])
            samplesY[-1].append(sample)
        else:
            prevLabel = sample
            samplesX.append([x[index]])
            samplesY.append([sample])
    return samplesX

def calculateSeparability(params, creator, testSet):
    reservoir = creator(**params)
    trainset, testset, trainloader, testloader = createData(inputFiles=testSet, testFiles=['ni'])
    x, y = getData(trainloader)
    x2 = sampleCollector(x, y)
    matrix = np.ndarray((reservoir.units, len(x2)))
    for i in range(len(x2)):
        pred = reservoir.run(np.array(x2[i]))[-1]
        matrix[:, i] = pred
    return np.linalg.matrix_rank(np.matrix.transpose(matrix), tol=1)

def calculateSeparabilityForData(params, creator, x):
    reservoir = creator(**params)
    matrix = np.ndarray((reservoir.units, len(x)))
    for i in range(len(x)):
        pred = reservoir.run(np.array(x[i]))[-1]
        matrix[:, i] = pred
    return np.linalg.matrix_rank(matrix)

def calculateGeneralization(params, creator, testSet, noiseLevel=0.4):
    reservoir = creator(**params)
    trainset, testset, trainloader, testloader = createData(inputFiles=testSet, testFiles=['ni'])
    x, y = getData(trainloader)
    x2 = sampleCollector(x, y)
    matrix = np.ndarray((reservoir.units, len(x2)))
    for i in range(len(x2)):
        input = np.array(x2[i])
        noise = np.random.rand(*input.shape)*2*noiseLevel-noiseLevel
        pred = reservoir.run(input+noise)[-1]
        matrix[:, i] = pred
    return np.linalg.matrix_rank(np.matrix.transpose(matrix), tol=1)

def calculateGeneralizationForData(params, creator, x):
    reservoir = creator(**params)
    matrix = np.ndarray((reservoir.units, len(x)))
    for i in range(len(x)):
        input = np.array(x[i])
        noise = np.random.rand(*input.shape)*0.2-0.15
        pred = reservoir.run(input+noise)[-1]
        matrix[:, i] = pred
    return np.linalg.matrix_rank(matrix)

def memoryCapacity(params, creator, numInputs=1, trainSize = 1000):
    maxDelay = 30
    capacity = 0
    for i in range(maxDelay):
        randomSeries = np.random.rand(trainSize+i, numInputs)
        targets = randomSeries[0:trainSize]
        inputs = randomSeries[i:]
        reservoir = creator(**params)
        readout = Ridge(output_dim=1)
        esn = reservoir>>readout
        try:
            esn.fit(inputs, targets)
        except:
            pass
        newRandomSeries = np.random.rand(trainSize+i, numInputs)
        preds = np.squeeze(esn.run(newRandomSeries)[i:])
        cov = abs(np.cov(np.squeeze(newRandomSeries[0:trainSize]), preds)[0, -1])
        v1 = np.var(np.squeeze(newRandomSeries[0:trainSize]))
        v2 = np.var(preds)
        capacity+=(np.corrcoef(np.squeeze(newRandomSeries[0:trainSize]), preds)[0, -1])**2
        # capacity+=cov*cov/(v1*v2)
    return capacity

def getBehaviourSpace(reservoirs):
    separability = []
    generalizability = []
    mc = []
    errors = []
    for reservoirParams in reservoirs:
        error = reservoirParams['error']
        errors.append(error)
        creator = reservoirParams['creator']
        separability.append(calculateSeparability(reservoirParams['params'], creator, reservoirParams['testSet']))
        generalizability.append(calculateGeneralization(reservoirParams['params'], creator, reservoirParams['testSet']))
        mc.append(memoryCapacity(reservoirParams['params'], creator))
    file = open('logs/behaviorSpace_{}'.format(math.floor(time.time())), 'wb')
    pickle.dump({"separability": separability, "generalizability": generalizability, "mc": mc, "errors": errors}, file)
    file.close()
    return separability, generalizability, mc, errors

def makeGraph(x, y, z, c):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=0.5)
    ax.set_xlabel('$KR$')
    ax.set_ylabel('$GR$')
    ax.set_zlabel('$MC$')
    fig.colorbar(p,ax=[ax],location='left')
    
def measureTrainingTime(model, trainFunction, params, numEvals=3):
    times = []
    for i in range(numEvals):
        for testSet in params.keys():
            start = time.time()
            files = ['s','j','na','l','ni']
            files.remove(testSet)
            trainset, testset, trainloader, testloader = createData(inputFiles=files, testFiles=[testSet])
            esn = model(**params[testSet])
            trainFunction(trainloader, esn)
            times.append(time.time()-start)
    return (sum(times)/15, statistics.pstdev(times))

def makeConfusionMatrix(targets, preds):
    cm = sklearn.metrics.confusion_matrix(targets, preds)
    Evaluation.plot_confusion_matrix(cm, gestureNames, title="")
    plt.tight_layout()
    plt.ylim(10.5,-0.5)