import math
import Evaluation
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from Utils import *
from Models import *
from bayes_opt import BayesianOptimization
import torch
import time
import traceback
import json
import pickle
from sklearn.metrics import mean_squared_error

files = ['s','j','na','l','ni']
totalGestureNames = ['left','right','forward','backward','bounce up','bounce down','turn left','turn right','shake lr','shake ud', \
                     'tap 1','tap 2','tap 3','tap 4','tap 5','tap 6','no gesture']
gestureNames = []
usedGestures = [0,1,2,3,4,5,6,7,8,9]
for i in usedGestures:
    gestureNames.append(totalGestureNames[i])
gestureNames.append('no gesture')
learnTreshold = False
nFolds = 5

def fix_seed(manualSeed):
    
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    
def createData(inputFiles, testFiles):
    trainset = UniHHIMUGestures(dataDir='dataSets/', 
                                train=True, 
                                inputFiles=inputFiles,
                                testFiles=testFiles,
                                useNormalized=2, 
                                learnTreshold=learnTreshold,
                                shuffle=True,
                                nFolds=len(inputFiles)
                               )

    testset = UniHHIMUGestures(dataDir='dataSets/', 
                               train=False, 
                               inputFiles=inputFiles,
                               testFiles=testFiles,
                               useNormalized=2, 
                               learnTreshold=learnTreshold,
                               shuffle=True,
                               nFolds=len(inputFiles)                              
                              )

    trainloader = DataLoader(trainset, batch_size=1,
                            shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=1,
                            shuffle=True, num_workers=1)
    return trainset, testset, trainloader, testloader

def getData(loader, fraction):
    x = []
    y = []
    for inputs, targets in loader:
        x.append(inputs[0][:int(inputs[0].shape[0]*fraction)])
        y.append(targets[0][:int(targets[0].shape[0]*fraction)])
    x = np.concatenate(x)
    y = np.concatenate(y)
    return (x, y)

def trainESN(trainloader, esn, fraction=1):
    x, y = getData(trainloader, fraction)
    return esn.fit(x, y, warmup=100)

def trainESN2(trainloader, esn, fraction=1):
    x, y = getData(trainloader, fraction)
    return esn.train(x, y)

def trainESN3(trainloader, esn, fold, isVal):
    x, y = getData(trainloader, 1)
    size = x.shape[0]
    testIndices = list(range(int(size*fold/nFolds), int((size+1)*fold/nFolds)))
    mask = np.ones(x.shape[0], dtype=bool)
    mask[testIndices] = False
    x = x[mask]
    y = y[mask]
    if isVal:
        x = x[:int(x.shape[0]*(nFolds-2)/(nFolds-1))]
        y = y[:int(y.shape[0]*(nFolds-2)/(nFolds-1))]
    return esn.fit(x, y, warmup=100)

def trainESN4(trainloader, esn, fold, isVal):
    x, y = getData(trainloader, 1)
    size = x.shape[0]
    testIndices = list(range(int(size*fold/nFolds), int((size+1)*fold/nFolds)))
    mask = np.ones(x.shape[0], dtype=bool)
    mask[testIndices] = False
    x = x[mask]
    y = y[mask]
    if isVal:
        x = x[:int(x.shape[0]*(nFolds-2)/(nFolds-1))]
        y = y[:int(y.shape[0]*(nFolds-2)/(nFolds-1))]
    return esn.train(x, y)

def testESN(esn, testFiles, startFraction=0, stopFraction=1, fixed_threshold=0.4):
    testF1MaxApps = []
    testAccuracies = []
    errors = []
    
    _, _, trainloader, testloader = createData(inputFiles=testFiles, testFiles=testFiles)

    for test_inputs, test_targets in trainloader:
        inputs = test_inputs[0]
        targets = test_targets[0]
        inputs = inputs[int(inputs.shape[0]*startFraction):int(inputs.shape[0]*stopFraction)].numpy()
        targets = targets[int(targets.shape[0]*startFraction):int(targets.shape[0]*stopFraction)].numpy()
        prediction = esn.run(inputs)
        errors.append(mean_squared_error(targets, prediction))

        t_target = targets
        threshold = np.ones((prediction.shape[0],1))*fixed_threshold

        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction,t_target,threshold, 10)

        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, t_target, 0.5)
        testF1MaxApps.append(np.mean(sklearn.metrics.f1_score(targ_MaxApp,pred_MaxApp,average=None)))
        testAccuracies.append(np.mean(sklearn.metrics.accuracy_score(targ_MaxApp,pred_MaxApp)))
    return testF1MaxApps, testAccuracies, errors, targ_MaxApp, pred_MaxApp

def testModel(params, trainFiles, testFiles, trainFraction=1, startFraction=0, stopFraction=1, numEvals=1, modelCreator=createESN, trainFunc=trainESN):
    scores = []
    accuracies = []
    errors = []
    targets = np.array([])
    preds = np.array([])
    times = []
    for _ in range(numEvals):
        _, _, trainloader, _ = createData(inputFiles=trainFiles, testFiles=testFiles)
        esn = modelCreator(**params)
        for i in range(5):
            try:
                start = time.time()
                trainFunc(trainloader, esn, trainFraction)
                times.append(time.time()-start)
                score, accuracy, mse, target, pred = testESN(esn, testFiles, startFraction, stopFraction)
                errors.append(mse)
                targets = np.append(targets, target)
                preds = np.append(preds, pred)
                scores.extend(score)
                accuracies.extend(accuracy)
                break
            except Exception as e:
                print(traceback.format_exc())
                continue
    if (len(scores)==0):
        return [0], [0], [10000000], [], [], []
    return (scores, accuracies, errors, targets, preds, times)

def testModel2(params, trainFiles, testFiles, fold, isVal, numEvals=1, modelCreator=createESN, trainFunc=trainESN3):
    scores = []
    accuracies = []
    errors = []
    targets = np.array([])
    preds = np.array([])
    times = []
    for _ in range(numEvals):
        for i in range(5):
            try:
                _, _, trainloader, _ = createData(inputFiles=trainFiles, testFiles=testFiles)
                esn = modelCreator(**params)
                start = time.time()
                trainFunc(trainloader, esn, fold, isVal)
                times.append(time.time()-start)
                score, accuracy, mse, target, pred = testESN(esn, testFiles, fold/nFolds, (fold+1)/nFolds)
                errors.append(mse)
                targets = np.append(targets, target)
                preds = np.append(preds, pred)
                scores.extend(score)
                accuracies.extend(accuracy)
                break
            except Exception as e:
                print(traceback.format_exc())
                continue
    if (len(scores)==0):
        return [0], [0], [], [], [], []
    return (scores, accuracies, errors, targets, preds, times)

def optimizer_global1(pbounds, modelCreator=createESN, trainFunc=trainESN, numEvals=3):
    optimalParams = {}
    f1Scores = []
    accuracies = []
    targets = np.array([])
    preds = np.array([])
    times = []
    for idx in range(5):
        inputFiles = files[:idx] + files[idx+1:]
        validationFiles = [inputFiles[idx%4]]
        trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]
        testFiles = files[idx:idx+1]
        print(inputFiles, trainFiles, validationFiles, testFiles)
        
        def black_box_function(**params):
            try:
                scores1, _, _, _, _, _ = testModel(params, trainFiles, validationFiles, 1, 0, 1, numEvals, modelCreator, trainFunc)
                f1 = np.array(scores1).mean()
                return f1
            except:
                print(traceback.format_exc())
                return -0.01

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
        )
        
        optimizer.maximize(
            init_points=15,
            n_iter=15,
        )
        
        s, a, _, target, pred, trainingTimes = testModel(optimizer.max['params'], inputFiles, testFiles, 1, 0, 1, 10, modelCreator, trainFunc)
        optimalParams[testFiles[0]]=optimizer.max['params']
        print(testFiles, np.array(s).mean(), np.array(s).std(), np.array(a).mean(), np.array(a).std())
        f1Scores+=s
        accuracies+=a
        targets = np.append(targets, target)
        preds = np.append(preds, pred)
        times.extend(trainingTimes)
    print(np.array(f1Scores).mean(), np.array(f1Scores).std(), np.array(accuracies).mean(), np.array(accuracies).std())
    print("Training time:", sum(times)/len(times), statistics.pstdev(times))
    makeConfusionMatrix(targets, preds)
    return optimalParams

def optimizer_user(pbounds, modelCreator, trainFunc, numEvals=3):
    files = ['s', 'j', 'na','l','ni']
    optimalParams = {}
    f1Scores = {}
    accuracies = {}
    times = []
    targets = np.array([])
    preds = np.array([])
    for eachFile in files:
        f1Scores[eachFile] = []
        accuracies[eachFile] = []
    for idx in range(5):
        trainFiles = [files[idx]]
        testFiles = [files[idx]]
        
        for i in range(nFolds):
            json.dump({'found': False}, open("temp.json", "w"))
            def black_box_function(**params):
                try:
                    data = json.load(open('temp.json'))
                    found = data['found']
                    if found: 
                        return -0.01
                    scores, _, _, _, _, _ = testModel2(params, trainFiles, testFiles, i, True, numEvals, modelCreator, trainFunc)
                    f1 = np.array(scores).mean()
                    if f1>0.999:
                        json.dump({'found': True}, open("temp.json", "w"))
                    return f1
                except:
                    print(traceback.format_exc())
                    return -0.01

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
            )

            optimizer.maximize(
                init_points=15,
                n_iter=15,
            )
            s, a, _, target, pred, trainingTimes = testModel2(optimizer.max['params'], trainFiles, testFiles, i, False, 10, modelCreator, trainFunc)
            targets = np.append(targets, target)
            preds = np.append(preds, pred)
            print(np.array(s).mean(), np.array(s).std())
            optimalParams[testFiles[0]]=optimizer.max['params']
            f1Scores[testFiles[0]].extend(s)
            accuracies[testFiles[0]].extend(a)
            times.extend(trainingTimes)
        print(np.array(f1Scores[testFiles[0]]).mean(), np.array(f1Scores[testFiles[0]]).std())
    for key in list(f1Scores.keys()):
        scores = f1Scores[key]
        acc = accuracies[key]
        print("{} f1 score: {} ({}), accuracy: {}, ({})".format(key, np.array(scores).mean(), np.array(scores).std(), np.array(acc).mean(), np.array(acc).std()))
    print(np.concatenate(list(f1Scores.values())).mean(), np.concatenate(list(f1Scores.values())).std(), np.concatenate(list(accuracies.values())).mean(), np.concatenate(list(accuracies.values())).std())
    print("Training time:", sum(times)/len(times), statistics.pstdev(times))
    makeConfusionMatrix(targets, preds)
    return optimalParams

def optimizer_global2(pbounds, modelCreator, trainFunc, numEvals=3):
    files = ['s','j','na','l','ni']
    optimalParams = {}
    f1Scores = {}
    accuracies = {}
    times = []
    for eachFile in files:
        f1Scores[eachFile] = []
        accuracies[eachFile] = []

    for i in range(nFolds):
        def black_box_function(**params):
            try:
                scores, _, _, _, _, _ = testModel2(params, files, files, i, True, numEvals, modelCreator, trainFunc)
                f1 = np.array(scores).mean()
                return f1
            except:
                print(traceback.format_exc())
                return -0.01

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
        )

        optimizer.maximize(
            init_points=15,
            n_iter=15,
        )
        for eachFile in files:
            subjectScores, subjectAccuracies, _, target, pred, trainingTimes = testModel2(optimizer.max['params'], files, [eachFile], i, False, 10, modelCreator, trainFunc)
            f1Scores[eachFile].extend(subjectScores)
            accuracies[eachFile].extend(subjectAccuracies)
            times.extend(trainingTimes)
        for key in list(f1Scores.keys()):
            scores = f1Scores[key]
            acc = accuracies[key]
            print(np.array(scores).mean(), np.array(scores).std(), np.array(acc).mean(), np.array(acc).std())
    for key in list(f1Scores.keys()):
        scores = f1Scores[key]
        acc = accuracies[key]
        print("{} f1 score: {} ({}), accuracy: {}, ({})".format(key, np.array(scores).mean(), np.array(scores).std(), np.array(acc).mean(), np.array(acc).std()))
    print(np.concatenate(list(f1Scores.values())).mean(), np.concatenate(list(f1Scores.values())).std(), np.concatenate(list(accuracies.values())).mean(), np.concatenate(list(accuracies.values())).std())
    print("Training time:", sum(times)/len(times), statistics.pstdev(times))
    return optimalParams

def optimizer(pbounds, modelCreator, trainAndTestModel, confusionMatrix=False, behaviorSpace=False, trainingTime=False, numEvals=3):
    optimalParams = {}
    f1Scores = []
    accuracies = []
    targets = np.array([])
    preds = np.array([])
    p = []
    times = []
    if modelCreator==createIPESN:
        reservoirCreator = createIPReservoir
    else:
        reservoirCreator = createReservoir
    for idx in range(5):
        testFiles = files[idx:idx+1]

        def black_box_function(**params):
            scores, _, _, _, _ = trainAndTestModel(params, idx, numEvals, True)
            f1 = np.array(scores).mean()
            return f1

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
        )

        optimizer.maximize(
            init_points=15,
            n_iter=15,
        )
        s, a, target, pred, userTimes = trainAndTestModel(optimizer.max['params'], idx, 20, False)
        optimalParams[testFiles[0]]=optimizer.max['params']
        for score, params in zip(optimizer._space._target, optimizer._space._params):
            params = optimizer._space.array_to_params(params)
            del params['ridge']
            params['units'] = int(params['units'])
            p.append({'params':params, 'score': score, 'creator':reservoirCreator, 'testSet': testFiles})
        print(s, a)
        print(testFiles, np.array(s).mean(), np.array(s).std(), np.array(a).mean(), np.array(a).std())
        f1Scores+=s
        accuracies+=a
        targets = np.append(targets, target)
        preds = np.append(preds, pred)
        times.extend(userTimes)
    print(np.array(f1Scores).mean(), np.array(f1Scores).std(), np.array(accuracies).mean(), np.array(accuracies).std())
    if confusionMatrix:
        makeConfusionMatrix(targets, preds)
    if trainingTime:
        print("Training time:", sum(times)/len(times), statistics.pstdev(times))
    if behaviorSpace:
        x, y, c, z = getBehaviourSpace(p)
        makeGraph(x, y, c, z)
    return optimalParams

# def optimizer_global1(pbounds, modelCreator, trainFunc, confusionMatrix=False, behaviorSpace=False, trainingTime=False, numEvals=3):
#     def trainAndTestModel(params, idx, numEvals, isVal):
#         if not isVal:
#             trainFiles = files[:idx] + files[idx+1:]
#             testFiles = files[idx:idx+1]
#         else:
#             inputFiles = files[:idx] + files[idx+1:]
#             trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]
#             testFiles = [inputFiles[idx%4]]
#         scores = []
#         accuracies = []
#         targets = np.array([])
#         preds = np.array([])
#         times = []
#         for i in range(numEvals):
#             _, _, trainloader, testloader = createData(inputFiles=trainFiles, testFiles=testFiles)
#             x, y = getData(trainloader)
#             esn = modelCreator(**params)
#             try:
#                 start = time.time()
#                 trainFunc(esn, x, y)
#                 times.append(time.time()-start)
#             except Exception as e:
#                 print(e)
#                 return 0, 0, [0], [0]
#             score, accuracy, target, pred = testESN(esn, getData(testloader), learnTreshold)
#             targets = np.append(targets, target)
#             preds = np.append(preds, pred)
#             scores.append(score)
#             accuracies.append(accuracy)
#         return (scores, accuracies, targets, preds, times)
    
#     optimizer(pbounds, modelCreator, trainAndTestModel, confusionMatrix, behaviorSpace, trainingTime, numEvals)

# def optimizer_user(pbounds, modelCreator, trainFunc, confusionMatrix=True, behaviorSpace=False, trainingTime=True, trainFraction=0.6, valFraction=0.2, numEvals=3):
#     def trainAndTestModel(params, idx, numEvals, isVal):
#         testFile = [files[idx]]
#         scores = []
#         accuracies = []
#         targets = np.array([])
#         preds = np.array([])
#         times = []
#         for i in range(numEvals):
#             _, _, _, testloader = createData(inputFiles=testFile, testFiles=testFile)
#             x, y = getData(testloader)
#             if isVal:
#                 fraction = trainFraction
#                 stopFraction = trainFraction+valFraction
#             else:
#                 fraction = trainFraction+valFraction
#                 stopFraction = 1
#             train_x = x[:int(x.shape[0]*fraction)]
#             train_y = y[:int(y.shape[0]*fraction)]
#             test_x = x[int(x.shape[0]*fraction):int(x.shape[0]*stopFraction)]
#             test_y = y[int(y.shape[0]*fraction):int(y.shape[0]*stopFraction)]
                
#             esn = modelCreator(**params)
#             try:
#                 start = time.time()
#                 trainFunc(esn, train_x, train_y)
#                 times.append(time.time()-start)
#             except Exception as e:
#                 print(e)
#                 return 0, 0, [0], [0], [0]
#             score, accuracy, target, pred = testESN(esn, (test_x, test_y), learnTreshold)
#             targets = np.append(targets, target)
#             preds = np.append(preds, pred)
#             scores.append(score)
#             accuracies.append(accuracy)
#         return (scores, accuracies, targets, preds, times)
    
#     optimizer(pbounds, modelCreator, trainAndTestModel, confusionMatrix, behaviorSpace, trainingTime, numEvals)

# def optimizer_global2(pbounds, modelCreator, trainFunc, confusionMatrix=False, behaviorSpace=False, trainingTime=False, trainFraction=0.6, valFraction=0.2, numEvals=3):
#     def getData(isVal):
#         x_train = []
#         y_train = []
#         x_test = []
#         y_test = []
#         if isVal:
#             fraction = trainFraction
#             stopFraction = trainFraction+valFraction
#         else:
#             fraction = trainFraction+valFraction
#             stopFraction = 1
#         for file in files:
#             _, _, trainLoader, testloader = createData(inputFiles=[file], testFiles=[file])
#             for inputs, targets in testloader:
#                 inputs = inputs[0]
#                 targets = targets[0]
#                 print(inputs.shape, int(inputs.shape[0]*fraction), int(inputs.shape[0]*stopFraction))
#                 x_train.append(inputs[:int(inputs.shape[0]*fraction)])
#                 y_train.append(targets[:int(targets.shape[0]*fraction)])
#                 x_test.append(inputs[int(inputs.shape[0]*fraction):int(inputs.shape[0]*stopFraction)].numpy())
#                 y_test.append(targets[int(targets.shape[0]*fraction):int(targets.shape[0]*stopFraction)].numpy())
#         x_train = np.concatenate(x_train)
#         y_train = np.concatenate(y_train)
#         return (x_train, y_train, x_test, y_test)
    
#     def trainAndTestModel(params, numEvals, isVal):
#         scores = {}
#         accuracies = {}
#         for eachFile in files:
#             scores[eachFile] = []
#             accuracies[eachFile] = []
#         targets = np.array([])
#         preds = np.array([])
#         times = []
#         for i in range(numEvals):
#             train_x, train_y, test_x, test_y = getData(isVal)
#             esn = modelCreator(**params)
#             try:
#                 start = time.time()
#                 trainFunc(esn, train_x, train_y)
#                 times.append(time.time()-start)
#             except Exception as e:
#                 print(e)
#                 return 0, 0, [0], [0]
#             for i, eachFile in enumerate(files):
#                 score, accuracy, target, pred = testESN(esn, (test_x[i], test_y[i]), learnTreshold)
#                 scores[eachFile].append(score)
#                 accuracies[eachFile].append(accuracy)
#                 targets = np.append(targets, target)
#                 preds = np.append(preds, pred)
#         return (scores, accuracies, targets, preds, times)
    
#     p = []
#     if modelCreator==createIPESN:
#         reservoirCreator = createIPReservoir
#     else:
#         reservoirCreator = createReservoir

#     def black_box_function(**params):
#         scores, _, _, _, _ = trainAndTestModel(params, numEvals, True)
#         f1 = np.array(list(scores.values())).mean()
#         return f1

#     optimizer = BayesianOptimization(
#         f=black_box_function,
#         pbounds=pbounds,
#     )

#     optimizer.maximize(
#         init_points=1,
#         n_iter=1,
#     )
#     s, a, targets, preds, userTimes = trainAndTestModel(optimizer.max['params'], 10, False)
#     for score, params in zip(optimizer._space._target, optimizer._space._params):
#         params = optimizer._space.array_to_params(params)
#         del params['ridge']
#         params['units'] = int(params['units'])
#         p.append({'params':params, 'score': score, 'creator':reservoirCreator, 'testSet': files})
#     for key in s.keys():
#         scores = s[key]
#         accuracies = a[key]
#         print(key, ": ", np.mean(scores), np.std(scores), np.mean(accuracies), np.std(accuracies))
#     print(s, a)
#     if confusionMatrix:
#         makeConfusionMatrix(targets, preds)
#     if trainingTime:
#         print("Training time:", sum(userTimes)/len(userTimes), statistics.pstdev(userTimes))
#     if behaviorSpace:
#         x, y, c, z = getBehaviourSpace(p)
#         makeGraph(x, y, c, z)

def optimizer_global1_bs(pbounds, modelCreator=createESN, trainFunc=trainESN, numEvals=3):
    files = ['j','s','na','l','ni']
    optimalParams = {}
    f1Scores = []
    accuracies = []
    targets = np.array([])
    preds = np.array([])
    times = []
    p = []
    if modelCreator==createIPESN:
        reservoirCreator = createIPReservoir
    else:
        reservoirCreator = createReservoir
    for idx in range(5):
        inputFiles = files[:idx] + files[idx+1:]
        validationFiles = [inputFiles[idx%4]]
        trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]
        testFiles = files[idx:idx+1]
        print(inputFiles, trainFiles, validationFiles, testFiles)

        def black_box_function(**params):
            try:
                _, _, e, _, _, _ = testModel(params, trainFiles, validationFiles, 1, 0, 1, numEvals, modelCreator, trainFunc)
                error = np.array(e).mean()
                return 1/error
            except:
                print(traceback.format_exc())
                return 0

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
        )

        optimizer.maximize(
            init_points=100,
            n_iter=100,
        )
        s, a, e, target, pred, trainingTimes = testModel(optimizer.max['params'], inputFiles, testFiles, 1, 0, 1, 1, modelCreator, trainFunc)
        for score, params in zip(optimizer._space._target, optimizer._space._params):
            params = optimizer._space.array_to_params(params)
            del params['ridge']
            params['units'] = int(params['units'])
            p.append({'params':params, 'error': 1/score, 'creator':reservoirCreator, 'testSet': testFiles})
        optimalParams[testFiles[0]]=optimizer.max['params']
        print(testFiles, np.array(s).mean(), np.array(s).std(), np.array(a).mean(), np.array(a).std())
        f1Scores+=s
        accuracies+=a
        targets = np.append(targets, target)
        preds = np.append(preds, pred)
        times.extend(trainingTimes)
    print(np.array(f1Scores).mean(), np.array(f1Scores).std(), np.array(accuracies).mean(), np.array(accuracies).std())
    print("Training time:", sum(times)/len(times), statistics.pstdev(times))
    # makeConfusionMatrix(targets, preds)
    file = open('logs/modelParams_{}'.format(math.floor(time.time())), 'wb')
    pickle.dump(p, file)
    file.close()
    x, y, c, z = getBehaviourSpace(p)
    makeGraph(x, y, c, z)
    return optimalParams