import numpy as np
import copy
from NAS.parallel_processing import executeParallelBatch
from NAS.utils import (
    nodeParameterRanges,
    constructModel,
    runModel,
    trainModel,
)
from bayes_opt import BayesianOptimization
from reservoirpy.observables import nrmse
import pickle
import os
import time

class ESN_BO:
    """
    Hyper parameter optimization algorithm using bayesian optimization
    on top of an architecture obtained from NAS
    """

    def __init__(
        self,
        trainX,
        trainY,
        valX,
        valY,
        n_rand,
        iterations,
        outputDim,
        seedModel,
        numEvals=3,
        n_jobs=3,
        errorMetrics = [nrmse],
        defaultErrors = [np.inf],
        minimizeFitness=True,
        timeout = 180,
        saveLocation = None,
        isAutoRegressive = False,
    ):
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.n_rand = n_rand
        self.iterations = iterations
        self.outputDim = outputDim
        self.seedModel = seedModel
        self.numEvals=numEvals
        self.n_jobs = n_jobs
        self.errorMetrics = errorMetrics
        self.defaultErrors = defaultErrors
        self.saveLocation = saveLocation if saveLocation is not None else "temp"
        self.isAutoRegressive = isAutoRegressive
        self.minimizeFitness = minimizeFitness
        self.timeout = timeout
        
        self.pbounds = {}
        self.isInt = {}
        self.defaultParams = {}
        for i, node in enumerate(seedModel['nodes'][1:]):
            for param in node['params']:
                if param=="output_dim" or param=="fb_connectivity":
                    continue
                lowerLimit = nodeParameterRanges[node['type']][param]['lower']
                upperLimit = nodeParameterRanges[node['type']][param]['upper']
                self.pbounds[str(i+1) + "_" + param] = (lowerLimit, upperLimit)
                self.isInt[str(i+1) + "_" + param] = nodeParameterRanges[node['type']][param]['intOnly']
                self.defaultParams[str(i+1) + "_" + param] = node['params'][param]
        
        # Variables to keep track of models
        self.paramsTested = []
        self.performances = []
        self.times = []
        self.bestModel = None

        # Make sure that save folder exists
        directory = os.path.dirname(self.saveLocation)
        os.makedirs(directory, exist_ok=True)

    def evaluate(self, individual):
        start = time.time()
        results = executeParallelBatch(
            self.evaluateArchitectureAutoRegressive if self.isAutoRegressive else self.evaluateArchitecture,
            [(individual,) for _ in range(self.numEvals)], self.numEvals, self.timeout * self.numEvals
        )

        valid_results = [res for res in results if res is not None]
        if not valid_results:
            errors = self.defaultErrors
            bestModel = None
        else:
            main_errors = [res[0][0] for res in valid_results]
            best_error_index = np.argmin(main_errors) if self.minimizeFitness else np.argmax(main_errors)
            errors = valid_results[best_error_index][0]
            bestModel = valid_results[best_error_index][1]

        self.paramsTested.append(individual)
        self.performances.append(errors)

        all_main_errors = [e[0] for e in self.performances]
        best_of_all_time_main_error = min(all_main_errors) if self.minimizeFitness else max(all_main_errors)
        if self.minimizeFitness and best_of_all_time_main_error >= errors[0] or not self.minimizeFitness and best_of_all_time_main_error <= errors[0]:
            self.bestModel = bestModel

        end = time.time()
        print(f"Time taken: {end - start}")
        self.times.append(end - start)
        return errors

    def evaluateArchitecture(self, individual):
        """
        Instantiate a random model using given architecture, then train and evaluate it
        on one step ahead prediction using errorMetrics on valX and valY.
        """

        try:
            model = constructModel(individual)
            model = trainModel(model, self.trainX, self.trainY)
            model_copy = copy.deepcopy(model)
            preds = runModel(model, self.valX)
            errors = [metric(self.valY, preds) for metric in self.errorMetrics]
            return errors, model_copy
        except Exception as e:
            errors = self.defaultErrors
            return errors, None

    def evaluateArchitectureAutoRegressive(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        using errorMetrics on valX and valY. Test prediction is done auto-regressively,
        the output from the current timestep is used as input for next timestep
        """
        try:
            model = constructModel(individual)
            model = trainModel(model, self.trainX, self.trainY)
            model_copy = copy.deepcopy(model)
            prevOutput = self.valX[0]
            preds = []
            for _ in range(len(self.valX)):
                pred = runModel(model, prevOutput)
                prevOutput = pred
                preds.append(pred[0])
            preds = np.array(preds)
            errors = [metric(self.valY, preds) for metric in self.errorMetrics]
            return errors, model_copy
        except:
            errors = self.defaultErrors
            return errors, None
    
    def black_box(self, **params):
        modifiedArchitecture = copy.deepcopy(self.seedModel)
        for param in params:
            nodeIndex, paramName = int(param[:1]), param[2:]
            isInt = self.isInt[param]
            paramValue = int(params[param]) if isInt else params[param]
            modifiedArchitecture["nodes"][nodeIndex]['params'][paramName] = paramValue
        
        errors = self.evaluate(modifiedArchitecture)
        return -errors[0]
    
    def run(self):
        optimizer = BayesianOptimization(
            f=self.black_box,
            pbounds=self.pbounds,
            random_state=1,
            allow_duplicate_points=True
        )
        # optimizer.probe(params=self.defaultParams)

        optimizer.maximize(
            init_points=self.n_rand,
            n_iter=self.iterations,
        )
        params = optimizer.max["params"]
        bestError = -optimizer.max["target"]
        
        bestArchitecture = copy.deepcopy(self.seedModel)
        for param in params:
            nodeIndex, paramName = int(param[:1]), param[2:]
            isInt = self.isInt[param]
            paramValue = int(params[param]) if isInt else params[param]
            bestArchitecture["nodes"][nodeIndex]['params'][paramName] = paramValue
        
        file = open(self.saveLocation, 'wb')
        pickle.dump(self, file)

        mainErrors = [e[0] for e in self.performances]
        bestErrors = self.performances[mainErrors.index(min(mainErrors) if self.minimizeFitness else max(mainErrors))]

        return bestErrors, self.bestModel