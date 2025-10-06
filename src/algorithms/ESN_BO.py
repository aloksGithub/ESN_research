import numpy as np
import copy
from bayes_opt import BayesianOptimization
import pickle
import os
import time
from ..algorithms.types import EvalParams, ExperimentData
from ..parallel_processing import executeParallelBatch
from ..utils import (
    evaluateArchitecture,
    nodeParameterRanges,
)

class ESN_BO:
    """
    Hyper parameter optimization algorithm using bayesian optimization
    on top of an architecture obtained from NAS
    """

    def __init__(
        self,
        experimentData: ExperimentData,
        evalParams: EvalParams,
        n_rand,
        iterations,
        seedModel,
        n_jobs=3,
        saveLocation=None,
    ):
        self.experimentData = experimentData
        self.evalParams = evalParams
        self.n_rand = n_rand
        self.iterations = iterations
        self.outputDim = self.experimentData.trainY.shape[-1]
        self.seedModel = seedModel
        self.n_jobs = n_jobs
        self.saveLocation = saveLocation if saveLocation is not None else "temp"

        self.pbounds = {}
        self.isInt = {}
        self.defaultParams = {}
        for i, node in enumerate(seedModel["nodes"][1:]):
            for param in node["params"]:
                if param == "output_dim" or param == "fb_connectivity":
                    continue
                lowerLimit = nodeParameterRanges[node["type"]][param]["lower"]
                upperLimit = nodeParameterRanges[node["type"]][param]["upper"]
                self.pbounds[str(i + 1) + "_" + param] = (lowerLimit, upperLimit)
                self.isInt[str(i + 1) + "_" + param] = nodeParameterRanges[
                    node["type"]
                ][param]["intOnly"]
                self.defaultParams[str(i + 1) + "_" + param] = node["params"][param]

        # Variables to keep track of models
        self.paramsTested = []
        self.performances = []
        self.times = []
        self.bestModel = None
        self.start_time = time.time()
        self.end_time = time.time()

        # Make sure that save folder exists
        directory = os.path.dirname(self.saveLocation)
        os.makedirs(directory, exist_ok=True)

    def evaluateArchitecture(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        on one step ahead prediction using errorMetrics on valX and valY.
        """
        _, errors, model = evaluateArchitecture(
            individual,
            self.experimentData.trainX,
            self.experimentData.trainY,
            self.experimentData.valX,
            self.experimentData.valY,
            1,
            self.evalParams.errorMetrics,
            self.evalParams.defaultErrors,
            self.evalParams.isAutoRegressive,
        )
        return errors, model

    def evaluate(self, individual):
        start = time.time()
        results = executeParallelBatch(
            self.evaluateArchitecture,
            [(individual,) for _ in range(self.evalParams.numEvals)],
            self.evalParams.numEvals,
            self.evalParams.timeout,
        )

        valid_results = [res for res in results if res is not None]
        if not valid_results:
            errors = self.evalParams.defaultErrors
            bestModel = None
        else:
            main_errors = [res[0][0] for res in valid_results]
            best_error_index = (
                np.argmin(main_errors)
                if self.evalParams.minimizeFitness
                else np.argmax(main_errors)
            )
            errors = valid_results[best_error_index][0]
            bestModel = valid_results[best_error_index][1]

        self.paramsTested.append(individual)
        self.performances.append(errors)

        all_main_errors = [e[0] for e in self.performances]
        best_of_all_time_main_error = (
            min(all_main_errors) if self.evalParams.minimizeFitness else max(all_main_errors)
        )
        if (
            self.evalParams.minimizeFitness
            and best_of_all_time_main_error >= errors[0]
            or not self.evalParams.minimizeFitness
            and best_of_all_time_main_error <= errors[0]
        ):
            self.bestModel = bestModel

        end = time.time()
        print(f"Time taken: {end - start}")
        self.times.append(end - start)
        return errors

    def black_box(self, **params):
        modifiedArchitecture = copy.deepcopy(self.seedModel)
        for param in params:
            nodeIndex, paramName = int(param[:1]), param[2:]
            isInt = self.isInt[param]
            paramValue = int(params[param]) if isInt else params[param]
            modifiedArchitecture["nodes"][nodeIndex]["params"][paramName] = paramValue

        errors = self.evaluate(modifiedArchitecture)
        return -errors[0]

    def run(self):
        optimizer = BayesianOptimization(
            f=self.black_box,
            pbounds=self.pbounds,
            random_state=1,
            allow_duplicate_points=True,
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
            bestArchitecture["nodes"][nodeIndex]["params"][paramName] = paramValue

        file = open(self.saveLocation, "wb")
        pickle.dump(self, file)

        mainErrors = [e[0] for e in self.performances]
        bestErrors = self.performances[
            mainErrors.index(
                min(mainErrors) if self.evalParams.minimizeFitness else max(mainErrors)
            )
        ]
        self.end_time = time.time()

        return bestErrors, self.bestModel
