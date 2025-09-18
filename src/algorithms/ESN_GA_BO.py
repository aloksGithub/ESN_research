from bayes_opt import BayesianOptimization
from deap import creator
import pickle
import copy
import time

from ..algorithms.GA_Base import GA_Base
from ..algorithms.types import EvalParams, ExperimentData, GAParams, ModelParams
from ..parallel_processing import executeParallelBatch
from ..utils import (
    evaluateArchitecture,
    isValidArchitecture,
    nodeParameterRanges,
)

class ESNAS(GA_Base):
    """Genetic algorithm to obtain an optimized ESN architecture for a dataset"""

    def __init__(
        self,
        experimentData: ExperimentData,
        evalParams: EvalParams,
        gaParams: GAParams,
        modelParams: ModelParams,
        seedModels=[],
        n_jobs=1,
        saveModels=False,
        saveLocation=None,
        bo_init=2,
        bo_iter=2,
    ):
        super().__init__(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            seedModels,
            n_jobs,
            saveModels,
            saveLocation,
        )

        self.bo_init = bo_init
        self.bo_iter = bo_iter
        self.individualsPerGeneration = self.gaParams.populationSize * (
            self.bo_init + self.bo_iter
        )

    def calculateBounds(self, individual):
        pbounds = {}
        isInt = {}
        defaultParams = {}
        for i, node in enumerate(individual["nodes"][1:]):
            for param in node["params"]:
                if param == "output_dim" or param == "fb_connectivity":
                    continue
                lowerLimit = nodeParameterRanges[node["type"]][param]["lower"]
                upperLimit = nodeParameterRanges[node["type"]][param]["upper"]
                pbounds[str(i + 1) + "_" + param] = (lowerLimit, upperLimit)
                isInt[str(i + 1) + "_" + param] = nodeParameterRanges[node["type"]][
                    param
                ]["intOnly"]
                defaultParams[str(i + 1) + "_" + param] = node["params"][param]
        return pbounds, isInt, defaultParams

    def bo(self, individual):
        pbounds, isInt, defaultParams = self.calculateBounds(individual)

        individuals = []
        individualErrors = []
        bestError = (
            self.evalParams.defaultErrors[0]
            if self.evalParams.minimizeFitness
            else -self.evalParams.defaultErrors[0]
        )
        bestModel = None

        def black_box(**params):
            modifiedArchitecture = copy.deepcopy(individual)
            for param in params:
                nodeIndex, paramName = int(param[:1]), param[2:]
                paramValue = int(params[param]) if isInt[param] else params[param]
                modifiedArchitecture["nodes"][nodeIndex]["params"][
                    paramName
                ] = paramValue
            
            is_valid = isValidArchitecture(
                modifiedArchitecture,
                self.experimentData.trainX,
                self.evalParams.memoryLimit
            )

            if is_valid:
                currIndividual, errors, model = evaluateArchitecture(
                    modifiedArchitecture,
                    self.experimentData.trainX,
                    self.experimentData.trainY,
                    self.experimentData.valX,
                    self.experimentData.valY,
                    self.evalParams.numEvals,
                    self.evalParams.errorMetrics,
                    self.evalParams.defaultErrors,
                    self.evalParams.isAutoRegressive,
                )
            else:
                currIndividual = modifiedArchitecture
                errors = self.evalParams.defaultErrors
                model = None

            individuals.append(currIndividual)
            individualErrors.append(errors)

            nonlocal bestError
            nonlocal bestModel
            if (
                errors[0] < bestError
                and self.evalParams.minimizeFitness
                or (errors[0] > bestError and not self.evalParams.minimizeFitness)
            ):
                bestError = errors[0]
                bestModel = model

            return -errors[0]

        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True,
            verbose=0,
        )
        optimizer.probe(params=defaultParams)
        optimizer.maximize(
            init_points=self.bo_init,
            n_iter=self.bo_iter,
        )
        return (individuals, individualErrors, bestModel)

    def evaluateParallel(self, population):
        print("Evaluating population")
        startTime = time.time()
        results = executeParallelBatch(
            self.bo,
            [(individual,) for individual in population],
            self.n_jobs,
            self.evalParams.timeout * self.evalParams.numEvals * (self.bo_init + self.bo_iter),
        )
        
        for i in range(len(results)):
            if results[i] is None:
                results[i] = (
                    [population[i]] * (self.bo_iter + self.bo_init),
                    [self.evalParams.defaultErrors] * (self.bo_iter + self.bo_init),
                    None,
                )

        new_individuals = []
        for result in results:
            individuals, individualErrors, bestModel = result
            for i, individual in enumerate(individuals):
                ga_individual = creator.Individual(individual)
                ga_individual.fitness.values = (individualErrors[i][0],)
                new_individuals.append(ga_individual)

            self.architectures += individuals
            self.fitnesses += individualErrors
            bestBoError = min([elem[0] for elem in individualErrors])

            bestOverallError = min([elem[0] for elem in self.fitnesses])
            if bestBoError <= bestOverallError or len(self.fitnesses) == 0:
                self.bestModel = bestModel
        print("evaluated in", time.time() - startTime)
        return [
            fitness[0]
            for fitness in self.fitnesses[
                -len(population) * (self.bo_iter + self.bo_init) :
            ]
        ], new_individuals

    def generationRun(self, gen):
        startTime = time.time()
        print("=======================Generation {}=======================".format(gen))
        self.generationsSinceImprovement += 1
        offspring = self.generateOffspring(
            list(map(self.toolbox.clone, self.population))
        )
        print("Offspring generated", time.time() - startTime)

        # Evaluate offspring
        offSpringFitnesses, offspring = self.evaluateParallel(offspring)
        print("Offspring evaluated", time.time() - startTime)
        if (
            self.evalParams.minimizeFitness
            and min(offSpringFitnesses) < self.prevFitness
            or not self.evalParams.minimizeFitness
            and max(offSpringFitnesses) > self.prevFitness
        ):
            self.prevFitness = min(offSpringFitnesses)
            self.generationsSinceImprovement = 0

        if self.generationsSinceImprovement >= self.gaParams.stagnationReset:
            print("Resetting population due to stagnation")

            self.prevFitness = self.evalParams.defaultErrors[0]
            newRandomPopulation = self.generatePopulation(
                self.gaParams.populationSize - 1
            )
            _, newRandomPopulation = self.evaluateParallel(newRandomPopulation)
            print("New random population evaluated", time.time() - startTime)
            self.population[:] = (
                self.toolbox.selectBest(self.population, 1) + newRandomPopulation
            )
            self.modelGenerationIndices.append(gen)
        else:
            self.population[:] = offspring

        objective = [errors[0] for errors in self.fitnesses]
        bestIndex = (
            objective.index(min(objective))
            if self.evalParams.minimizeFitness
            else objective.index(max(objective))
        )
        self.bestFitness = self.fitnesses[bestIndex]
        numFailures = 0
        for index, fitness in enumerate(
            self.fitnesses[-self.individualsPerGeneration :]
        ):
            if fitness[0] == self.evalParams.defaultErrors[0]:
                # print(self.architectures[-self.populationSize:][index])
                numFailures += 1
        print("Best so far:", self.bestFitness)
        print(
            "Failure rate: {}%".format(
                100 * numFailures / self.individualsPerGeneration
            )
        )
        self.generationTimes.append(time.time() - startTime)
        print("Time taken:", time.time() - startTime)

        file = open(self.saveLocation, "wb")
        pickle.dump(self, file)

    def run(self):
        startTime = time.time()
        random_population = self.generatePopulation(
            self.gaParams.populationSize - len(self.seedModels)
        )
        seed_population = [
            creator.Individual(individual) for individual in self.seedModels
        ]
        self.population = seed_population + random_population

        _, self.population = self.evaluateParallel(self.population)
        self.modelGenerationIndices.append(0)
        endTime = time.time()
        self.generationTimes.append(endTime - startTime)

        for gen in range(self.generation, self.gaParams.generations + 1):
            self.generationRun(gen)

        file = open(self.saveLocation, "rb")
        return pickle.load(file)
