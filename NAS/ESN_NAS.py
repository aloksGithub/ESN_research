from typing import List
import reservoirpy as rpy
from NAS.utils import (
    evaluateArchitecture,
    generateRandomArchitecture,
    generateRandomArchitectureOld,
    generateRandomNodeParams,
    nodeConstructors,
    nodeParameterRanges,
    constructModel,
    runModel,
    trainModel,
    isValidArchitecture,
)
from reservoirpy.observables import nrmse
import numpy as np
import random
from deap import base, creator, tools
import warnings
import pickle
from NAS.memory_estimator import measure_memory_usage
import copy

warnings.filterwarnings("ignore")
import time
from NAS.parallel_processing import executeParallelBatch
import os

rpy.verbosity(0)


class ExperimentData:
    def __init__(self, trainX, trainY, valX, valY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY


class GAParams:
    def __init__(
        self,
        generations,
        populationSize,
        crossoverProbability,
        mutationProbability,
        eliteSize,
        stagnationReset,
        earlyStop=None,
    ):
        self.generations = generations
        self.populationSize = populationSize
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        self.eliteSize = eliteSize
        self.stagnationReset = stagnationReset
        self.earlyStop = earlyStop


class EvalParams:
    def __init__(
        self,
        numEvals,
        errorMetrics,
        defaultErrors,
        isAutoRegressive,
        timeout,
        memoryLimit,
        minimizeFitness,
    ):
        self.numEvals = numEvals
        self.errorMetrics = errorMetrics
        self.defaultErrors = defaultErrors
        self.isAutoRegressive = isAutoRegressive
        self.timeout = timeout
        self.memoryLimit = memoryLimit
        self.minimizeFitness = minimizeFitness

class ModelParams:
    def __init__(self, num_nodes_range=(2, 4)):
        self.num_nodes_range = num_nodes_range


class GA_Base:
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
    ):
        self.experimentData = experimentData
        self.evalParams = evalParams
        self.gaParams = gaParams
        self.modelParams = modelParams
        self.outputDim = self.experimentData.trainY.shape[-1]

        if self.evalParams.minimizeFitness:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()

        self.toolbox.register("mate", self.crossover_one_point)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("selectTournament", tools.selTournament)
        self.toolbox.register("selectBest", tools.selBest)
        self.toolbox.register("selectWorst", tools.selWorst)

        self.seedModels = seedModels
        self.n_jobs = n_jobs
        self.saveModels = saveModels
        self.saveLocation = saveLocation if saveLocation is not None else "temp"

        self.generation = 1
        self.fitnesses = []
        self.generationTimes = []
        self.architectures = []
        self.models = []
        self.modelGenerationIndices = []
        self.generationsSinceImprovement = 0
        self.bestModel = None
        self.prevFitness = self.evalParams.defaultErrors[0]
        self.population = []
        self.bestFitness = self.evalParams.defaultErrors

        # Make sure that save folder exists
        directory = os.path.dirname(self.saveLocation)
        os.makedirs(directory, exist_ok=True)

    def evaluateArchitecture(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        on one step ahead prediction using errorMetrics on valX and valY.
        """
        return evaluateArchitecture(
            individual,
            self.experimentData.trainX,
            self.experimentData.trainY,
            self.experimentData.valX,
            self.experimentData.valY,
            self.evalParams.numEvals,
            self.evalParams.errorMetrics,
            self.evalParams.defaultErrors,
            self.evalParams.isAutoRegressive,
        )

    def generatePopulation(self, numIndividuals: int):
        print("Generating population")
        generatedArchitectures = []

        while len(generatedArchitectures) < numIndividuals:
            results = executeParallelBatch(
                generateRandomArchitectureOld,
                [
                    (
                        self.experimentData.trainY.shape[-1],
                        self.experimentData.trainX,
                        self.evalParams.memoryLimit,
                        self.modelParams.num_nodes_range,
                    )
                    for _ in range(numIndividuals - len(generatedArchitectures))
                ],
                self.n_jobs,
                10,     # Only a small timeout is needed to check architecture validity
                log_level=0,
            )
            for result in results:
                if result is not None:
                    generatedArchitectures.append(result)

        population = [
            creator.Individual(individual)
            for individual in generatedArchitectures[: self.gaParams.populationSize]
        ]
        return population

    def generateOffspring(self, population):
        print("Generating offspring")
        startTime = time.time()
        offspring = self.toolbox.selectBest(population, self.gaParams.eliteSize)
        candidates = []

        while len(offspring) < self.gaParams.populationSize:
            while len(candidates) < self.n_jobs:
                parent1 = self.toolbox.selectTournament(
                    population, 1, len(population) // 4
                )[0]
                parent2 = self.toolbox.selectTournament(
                    population, 1, len(population) // 4
                )[0]

                child1, child2 = self.crossover_one_point(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                candidates.append(child1)
                candidates.append(child2)
            
            validities = executeParallelBatch(
                self.checkModelValidity,
                [(c,) for c in candidates[:self.n_jobs]],
                self.n_jobs,
                10,
                0,
            )
            for validity in validities:
                if validity is not None and validity[0]:
                    offspring.append(validity[1])
            candidates = []

        print(f"Time taken to generate offspring: {time.time() - startTime} seconds")
        return offspring[: self.gaParams.populationSize]
    
    def checkModelValidity(self, architecture):
        return (
            isValidArchitecture(
                architecture,
                self.experimentData.trainX,
                self.evalParams.memoryLimit,
            ),
            architecture,
        )

    # Crossover function
    def crossover_one_point(self, ind1, ind2):
        ind1Copy = copy.deepcopy(ind1)
        ind2Copy = copy.deepcopy(ind2)
        if random.random() >= self.gaParams.crossoverProbability:
            return (ind1Copy, ind2Copy)
        maxNodeIndex = max(len(ind1Copy["nodes"]), len(ind2Copy["nodes"])) - 1
        point1 = random.randint(1, maxNodeIndex - 1)
        point2 = random.randint(point1, maxNodeIndex)
        child1_nodes = (
            ind1Copy["nodes"][:point1]
            + ind2Copy["nodes"][point1:point2]
            + ind1Copy["nodes"][point2:]
        )
        child2_nodes = (
            ind2Copy["nodes"][:point1]
            + ind1Copy["nodes"][point1:point2]
            + ind2Copy["nodes"][point2:]
        )
        ind1Copy["nodes"] = child1_nodes
        ind2Copy["nodes"] = child2_nodes
        return (ind1Copy, ind2Copy)

    # Mutation function
    def mutate(self, ind):
        """
        Mutate an individual. We can either:
        1. Swap out a node (excluding Input and Ridge nodes).
        2. Change a parameter of a node (again excluding Input and Ridge nodes).
        """
        indCopy = copy.deepcopy(ind)
        if random.random() >= self.gaParams.mutationProbability:
            return indCopy
        mutation_type = random.choice(["swap_node", "change_param"])

        if mutation_type == "swap_node":
            idx = random.randint(
                1, len(indCopy["nodes"]) - 2
            )  # Excluding Input and Ridge
            node_type = random.choice(list(nodeConstructors.keys() - {"Input"}))
            indCopy["nodes"][idx] = {
                "type": node_type,
                "params": generateRandomNodeParams(node_type, self.outputDim),
            }

        elif mutation_type == "change_param":
            idx = random.randint(
                1, len(indCopy["nodes"]) - 2
            )  # Excluding Input and Ridge
            node_type = indCopy["nodes"][idx]["type"]
            param_name = random.choice(list(nodeParameterRanges[node_type].keys()))
            param_range = nodeParameterRanges[node_type][param_name]

            if param_range["intOnly"]:
                indCopy["nodes"][idx]["params"][param_name] = random.randint(
                    param_range["lower"], param_range["upper"]
                )
            else:
                indCopy["nodes"][idx]["params"][param_name] = (
                    random.random() * (param_range["upper"] - param_range["lower"])
                    + param_range["lower"]
                )
        return indCopy


class ESN_NAS(GA_Base):
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

    def evaluateParallel(self, population):
        print("Evaluating population")
        results = executeParallelBatch(
            (self.evaluateArchitecture),
            [(individual,) for individual in population],
            self.n_jobs,
            self.evalParams.timeout * self.evalParams.numEvals,
        )
        for i in range(len(results)):
            if results[i] is None:
                results[i] = (population[i], self.evalParams.defaultErrors, None)

        for result in results:
            ind, errors, model = result
            self.fitnesses.append(errors)
            self.architectures.append(ind)
            if (
                errors[0] <= min([elem[0] for elem in self.fitnesses])
                or len(self.fitnesses) == 0
            ):
                self.bestModel = model
            if self.saveModels:
                self.models.append(model)
            ind.fitness.values = (errors[0],)
        return [performanceData[1][0] for performanceData in results]

    def generationRun(self, gen: int):
        startTime = time.time()
        print("=======================Generation {}=======================".format(gen))
        self.generationsSinceImprovement += 1
        offspring = self.generateOffspring(
            list(map(self.toolbox.clone, self.population))
        )

        # Evaluate offspring
        offSpringFitnesses = self.evaluateParallel(offspring)
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
            self.evaluateParallel(newRandomPopulation)
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
            self.fitnesses[-self.gaParams.populationSize :]
        ):
            if fitness[0] == self.evalParams.defaultErrors[0]:
                # print(self.architectures[-self.populationSize:][index])
                numFailures += 1
        print("Best so far:", self.bestFitness)
        print(
            "Failure rate: {}%".format(100 * numFailures / self.gaParams.populationSize)
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

        self.evaluateParallel(self.population)
        self.modelGenerationIndices.append(0)
        endTime = time.time()
        self.generationTimes.append(endTime - startTime)

        for gen in range(self.generation, self.gaParams.generations + 1):
            self.generationRun(gen)
            if self.gaParams.earlyStop is not None:
                if self.evalParams.minimizeFitness:
                    if self.bestFitness[0] <= self.gaParams.earlyStop:
                        break
                else:
                    if self.bestFitness[0] >= self.gaParams.earlyStop:
                        break

        file = open(self.saveLocation, "rb")
        return pickle.load(file)
