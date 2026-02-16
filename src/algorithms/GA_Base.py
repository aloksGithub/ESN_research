
import random
from deap import base, creator, tools
import copy
import time
import os

from .types import EvalParams, ExperimentData, GAParams, ModelParams
from ..utils import (
    evaluateArchitecture,
    generateRandomArchitectureOld,
    generateRandomNodeParams,
    nodeConstructors,
    nodeParameterRanges,
    isValidArchitecture,
)


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
            result = generateRandomArchitectureOld(
                self.experimentData.trainY.shape[-1],
                self.experimentData.trainX,
                self.evalParams.memoryLimit,
                self.modelParams.num_nodes_range,
            )
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
            
            for c in candidates[:self.n_jobs]:
                validity = self.checkModelValidity(c)
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
