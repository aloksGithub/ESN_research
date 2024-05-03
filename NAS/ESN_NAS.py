from functools import partial
import reservoirpy as rpy
from NAS.utils import (
    generateRandomArchitecture,
    generateRandomNodeParams,
    nodeConstructors,
    nodeParameterRanges,
    constructModel,
    runModel,
    trainModel,
    isValidArchitecture
)
from reservoirpy.observables import nrmse
import numpy as np
import random
from deap import base, creator, tools
from joblib import Parallel, delayed
import warnings
import pickle
from NAS.memory_estimator import estimateMemory
import traceback
warnings.filterwarnings("ignore")

rpy.verbosity(0)

# Crossover function
def crossover_one_point(ind1, ind2):
    maxNodeIndex = max(len(ind1['nodes']), len(ind2['nodes'])) - 1
    point1 = random.randint(1, maxNodeIndex-1)
    point2 = random.randint(point1, maxNodeIndex)
    child1_nodes = ind1['nodes'][:point1] + ind2['nodes'][point1:point2] + ind1['nodes'][point2:]
    child2_nodes = ind2['nodes'][:point1] + ind1['nodes'][point1:point2] + ind2['nodes'][point2:]

    return ({"nodes": child1_nodes, "edges": ind1['edges']}, {"nodes": child2_nodes, "edges": ind2['edges']})

# Mutation function
def mutate(ind, output_dim):
    """
    Mutate an individual. We can either:
    1. Swap out a node (excluding Input and Ridge nodes).
    2. Change a parameter of a node (again excluding Input and Ridge nodes).
    """
    mutation_type = random.choice(["swap_node", "change_param"])
    
    if mutation_type == "swap_node":
        idx = random.randint(1, len(ind['nodes'])-2)  # Excluding Input and Ridge
        node_type = random.choice(list(nodeConstructors.keys() - {"Input"}))
        ind['nodes'][idx] = {"type": node_type, "params": generateRandomNodeParams(node_type, output_dim)}
    
    elif mutation_type == "change_param":
        idx = random.randint(1, len(ind['nodes'])-2)  # Excluding Input and Ridge
        node_type = ind['nodes'][idx]['type']
        param_name = random.choice(list(nodeParameterRanges[node_type].keys()))
        param_range = nodeParameterRanges[node_type][param_name]
        
        if param_range["intOnly"]:
            ind['nodes'][idx]['params'][param_name] = random.randint(param_range["lower"], param_range["upper"])
        else:
            ind['nodes'][idx]['params'][param_name] = random.random() * (param_range["upper"] - param_range["lower"]) + param_range["lower"]
    
    return ind

def generateArchitectures(generator, n, n_jobs):
    architectures = Parallel(n_jobs=n_jobs)(delayed(generator)() for i in range(n))
    return architectures

class ESN_NAS:
    """Genetic algorithm to obtain an optimized ESN architecture for a dataset"""

    def __init__(
        self,
        trainX,
        trainY,
        valX,
        valY,
        generations,
        populationSize,
        outputDim,
        crossoverProbability = 0.7,
        mutationProbability = 0.2,
        eliteSize = 1,
        numEvals = 1,
        errorMetrics = [nrmse],
        defaultErrors = [np.inf],
        seedModels=[],
        n_jobs=1,
        memoryLimit=4*1024,
        minimizeFitness=True,
        saveModels=False,
        timeout = 180,
        stagnationReset = 5,
        saveLocation = None
    ):
        if minimizeFitness:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.Fitness)

        self.generations = generations
        self.populationSize = populationSize
        self.errorMetrics = errorMetrics
        self.defaultErrors = defaultErrors
        self.seedModels = seedModels
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        self.eliteSize = eliteSize
        self.numEvals = numEvals
        self.n_jobs = n_jobs
        self.memoryLimit = memoryLimit
        self.minimizeFitness = minimizeFitness
        self.saveModels = saveModels
        self.timeout = timeout
        self.stagnationReset = stagnationReset
        self.saveLocation = saveLocation if saveLocation is not None else "temp"

        self.fitnessCache = []
        self.generation = 1
        self.fitnesses = []
        self.architectures = []
        self.models = []
        self.modelGenerationIndices = []
        self.generationsSinceImprovement = 0
        self.bestModel = None
        if minimizeFitness:
            self.defaultFitness = np.inf
        else:
            self.defaultFitness = 0
        self.prevFitness = self.defaultFitness

        self.toolbox = base.Toolbox()
        
        self.toolbox.register("mate", crossover_one_point)
        self.toolbox.register("mutate", mutate, output_dim = outputDim)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("selectWorst", tools.selWorst)

        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
    
    def evaluateArchitecture(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        on one step ahead prediction using errorMetrics on valX and valY.
        """
        index = len(self.fitnessCache)
        self.fitnessCache.append([individual, self.defaultErrors, constructModel(individual)])

        if not isValidArchitecture(individual, len(self.trainX), self.memoryLimit):
            return self.fitnessCache[index][1], self.fitnessCache[index][2]

        errors = []
        models = []
        for _ in range(self.numEvals):
            model = constructModel(individual)
            try:
                model = trainModel(model, self.trainX, self.trainY)
                preds = runModel(model, self.valX)
                modelErrors = [metric(self.valY, preds) for metric in self.errorMetrics]
                errors.append(modelErrors)
                models.append(model)
            except Exception as e:
                print(e)
                errors.append(self.defaultErrors)
                models.append(model)
                
            # Find index for model with best error metrics
            error0 = [modelErrors[0] for modelErrors in errors]
            bestErrorIndex = error0.index(max(error0)) if self.defaultErrors[0]==0 else error0.index(min(error0))
            
            self.fitnessCache[index][1] = errors[bestErrorIndex]
            self.fitnessCache[index][2] = models[bestErrorIndex]

        return self.fitnessCache[index][1], self.fitnessCache[index][2]

    def evaluateArchitectureAutoRegressive(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        using errorMetrics on valX and valY. Test prediction is done auto-regressively,
        the output from the current timestep is used as input for next timestep
        """
        index = len(self.fitnessCache)
        self.fitnessCache.append([individual, self.defaultErrors, constructModel(individual)])
        
        if not isValidArchitecture(individual, len(self.trainX), self.memoryLimit):
            return self.fitnessCache[index][1], self.fitnessCache[index][2]

        errors = []
        models = []
        for _ in range(self.numEvals):
            model = constructModel(individual)
            try:
                model = trainModel(model, self.trainX, self.trainY)
                prevOutput = self.valX[0]
                preds = []
                for _ in range(len(self.valX)):
                    pred = runModel(model, prevOutput)
                    prevOutput = pred
                    preds.append(pred[0])
                preds = np.array(preds)
                modelErrors = [metric(self.valY, preds) for metric in self.errorMetrics]
                errors.append(modelErrors)
                models.append(model)
            except:
                errors.append(self.defaultErrors)
                models.append(model)

            # Find index for model with best error metrics
            error0 = [modelErrors[0] for modelErrors in errors]
            bestErrorIndex = error0.index(max(error0)) if self.defaultErrors[0]==0 else error0.index(min(error0))
            
            self.fitnessCache[index][1] = errors[bestErrorIndex]
            self.fitnessCache[index][2] = models[bestErrorIndex]
        
        return errors[bestErrorIndex], models[bestErrorIndex]

    def evaluateParallel(self, population):
        self.fitnessCache = []
        try:
            parallel = Parallel(n_jobs=self.n_jobs, timeout=self.timeout, require='sharedmem')
            parallel(delayed(self.evaluateArchitecture)(architecture) for architecture in population)
        except:
            print("Timedout")
            pass
        
        for result in self.fitnessCache:
            ind, errors, model = result
            self.fitnesses.append(errors)
            self.architectures.append(ind)
            if errors[0]<=min([elem[0] for elem in self.fitnesses]) or len(self.fitnesses)==0:
                self.bestModel = model
            if self.saveModels:
                self.models.append(model)
            ind.fitness.values = (errors[0],)
        return [performanceData[1][0] for performanceData in self.fitnessCache]

    def generatePopulation(self, numIndividuals):
        generatedArchitectures = generateArchitectures(
            partial(
                generateRandomArchitecture,
                sampleX=self.trainX[:2000],
                sampleY=self.trainY[:2000],
                validThreshold=10,
                maxInput=len(self.trainX),
                memoryLimit=self.memoryLimit,
                numVal=200
            ),
            numIndividuals,
            self.n_jobs
        )
        population = [creator.Individual(individual) for individual in generatedArchitectures]
        return population

    def run(self):
        random_population = self.generatePopulation(self.populationSize - len(self.seedModels))
        seed_population = [creator.Individual(individual) for individual in self.seedModels]
        population = seed_population + random_population

        self.evaluateParallel(population)
        self.modelGenerationIndices.append(0)
        
        for gen in range(self.generation, self.generations + 1):
            self.generationsSinceImprovement+=1
            offspring = self.toolbox.select(population, self.populationSize)
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossoverProbability:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutationProbability:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            offSpringFitnesses = self.evaluateParallel(offspring)
            if self.minimizeFitness and min(offSpringFitnesses)<self.prevFitness or not self.minimizeFitness and max(offSpringFitnesses)>self.prevFitness:
                self.generationsSinceImprovement = 0

            if self.generationsSinceImprovement>=self.stagnationReset:
                print("Resetting population due to stagnation")

                self.prevFitness = self.defaultFitness
                newRandomPopulation = self.generatePopulation(self.populationSize-1)
                self.evaluateParallel(newRandomPopulation)
                population[:] = self.toolbox.select(population, 1) + newRandomPopulation
                self.modelGenerationIndices.append(gen)
            else:
                population[:] = offspring
            
            bestFitness = min(self.fitnesses)
            numFailures = 0
            for index, fitness in enumerate(self.fitnesses[-self.populationSize:]):
                if fitness[0]==self.defaultFitness:
                    print(self.architectures[-self.populationSize:][index])
                    numFailures+=1
            print("=======================Generation {}=======================".format(gen))
            print("Best so far:", bestFitness)
            print("Failure rate: {}%".format(100*numFailures/self.populationSize))

            checkpoint = {
                "generation": gen,
                "fitnesses": self.fitnesses,
                "architectures": self.architectures,
                "models": self.models,
                "modelGenerationIndices": self.modelGenerationIndices,
                "generationsSinceImprovement": self.generationsSinceImprovement,
                "population": population,
                "prevFitness": self.prevFitness,
                "defaultFitness": self.defaultFitness,
                "bestModel": self.bestModel
            }

            file = open(self.saveLocation, 'wb')
            pickle.dump(checkpoint, file)

        
        file = open(self.saveLocation, 'rb')
        return pickle.load(file)
