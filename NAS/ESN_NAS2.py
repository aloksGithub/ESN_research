from bayes_opt import BayesianOptimization
import reservoirpy as rpy
from NAS.utils import (
    generateRandomArchitecture,
    generateRandomArchitectureOld,
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
import warnings
import pickle
from NAS.memory_estimator import measure_memory_usage
import copy
warnings.filterwarnings("ignore")
import time
from NAS.parallel_processing import executeParallelBatch
import os
rpy.verbosity(0)

class ESN_NAS2:
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
        saveLocation = None,
        isAutoRegressive = False,
        bo_init = 2,
        bo_iter = 2,
    ):
        if minimizeFitness:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.Fitness)

        self.generations = generations
        self.populationSize = populationSize
        self.outputDim = outputDim
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
        self.isAutoregressive = isAutoRegressive
        self.bo_init = bo_init
        self.bo_iter = bo_iter

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
        
        self.toolbox.register("mate", self.crossover_one_point)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("selectTournament", tools.selTournament)
        self.toolbox.register("selectBest", tools.selBest)
        self.toolbox.register("selectWorst", tools.selWorst)

        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY

        self.diagnosisResults = []
        self.population = []
        self.bestFitness = defaultErrors
        # Make sure that save folder exists
        directory = os.path.dirname(self.saveLocation)
        os.makedirs(directory, exist_ok=True)
        
    def checkModelValidity(self, architecture):
        return isValidArchitecture(architecture, self.trainX, self.trainY, self.memoryLimit, self.timeout, self.isAutoregressive ), architecture

    def generateOffspring(self, population):
        print("Generating offspring")
        offspring = self.toolbox.selectBest(population, self.eliteSize)
        candidates = []

        while len(offspring) < self.populationSize:
            while len(candidates) < self.n_jobs:
                parent1 = self.toolbox.selectTournament(population, 1, len(population)//4)[0]
                parent2 = self.toolbox.selectTournament(population, 1, len(population)//4)[0]

                child1, child2 = self.crossover_one_point(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                candidates.append(child1)
                candidates.append(child2)
            
            validities = executeParallelBatch(self.checkModelValidity, [(c,) for c in candidates], self.n_jobs, self.timeout)
            for validity in validities:
                if validity is not None and validity[0]:
                    offspring.append(validity[1])
            candidates = []
        return offspring[:self.populationSize]
    
    # Crossover function
    def crossover_one_point(self, ind1, ind2):
        ind1Copy = copy.deepcopy(ind1)
        ind2Copy = copy.deepcopy(ind2)
        if random.random() >= self.crossoverProbability: return (ind1Copy, ind2Copy)
        maxNodeIndex = max(len(ind1Copy['nodes']), len(ind2Copy['nodes'])) - 1
        point1 = random.randint(1, maxNodeIndex-1)
        point2 = random.randint(point1, maxNodeIndex)
        child1_nodes = ind1Copy['nodes'][:point1] + ind2Copy['nodes'][point1:point2] + ind1Copy['nodes'][point2:]
        child2_nodes = ind2Copy['nodes'][:point1] + ind1Copy['nodes'][point1:point2] + ind2Copy['nodes'][point2:]
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
        if random.random() >= self.mutationProbability: return indCopy
        mutation_type = random.choice(["swap_node", "change_param"])
        
        if mutation_type == "swap_node":
            idx = random.randint(1, len(indCopy['nodes'])-2)  # Excluding Input and Ridge
            node_type = random.choice(list(nodeConstructors.keys() - {"Input"}))
            indCopy['nodes'][idx] = {"type": node_type, "params": generateRandomNodeParams(node_type, self.outputDim)}
        
        elif mutation_type == "change_param":
            idx = random.randint(1, len(indCopy['nodes'])-2)  # Excluding Input and Ridge
            node_type = indCopy['nodes'][idx]['type']
            param_name = random.choice(list(nodeParameterRanges[node_type].keys()))
            param_range = nodeParameterRanges[node_type][param_name]
            
            if param_range["intOnly"]:
                indCopy['nodes'][idx]['params'][param_name] = random.randint(param_range["lower"], param_range["upper"])
            else:
                indCopy['nodes'][idx]['params'][param_name] = random.random() * (param_range["upper"] - param_range["lower"]) + param_range["lower"]
        return indCopy

    def evaluateArchitecture(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        on one step ahead prediction using errorMetrics on valX and valY.
        """

        errors = []
        models = []
        for _ in range(self.numEvals):
            try:
                model = constructModel(individual)
                model = trainModel(model, self.trainX, self.trainY)
                model_copy = copy.deepcopy(model)
                preds = runModel(model, self.valX)
                modelErrors = [metric(self.valY, preds) for metric in self.errorMetrics]
                errors.append(modelErrors)
                models.append(model_copy)
            except Exception as e:
                # print(e)
                errors.append(self.defaultErrors)
                models.append(None)
                
            # Find index for model with best error metrics
            error0 = [modelErrors[0] for modelErrors in errors]
            bestErrorIndex = error0.index(max(error0)) if self.defaultErrors[0]==0 else error0.index(min(error0))
            
        return individual, errors[bestErrorIndex], models[bestErrorIndex]

    def evaluateArchitectureAutoRegressive(self, individual):
        """
        Instantiate random models using given architecture, then train and evaluate them
        using errorMetrics on valX and valY. Test prediction is done auto-regressively,
        the output from the current timestep is used as input for next timestep
        """

        errors = []
        models = []
        for _ in range(self.numEvals):
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
                modelErrors = [metric(self.valY, preds) for metric in self.errorMetrics]
                errors.append(modelErrors)
                models.append(model_copy)
            except:
                errors.append(self.defaultErrors)
                models.append(None)

            # Find index for model with best error metrics
            error0 = [modelErrors[0] for modelErrors in errors]
            bestErrorIndex = error0.index(max(error0)) if self.defaultErrors[0]==0 else error0.index(min(error0))
            
        return individual, errors[bestErrorIndex], models[bestErrorIndex]
    
    def calculateBounds(self, individual):
        pbounds = {}
        isInt = {}
        defaultParams = {}
        for i, node in enumerate(individual['nodes'][1:]):
            for param in node['params']:
                if param=="output_dim" or param=="fb_connectivity":
                    continue
                lowerLimit = nodeParameterRanges[node['type']][param]['lower']
                upperLimit = nodeParameterRanges[node['type']][param]['upper']
                pbounds[str(i+1) + "_" + param] = (lowerLimit, upperLimit)
                isInt[str(i+1) + "_" + param] = nodeParameterRanges[node['type']][param]['intOnly']
                defaultParams[str(i+1) + "_" + param] = node['params'][param]
        return pbounds, isInt
    
    def bo(self, individual):
        pbounds, isInt = self.calculateBounds(individual)

        individuals = []
        individualErrors = []
        bestError = np.inf if self.minimizeFitness else -np.inf
        bestModel = None
        
        def black_box(**params):
            modifiedArchitecture = copy.deepcopy(individual)
            for param in params:
                nodeIndex, paramName = int(param[:1]), param[2:]
                paramValue = int(params[param]) if isInt[param] else params[param]
                modifiedArchitecture["nodes"][nodeIndex]['params'][paramName] = paramValue
            
            currIndividual, errors, model = self.evaluateArchitectureAutoRegressive(modifiedArchitecture) if self.isAutoregressive else self.evaluateArchitecture(modifiedArchitecture)
            individuals.append(currIndividual)
            individualErrors.append(errors)

            nonlocal bestError
            nonlocal bestModel
            if errors[0]<bestError and self.minimizeFitness or (errors[0]>bestError and not self.minimizeFitness):
                bestError = errors[0]
                bestModel = model

            return -errors[0]
        
        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True,
            verbose=0
        )
        optimizer.maximize(
            init_points=self.bo_init,
            n_iter=self.bo_iter,
        )
        return (individuals, individualErrors, bestModel)

    def evaluateParallel(self, population):
        print("Evaluating population")
        results = executeParallelBatch(self.bo, [(individual,) for individual in population], self.n_jobs, self.timeout*self.numEvals*(self.bo_init+self.bo_iter))
        for i in range(len(results)):
            if results[i] is None:
                results[i] = ([population[i]]*(self.bo_iter+self.bo_init), [self.defaultErrors]*(self.bo_iter+self.bo_init), None)
        new_individuals = []
        for result in results:
            individuals, individualErrors, bestModel = result
            for i, individual in enumerate(individuals):
                ga_individual = creator.Individual(individual)
                ga_individual.fitness.values = (individualErrors[i][0],)
                new_individuals.append(ga_individual)

            self.architectures+=individuals
            self.fitnesses+=individualErrors
            bestBoError = min([elem[0] for elem in individualErrors])
            bestOverallError = min([elem[0] for elem in self.fitnesses])
            if bestBoError<=bestOverallError or len(self.fitnesses)==0:
                self.bestModel = bestModel
        
        return [fitness[0] for fitness in self.fitnesses[-len(population)*(self.bo_iter+self.bo_init):]], new_individuals
    
    def generatePopulation(self, numIndividuals):
        print("Generating population")
        generatedArchitectures = []

        while len(generatedArchitectures)<numIndividuals:
            results = executeParallelBatch(
                generateRandomArchitectureOld,
                [(
                    self.trainX.shape[-1],
                    self.trainY.shape[-1],
                    self.trainX,
                    self.trainY,
                    self.memoryLimit,
                    self.timeout
                ) for _ in range(numIndividuals - len(generatedArchitectures))],
                self.n_jobs,
                self.timeout
            )
            for result in results:
                if result is not None:
                    generatedArchitectures.append(result)

        population = [creator.Individual(individual) for individual in generatedArchitectures[:self.populationSize]]
        return population
    
    def generationRun(self, gen):
        startTime = time.time()
        print("=======================Generation {}=======================".format(gen))
        self.generationsSinceImprovement+=1
        offspring = self.generateOffspring(list(map(self.toolbox.clone, self.population)))

        # Evaluate offspring
        offSpringFitnesses, offspring = self.evaluateParallel(offspring)
        if self.minimizeFitness and min(offSpringFitnesses)<self.prevFitness or not self.minimizeFitness and max(offSpringFitnesses)>self.prevFitness:
            self.prevFitness = min(offSpringFitnesses)
            self.generationsSinceImprovement = 0

        if self.generationsSinceImprovement>=self.stagnationReset:
            print("Resetting population due to stagnation")

            self.prevFitness = self.defaultFitness
            newRandomPopulation = self.generatePopulation(self.populationSize-1)
            _, newRandomPopulation = self.evaluateParallel(newRandomPopulation)
            self.population[:] = self.toolbox.selectBest(self.population, 1) + newRandomPopulation
            self.modelGenerationIndices.append(gen)
        else:
            self.population[:] = offspring
        
        objective = [errors[0] for errors in self.fitnesses]
        bestIndex = objective.index(min(objective)) if self.minimizeFitness else objective.index(max(objective))
        self.bestFitness = self.fitnesses[bestIndex]
        numFailures = 0
        for index, fitness in enumerate(self.fitnesses[-self.populationSize:]):
            if fitness[0]==self.defaultFitness:
                # print(self.architectures[-self.populationSize:][index])
                numFailures+=1
        print("Best so far:", self.bestFitness)
        print("Failure rate: {}%".format(100*numFailures/self.populationSize))
        print("Time taken:", time.time() - startTime)

        file = open(self.saveLocation, 'wb')
        pickle.dump(self, file)

    def run(self):
        random_population = self.generatePopulation(self.populationSize - len(self.seedModels))
        seed_population = [creator.Individual(individual) for individual in self.seedModels]
        self.population = seed_population + random_population

        _, self.population = self.evaluateParallel(self.population)
        self.modelGenerationIndices.append(0)
        
        for gen in range(self.generation, self.generations + 1):
            self.generationRun(gen)
        
        file = open(self.saveLocation, 'rb')
        return pickle.load(file)
    
    def diagnosis(self):
        """
        Occassionally, an individual will be spawned that might consume excessive memory and result in a crash
        while also slipping past the existing checks to prevent such individuals. This function goes over the
        current population to find such individuals. The evaluation is done serially for each individual. We
        check the training time and memory usage to find the bad individual. This individual will be replaced
        with a newly generated random individual.
        Note: This function has the possibility of resulting in another crash. To deal with such situations,
        we save the diagnosis results to file and the diagnosis function can be called again. The model
        that caused the crash will be marked as a bad model and replaced at the end of the function

        """

        def memoryCheckFunc():
            model = constructModel(individual)
            model = trainModel(model, self.trainX, self.trainY)
            runModel(model, self.valX)

        for individual in self.population[len(self.diagnosisResults):]:
            self.diagnosisResults.append(True)
            startTime = time.time()
            memoryUsage = measure_memory_usage(memoryCheckFunc)
            timeTaken = time.time() - startTime
            if timeTaken<self.timeout and memoryUsage<self.memoryLimit:
                self.diagnosisResults[-1] = False
        
        numNewModels = 0
        for diagnosis in self.diagnosisResults:
            if diagnosis: numNewModels+=1
        newModels = self.generatePopulation(numNewModels)

        for diagnosis, i in enumerate(self.diagnosisResults):
            if diagnosis:
                self.population[i] = newModels[-1]
                newModels = newModels[:-1]
        self.diagnosisResults = []