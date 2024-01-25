import reservoirpy as rpy
from reservoirpy.nodes import (Reservoir, IPReservoir, NVAR, RLS, Input)
from NAS.Ridge_parallel import Ridge
from NAS.LMS_serializable import LMS
# from reservoirpy.observables import (rmse, rsquare, nrmse, mse)
import numpy as np
from functools import reduce
import random
import math
from contextlib import contextmanager
import threading
import _thread
import time
from deap import base, creator, tools
from joblib import Parallel, delayed
import copy
from queue import Queue
import queue
import warnings
import pickle
warnings.filterwarnings("ignore")

rpy.verbosity(0)
output_dim = 1

class VotingEnsemble:
    def __init__(self, models, threshold):
        self.models = models
        self.threshold = threshold

    def run(self, x):
        preds = []
        for dataPoint in x:
            upVotes = 0
            downVotes = 0
            for model in self.models:
                pred = runModel(model, dataPoint)[0][0]
                if pred<-self.threshold:
                    downVotes+=1
                if pred>self.threshold:
                    upVotes+=1
            if upVotes>=2 and downVotes<=1:
                preds.append(self.threshold*1.1)
            elif downVotes>=2 and upVotes<=1:
                preds.append(-self.threshold*1.1)
        return np.expand_dims(np.array(preds), axis=1)

class StackedEnsemble:
    def __init__(self, models):
        self.finalLayer = Reservoir(600, sr=0.8, rc_connectivity=0.01, noise_in=1e-10) >> Ridge(output_dim=output_dim, ridge=1e-5)
        self.models = models

    def train(self, trainX, trainY):
        for model in self.models:
            trainModel(model, trainX, trainY)
        preds = []
        for model in self.models:
            preds.append(runModel(model, trainX))
        preds = np.array(preds)
        preds = preds.transpose(1, 0, 2).reshape(preds.shape[1], 3)
        self.finalLayer.fit(np.concatenate([preds, trainX], axis=1), trainY, 100)

    def run(self, x):
        preds = []
        for model in self.models:
            preds.append(runModel(model, x))
        preds = np.array(preds)
        preds = preds.transpose(1, 0, 2).reshape(preds.shape[1], 3)
        finalPreds = self.finalLayer.run(np.concatenate([preds, x], axis=1))
        return finalPreds

class Ensemble:
    def __init__(self, models):
        self.models = models

    def train(self, trainX, trainY):
        for model in self.models:
            trainModel(model, trainX, trainY)

    def run(self, x):
        preds = []
        for model in self.models:
            preds.append(runModel(model, x))
        return sum(arr for arr in preds)/len(preds)

nodeConstructors = {
    "Input": Input,
    "Reservoir": Reservoir,
    "IPReservoir": IPReservoir,
    "NVAR": NVAR,
    "Ridge": Ridge,
    "LMS": LMS,
    "RLS": RLS
}

nodeParameterRanges = {
    "Input": {},
    "Reservoir": {
        "units": {"lower": 10, "upper": 3000, "intOnly": True},
        "lr": {"lower": 0.2, "upper": 1, "intOnly": False},
        "sr": {"lower": 0.5, "upper": 2, "intOnly": False},
        "input_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False},
        "rc_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False},
        "fb_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False}
    },
    "IPReservoir": {
        "units": {"lower": 10, "upper": 3000, "intOnly": True},
        "lr": {"lower": 0.2, "upper": 1, "intOnly": False},
        "sr": {"lower": 0.5, "upper": 1, "intOnly": False},
        "mu": {"lower": -0.2, "upper": 0.2, "intOnly": False},
        "sigma": {"lower": 0, "upper": 2, "intOnly": False},
        "learning_rate": {"lower": 0, "upper": 0.01, "intOnly": False},
        "input_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False},
        "rc_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False},
        "fb_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False}
    },
    "NVAR": {
        "delay": {"lower": 1, "upper": 4, "intOnly": True},
        "order": {"lower": 1, "upper": 2, "intOnly": True},
        "strides": {"lower": 1, "upper": 2, "intOnly": True}
    },
    "Ridge": {
        "ridge": {"lower": 0, "upper": 0.0001, "intOnly": False}
    },
    "LMS": {
        "alpha": {"lower": 0, "upper": 1, "intOnly": False}
    },
    "RLS": {
        "alpha": {"lower": 0, "upper": 1, "intOnly": False}
    }
}

def generateRandomNodeParams(nodeType):
    params = {}
    if nodeType=="Ridge" or nodeType=="LMS" or nodeType=="RLS":
        params["output_dim"] = output_dim
    parameterRanges = nodeParameterRanges[nodeType]
    for parameterName in parameterRanges:
        parameterRange = parameterRanges[parameterName]
        if parameterRange["intOnly"]:
            params[parameterName] = random.randint(parameterRange["lower"], parameterRange["upper"])
        else:
            params[parameterName] = random.random() * (parameterRange["upper"] - parameterRange["lower"]) + parameterRange["lower"]
    return params

def isValidArchitecture(architecture):
    ipExists = False
    forceExists = False
    for i, node in enumerate(architecture["nodes"]):
        if node["type"]=="IPReservoir":
            ipExists = True
        if node["type"]=="LMS" or node["type"]=="RLS":
            forceExists = True
        if node["type"]=="NVAR":
            for connection in architecture["edges"]:
                if connection[1]!=i:
                    continue
                prevNode = architecture["nodes"][connection[0]]
                if prevNode["type"]=="Reservoir" or prevNode["type"]=="IPReservoir" or prevNode["type"]=="NVAR":
                    return False
    if ipExists and forceExists:
        return False
    return True

def generateRandomArchitecture(sampleX, sampleY):
    num_nodes = random.randint(2, 3)

    nodes = [
        {"type": "Input", "params": {}}
    ]

    for i in range(num_nodes):
        available_node_types = list(nodeConstructors.keys())
        if i==0:
            available_node_types.remove("LMS")
            available_node_types.remove("RLS")
            available_node_types.remove("Ridge")
        available_node_types.remove("Input")
        for node in nodes:
            if node["type"]=="IPReservoir":
                if "LMS" in available_node_types:
                    available_node_types.remove("LMS")
                if "RLS" in available_node_types:
                    available_node_types.remove("RLS")
            if (node["type"]=="LMS" or node["type"]=="RLS") and "IPReservoir" in available_node_types:
                available_node_types.remove("IPReservoir")
        
        node_type = random.choice(available_node_types)

        node_params = generateRandomNodeParams(node_type)
        nodes.append({"type": node_type, "params": node_params})

    edges = []
    connected_nodes = {0}  # start with the first node being "connected"
    
    for i in range(1, len(nodes)):
        while (True):
            source = random.choice([node for node in list(connected_nodes) if node != i])
            if (nodes[source]['type']=="Reservoir" or nodes[source]['type']=="IPReservoir" or nodes[source]['type']=="NVAR") and nodes[i]["type"]=="NVAR":
                continue
            if nodes[source]['type']=="IPReservoir" and (nodes[i]["type"]=="RLS" or nodes[i]["type"]=="LLS"):
                continue
            if [source, i] not in edges and [i, source] not in edges:
                edges.append([source, i]) 
                connected_nodes.add(i)
                break

        # unconnected_nodes = list((set(range(len(nodes))) - connected_nodes) - {i})
        # if unconnected_nodes:
        #     additional_target = random.choice(unconnected_nodes)
        #     if [i, additional_target] not in edges and [additional_target, i] not in edges:
        #         edges.append([i, additional_target])
        #         print("B", [i, additional_target], connected_nodes)
        #         connected_nodes.add(additional_target)


    # Adding the readout node
    ipExists = False
    for node in nodes:
        if node["type"]=="IPReservoir":
            ipExists = True
    if ipExists:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge")}
        ]
    else:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge")},
            {"type": "LMS", "params": generateRandomNodeParams("LMS")},
            {"type": "RLS", "params": generateRandomNodeParams("RLS")}
        ]
    nodes.append(random.choice(readouts))

    final_node_index = len(nodes) - 1
    for i in range(final_node_index):
        isOutputNode = True
        for edge in edges:
            if edge[0]==i: isOutputNode = False
        if isOutputNode:
            edges.append([i, final_node_index])

    architecture = {"nodes": nodes, "edges": edges}

    # Try to run the model on a small sample to see if it is a valid network
    # Otherwise generate a new architecture
    try:
        # performance, _ = evaluateArchitecture(architecture, sampleX, sampleY, sampleX, sampleY, 1, 1)
        # model = constructModel(architecture)
        performance, _, _ = evaluateArchitecture2(architecture, sampleX[:-200], sampleY[:-200], sampleX[-200:], sampleY[-200:], 1)
        if math.isnan(performance) or np.isinf(performance) or performance>1: raise Exception("Bad Model")
        print("Model found")
        return architecture
    except Exception as e:
        return generateRandomArchitecture(sampleX, sampleY)

def constructModel(architecture):
    nodes = [nodeConstructors[nodeData['type']](**nodeData['params']) for nodeData in architecture['nodes']]

    # Start with the first connection
    model = nodes[architecture['edges'][0][0]] >> nodes[architecture['edges'][0][1]]

    # Continue with the rest of the edges
    for edge in architecture['edges'][1:]:
        model = model & (nodes[edge[0]] >> nodes[edge[1]])

    return model

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

def trainUnderTime(queue, model, trainX, trainY):
    try:
        model = trainModel(model, trainX, trainY)
        queue.put(model)
    except:
        queue.put(None)
    
    
def trainModel(model, trainX, trainY):
    if isinstance(model, Ensemble) or isinstance(model, StackedEnsemble)  or isinstance(model, VotingEnsemble):
        model.train(trainX, trainY)
        return model
    nodes = [node.name for node in model.nodes]
    hasOnlineNode = False
    hasOfflineNode = False
    for node in nodes:
        if "LMS" in node or "RLS" in node:
            hasOnlineNode = True
        if "Ridge" in node:
            hasOfflineNode = True
    notLastNodes = []
    for edge in model.edges:
        if edge[0].name not in notLastNodes:
            notLastNodes.append(edge[0].name)
    output_nodes = list(set(nodes) - set(notLastNodes))
    outputNode = output_nodes[0]
    isOutputNodeOffline = "Ridge" in outputNode
    if hasOfflineNode:
        model.fit(trainX, trainY, warmup=min(int(len(trainX)/10), 82))
    if hasOnlineNode:
        model.train(trainX, trainY)
    if isOutputNodeOffline:
        model.fit(trainX, trainY, warmup=min(int(len(trainX)/10), 82))
    return model

def runModel(model, x):
    if isinstance(model, Ensemble) or isinstance(model, StackedEnsemble)  or isinstance(model, VotingEnsemble):
        return model.run(x)
    nodes = [node.name for node in model.nodes]
    notLastNodes = []
    for edge in model.edges:
        if edge[0].name not in notLastNodes:
            notLastNodes.append(edge[0].name)
    output_nodes = list(set(nodes) - set(notLastNodes))
    nodePreds = model.run(x)
    if isinstance(nodePreds, np.ndarray):
        return nodePreds
    else:
        return nodePreds[output_nodes[-1]]
    
def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_norm = np.linalg.norm(np.mean(y_true))
    
    return rmse / mean_norm
    
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / denominator)

def evaluateModelAutoRegressive2(model, trainX, trainY, valX, valY):
    try:
        model = trainModel(model, trainX, trainY)
        prevOutput = valX[0]
        preds = []
        for _ in range(len(valX)):
            pred = runModel(model, prevOutput)
            prevOutput = pred
            preds.append(pred[0])
        preds = np.array(preds)
        return nrmse(valY, preds), r_squared(valY, preds)
    except Exception as e:
        print(e)
        return np.inf, 0
    
def evaluateModelAutoRegressive(model, trainX, trainY, valX, valY):
    try:
        model = trainModel(model, trainX, trainY)
        prevOutput = valX[0]
        preds = []
        for _ in range(len(valX)):
            pred = runModel(model, prevOutput)
            prevOutput = pred
            preds.append(pred[0])
        preds = np.array(preds)
        return nrmse(valY, preds)
    except Exception as e:
        # print(e)
        return np.inf

def evaluateModel(model, trainX, trainY, valX, valY):
    try:
        model = trainModel(model, trainX, trainY)
        preds = runModel(model, valX)
        return nrmse(valY, preds)
    except Exception as e:
        # print(e)
        return np.inf

def evaluateArchitecture(individual, trainX, trainY, valX, valY, numEvals=3, timeout=60):
    if not isValidArchitecture(individual):
        return np.inf, constructModel(individual)
    q = queue.Queue()

    def work():
        model = constructModel(individual)
        performance = evaluateModelAutoRegressive(model, trainX, trainY, valX, valY)
        q.put([performance, model])

    performances = []
    models = []
    for _ in range(numEvals):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            performances.append(np.inf)
            models.append(constructModel(individual))
        else:
            result = q.get()
            performance, model = result[0], result[1]
            if math.isnan(performance):
                performances.append(np.inf)
            else:
                performances.append(performance)
            models.append(model)
    return min(performances), models[performances.index(min(performances))]

def evaluateArchitecture2(individual, trainX, trainY, valX, valY, numEvals=3, timeout=180):
    if not isValidArchitecture(individual):
        return np.inf, 0, constructModel(individual)
    q = queue.Queue()

    def work():
        model = constructModel(individual)
        nrmse, r_square = evaluateModelAutoRegressive2(model, trainX, trainY, valX, valY)
        q.put([nrmse, r_square, model])

    nrmses = []
    r_squared_vals = []
    models = []
    for _ in range(numEvals):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            nrmses.append(np.inf)
            r_squared_vals.append(0)
            models.append(constructModel(individual))
        else:
            result = q.get()
            nrmse, r_square, model = result[0], result[1], result[2]
            if math.isnan(nrmse):
                nrmses.append(np.inf)
                r_squared_vals.append(0)
            else:
                nrmses.append(nrmse)
                r_squared_vals.append(r_square)
            models.append(model)
    return min(nrmses), r_squared_vals[nrmses.index(min(nrmses))], models[nrmses.index(min(nrmses))]

# Crossover function
def crossover_one_point(ind1, ind2):
    maxNodeIndex = max(len(ind1['nodes']), len(ind2['nodes'])) - 1
    point1 = random.randint(1, maxNodeIndex-1)
    point2 = random.randint(point1, maxNodeIndex)
    child1_nodes = ind1['nodes'][:point1] + ind2['nodes'][point1:point2] + ind1['nodes'][point2:]
    child2_nodes = ind2['nodes'][:point1] + ind1['nodes'][point1:point2] + ind2['nodes'][point2:]

    return ({"nodes": child1_nodes, "edges": ind1['edges']}, {"nodes": child2_nodes, "edges": ind2['edges']})


# Mutation function
def mutate(ind):
    """
    Mutate an individual. We can either:
    1. Swap out a node (excluding Input and Ridge nodes).
    2. Change a parameter of a node (again excluding Input and Ridge nodes).
    """
    mutation_type = random.choice(["swap_node", "change_param"])
    
    if mutation_type == "swap_node":
        idx = random.randint(1, len(ind['nodes'])-2)  # Excluding Input and Ridge
        node_type = random.choice(list(nodeConstructors.keys() - {"Input"}))
        ind['nodes'][idx] = {"type": node_type, "params": generateRandomNodeParams(node_type)}
    
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

def runGA(params, useBackup = False):
    if params['minimizeFitness']:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    else:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", dict, fitness=creator.Fitness)

    if not useBackup:
        generation = 1
        allModels = []
        allFitnesses = []
        fitnesses2 = []
        allArchitectures = []
        generationsSinceImprovement = 0
        earlyStopReached = False
        if params['minimizeFitness']:
            defaultFitness = np.inf
        else:
            defaultFitness = 0
        prevFitness = defaultFitness
    else:
        file = open('backup/backup.obj', 'rb')
        data = pickle.load(file)
        
        generation = data["generation"]
        allModels = []
        allFitnesses = data["allFitnesses"]
        fitnesses2 = data["fitnesses2"]
        allArchitectures = data["allArchitectures"]
        generationsSinceImprovement = data["generationsSinceImprovement"]
        population = [creator.Individual(individual) for individual in data["population"]]
        earlyStopReached = data["earlyStopReached"]
        prevFitness = data["prevFitness"]
        params = data["params"]
        defaultFitness = data["defaultFitness"]

    toolbox = base.Toolbox()

    def initIndividual(icls, content):
        return icls(content)

    def initPopulation(pcls, ind_init, individuals):
        results = Parallel(n_jobs=params["n_jobs"])(delayed(ind_init)(c) for c in individuals)
        return pcls(results)
    
    
    toolbox.register("individual", tools.initIterate, creator.Individual, params["generator"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_seed", initPopulation, list, toolbox.individual_guess, params["seedModels"])
    toolbox.register("mate", crossover_one_point)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", params["evaluator"])
    
    if not useBackup:
        random_population = [creator.Individual(individual) for individual in  generateArchitectures(params["generator"], params["populationSize"] - len(params["seedModels"]), params["n_jobs"])]
        seed_population = [creator.Individual(individual) for individual in params["seedModels"]]
        population = seed_population + random_population

        fitnesses = Parallel(n_jobs=params["n_jobs"])(delayed(params["evaluator"])(architecture) for architecture in population)
        for ind, fitness_model in zip(population, fitnesses):
            fit, fit2, model = fitness_model
            allModels.append(model)
            allFitnesses.append(fit)
            fitnesses2.append(fit2)
            allArchitectures.append(ind)
            if ((fit <= params['earlyStop'] and params['minimizeFitness']) or 
                (not params['minimizeFitness'] and fit >= params['earlyStop'])):
                earlyStopReached = True
            if params["logModels"]:
                print(fit, ind)
            ind.fitness.values = (fit,)


    
    for gen in range(generation, params["generations"] + 1):
        if earlyStopReached:
            print("Early stopping criteria met")
            break
        generationsSinceImprovement+=1
        if params["logModels"]:
            print("Generation:", gen)
        offspring = toolbox.select(population, len(population) - (params["eliteSize"]))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < params["crossoverProbability"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < params["mutationProbability"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring
        fitnesses = Parallel(n_jobs=params["n_jobs"])(delayed(params["evaluator"])(architecture) for architecture in offspring)
        for ind, fitness_model in zip(offspring, fitnesses):
            fit, fit2, model = fitness_model
            allModels.append(model)
            allFitnesses.append(fit)
            fitnesses2.append(fit2)
            allArchitectures.append(ind)
            if ((fit<=params['earlyStop'] and params['minimizeFitness']) or (not params['minimizeFitness'] and fit>=params['earlyStop'])):
                earlyStopReached = True
            if (params['minimizeFitness'] and fit < prevFitness) or (not params['minimizeFitness'] and fit > prevFitness):
                prevFitness = fit
                generationsSinceImprovement = 0
            if params["logModels"]:
                print(fit, ind)
            ind.fitness.values = (fit,)

        elites = toolbox.select(population, params["eliteSize"])

        if generationsSinceImprovement>=params["stagnationReset"]:
            if params["logModels"]:
                print("Resetting population due to stagnation")
            prevFitness = defaultFitness
            newRandomPopulation = [creator.Individual(individual) for individual in  generateArchitectures(params["generator"], params["populationSize"] - 1, params["n_jobs"])]
            fitnesses = Parallel(n_jobs=params["n_jobs"])(delayed(params["evaluator"])(architecture) for architecture in newRandomPopulation)
            for ind, fitness_model in zip(newRandomPopulation, fitnesses):
                fit, fit2, model = fitness_model
                allModels.append(model)
                allFitnesses.append(fit)
                fitnesses2.append(fit2)
                allArchitectures.append(ind)
                if ((fit<=params['earlyStop'] and params['minimizeFitness']) or (not params['minimizeFitness'] and fit>=params['earlyStop'])):
                    earlyStopReached = True
                if params["logModels"]:
                    print(fit, ind)
                ind.fitness.values = (fit,)
            population[:] = toolbox.select(population, 1) + newRandomPopulation
        else:
            population[:] = elites + offspring
        
        bestFitness1 = min(allFitnesses)
        bestFitness2 = fitnesses2[allFitnesses.index(min(allFitnesses))]
        print("Best so far:", bestFitness1, bestFitness2)

        checkpoint = {
            "generation": gen+1,
            "allModels": allArchitectures,
            "allFitnesses": allFitnesses,
            "fitnesses2": fitnesses2,
            "allArchitectures": allArchitectures,
            "generationsSinceImprovement": generationsSinceImprovement,
            "population": population,
            "earlyStopReached": earlyStopReached,
            "prevFitness": prevFitness,
            "params": params,
            "defaultFitness": defaultFitness
        }
        file = open('backup/backup.obj', 'wb')

        # dump information to that file
        pickle.dump(checkpoint, file)

    paired_data = list(zip(allModels, allFitnesses, allArchitectures))
    if not params['minimizeFitness']:
        sorted_data = sorted(paired_data, key=lambda x: x[1], reverse=True)
    else:
        sorted_data = sorted(paired_data, key=lambda x: x[1], reverse=False)
    
    for i in range(len(sorted_data)-1, 0, -1):
        _, performance, architecture = sorted_data[i]
        if (params["minimizeFitness"] and performance>100) or (not params["minimizeFitness"] and performance<=0):
            del sorted_data[i]
            continue
        # try:
        #     model1 = constructModel(architecture)
        #     randomData = np.random.rand(100, sorted_data[0][0].nodes[0].input_dim)
        #     randomOutput = np.random.rand(100, output_dim)
        #     model1 = trainModel(model1, randomData, randomOutput)
        #     modelPreds = runModel(model1, randomData)
        # except:
        #     del sorted_data[i]
        #     continue
        for data in sorted_data[:i]:
            _, betterPerformance, betterArchitecture = data
            if betterArchitecture==architecture or betterPerformance==performance:
                del sorted_data[i]
                break
            # try:
            #     model2 = constructModel(betterArchitecture)
            #     model2 = trainModel(model2, randomData, randomOutput)
            #     betterModelPreds = runModel(model2, randomData)
            #     if modelPreds.shape[1]!=output_dim or np.all(betterModelPreds==modelPreds):
            #         del sorted_data[i]
            #         break
            # except:
            #     pass
    bestModels = [model for model, _, _ in sorted_data]
    performances = [performance for _, performance, _ in sorted_data]
    sortedArchitectures = [architecture for _, _, architecture in sorted_data]

    return bestModels, performances, sortedArchitectures