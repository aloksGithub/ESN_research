from tabulate import tabulate
import reservoirpy as rpy
from reservoirpy.nodes import (Reservoir, IPReservoir, NVAR, LMS, RLS, Ridge, ESN, Input)
from reservoirpy.datasets import (lorenz, mackey_glass, narma)
from reservoirpy.observables import (rmse, rsquare, nrmse, mse)
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import random
from deap import base, creator, tools
import math
import pandas as pd
import json
from contextlib import contextmanager
import threading
import _thread

rpy.verbosity(0)
output_dim = 1

rpy.verbosity(0)
output_dim = 1

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
        "units": {"lower": 10, "upper": 500, "intOnly": True},
        "lr": {"lower": 0.01, "upper": 1, "intOnly": False},
        "sr": {"lower": 0.5, "upper": 2, "intOnly": False}
    },
    "IPReservoir": {
        "units": {"lower": 10, "upper": 500, "intOnly": True},
        "lr": {"lower": 0, "upper": 1, "intOnly": False},
        "sr": {"lower": 0.5, "upper": 2, "intOnly": False},
        "mu": {"lower": -0.2, "upper": 0.2, "intOnly": False},
        "sigma": {"lower": 0, "upper": 2, "intOnly": False},
        "learning_rate": {"lower": 0, "upper": 0.01, "intOnly": False},
    },
    "NVAR": {
        "delay": {"lower": 0, "upper": 3, "intOnly": True},
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

def generateRandomArchitecture():
    num_nodes = random.randint(3, 7)

    nodes = [
        {"type": "Input", "params": {}}
    ]
    available_node_types = list(nodeConstructors.keys())
    available_node_types.remove("Input")

    for _ in range(num_nodes):
        node_type = random.choice(available_node_types)
        node_params = generateRandomNodeParams(node_type)
        nodes.append({"type": node_type, "params": node_params})

    edges = []
    connected_nodes = {0}  # start with the first node being "connected"
    
    for i in range(1, len(nodes)):
        source = random.choice([node for node in list(connected_nodes) if node != i])
        if [source, i] not in edges: edges.append([source, i]) 
        connected_nodes.add(i)

        unconnected_nodes = list((set(range(len(nodes))) - connected_nodes) - {i})
        if unconnected_nodes:
            additional_targets = [node for node in unconnected_nodes if node != i]  # ensure node doesn't connect to itself
            if additional_targets:
                additional_target = random.choice(additional_targets)
                if [i, additional_target] not in edges: edges.append([i, additional_target]) 
                connected_nodes.add(additional_target)


    # Adding the readout node
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

    toRemove = []
    for i in range(len(edges)):
        edge = edges[i]
        node1 = nodes[edge[0]]
        node2 = nodes[edge[1]]
        if node2['type']=="NVAR" and node1['type']!="Input":
            toRemove.append(i)
    for edgeIndex in reversed(toRemove):
        del edges[edgeIndex]
    
    for i in range(len(nodes)):
        if nodes[i]['type']=="NVAR" and not [0, i] in edges:
            edges.insert(0, [0, i])

    architecture = {"nodes": nodes, "edges": edges}

    # Try to run the model on a small sample to see if it is a valid network
    # Otherwise generate a new architecture
    try:
        model = constructModel(architecture)
        performance = evaluateModel(model, trainX, trainY, valX, valY)
        if math.isnan(performance) or np.isinf(performance) or performance>10: raise Exception("Bad Model")
        del model
        return architecture
    except:
        return generateRandomArchitecture()

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

def evaluateModel(model, trainX, trainY, valX, valY):
    if model.is_trained_online:
        try:
            with time_limit(30):
                model.train(trainX, trainY)
        except TimeoutException as e:
            print("Timed out!")
    else:
        try:
            with time_limit(30):
                model.fit(trainX, trainY, warmup=min(int(len(trainX)/10), 100))
        except TimeoutException as e:
            print("Timed out!")
    nodePreds = model.run(valX)
    if isinstance(nodePreds, np.ndarray):
        return nrmse(nodePreds, valY)
    else:
        bestError = np.inf
        for node in nodePreds:
            preds = nodePreds[node]
            bestError = min(bestError, nrmse(preds, valY))
        return bestError
    
def nmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    error = mse(y_true, y_pred)
    return error / np.asarray(y_true).mean()

def evaluateModelNMSE(model, trainX, trainY, valX, valY):
    if model.is_trained_online:
        model.train(trainX, trainY)
    else:
        model.fit(trainX, trainY, warmup=min(int(len(trainX)/10), 100))
    nodePreds = model.run(valX)
    if isinstance(nodePreds, np.ndarray):
        return nmse(nodePreds, valY)
    else:
        bestError = np.inf
        for node in nodePreds:
            preds = nodePreds[node]
            bestError = min(bestError, nmse(preds, valY))
        return bestError
    
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMin)

def evaluate(individual):
    model = constructModel(individual)
    try:
        performance = evaluateModel(model, trainX, trainY, valX, valY)
        del model
        return performance
    except:
        del model
        return np.inf

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


def runGA(populationSize=5, generations=5, seed_models=[]):
    eliteSize = int(populationSize/2)
    
    toolbox = base.Toolbox()

    def initIndividual(icls, content):
        return icls(content)
    def initPopulation(pcls, ind_init, individuals):
        return pcls(ind_init(c) for c in individuals)
    
    toolbox.register("individual", tools.initIterate, creator.Individual, generateRandomArchitecture)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_seed", initPopulation, list, toolbox.individual_guess, seed_models)
    toolbox.register("mate", crossover_one_point)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", evaluate)

    random_population = toolbox.population(n=populationSize)
    seed_population = toolbox.population_seed()
    population = seed_population + random_population
    CXPB, MUTPB, NGEN = 0.7, 0.2, generations

    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        # print(ind)
        # print(fit)
        ind.fitness.values = (fit,)
    
    for gen in range(NGEN):
        # print("Generation:", gen)
        offspring = toolbox.select(population, len(population) - eliteSize)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            # print(ind)
            # print(fit)
            ind.fitness.values = (fit,)

        elites = toolbox.select(population, eliteSize)

        population[:] = elites + offspring

    return tools.selBest(population, 1)[0]


def getData():
    data = np.array(mackey_glass(n_timesteps=5000))
    train = data[:2000]
    val = data[2000:3000]
    test = data[3000:]
    trainX = train[:-1]
    trainY = train[1:]
    valX = val[:-1]
    valY = val[1:]
    testX = test[:-1]
    testY = test[1:]
    return trainX, trainY, valX, valY, testX, testY
trainX, trainY, valX, valY, testX, testY = getData()

deepCascade = {
    'nodes': [
        {'type': 'Input', 'params': {}},
        {'type': 'NVAR', 'params': {'delay': 8, 'order': 3, 'strides': 4}},
        {'type': 'Reservoir', 'params': {'units': 1000, 'sr': 0.8, 'rc_connectivity': 0.01, 'noise_in': 1e-10}},
        {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 1e-05}},
        {'type': 'NVAR', 'params': {'delay': 8, 'order': 3, 'strides': 4}},
        {'type': 'Reservoir', 'params': {'units': 1000, 'sr': 0.8, 'rc_connectivity': 0.01, 'noise_in': 1e-10}},
        {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 1e-05}}
    ],
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
}

errors = []
for i in range(19):
    nvarModelArchitecture = {'nodes': [{'type': 'Input', 'params': {}}, {'type': 'NVAR', 'params': {'delay': 5, 'order': 2, 'strides': 1}}, {'type': 'Ridge', 'params': {'output_dim': 1, 'ridge': 1e-05}}], 'edges': [[0, 1], [1, 2]]}
    best_architecture = runGA(10, 5, [nvarModelArchitecture, deepCascade])
    best_model = constructModel(best_architecture)
    performance = evaluateModel(best_model, np.concatenate((trainX, valX), axis=0), np.concatenate((trainY, valY), axis=0), testX, testY)
    print("Performance", performance)
    errors.append(performance)

print(np.array(errors).mean(), np.array(errors).std())