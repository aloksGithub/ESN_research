import reservoirpy as rpy
from reservoirpy.nodes import (Reservoir, IPReservoir, NVAR, RLS, Input)
from NAS.Ridge_parallel import Ridge
from NAS.LMS_serializable import LMS
from reservoirpy.observables import nrmse
import numpy as np
import random
import math
import warnings
from NAS.memory_estimator import estimateMemory
import time
import networkx as nx
import random
warnings.filterwarnings("ignore")

rpy.verbosity(0)
globalFitnesses = []

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
        self.finalLayer = Reservoir(600, sr=0.8, rc_connectivity=0.01, noise_in=1e-10) >> Ridge(output_dim=1, ridge=1e-5)
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
        # "fb_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False}
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
        # "fb_connectivity": {"lower": 0.05, "upper": 0.5, "intOnly": False}
    },
    "NVAR": {
        "delay": {"lower": 1, "upper": 5, "intOnly": True},
        "order": {"lower": 1, "upper": 5, "intOnly": True},
        "strides": {"lower": 1, "upper": 5, "intOnly": True}
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

def generateRandomNodeParams(nodeType, output_dim):
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

def isValidArchitecture(architecture, sampleInput, sampleOutput, memoryLimit, timeLimit, isAutoRegressive=False):
    ipExists = False
    forceExists = False
    for i, node in enumerate(architecture["nodes"]):
        if node["type"]=="IPReservoir":
            ipExists = True
        if node["type"]=="LMS" or node["type"]=="RLS":
            forceExists = True
    if ipExists and forceExists:
        return False
    memoryEstimate = estimateMemory(architecture, len(sampleInput))
    if memoryEstimate>memoryLimit:
        return False
    
    try:
        start = time.time()
        error = evaluateArchitecture(
            architecture, sampleInput, sampleOutput, sampleInput, sampleOutput
        ) if not isAutoRegressive else evaluateArchitectureAutoRegressive(
            architecture, sampleInput, sampleOutput, sampleInput, sampleOutput
        )
        timeTaken1 = time.time() - start
        if timeTaken1*1.1>timeLimit:
            return False
        if np.isinf(error) or np.isnan(error):
            return False
    except:
        return False
    return True

def createRandomGraph(numNodes):
    while True:
        G = nx.gnp_random_graph(numNodes - 2, 0.7, directed=True)
        DAG = nx.DiGraph([(u, v, {'weight':random.randint(-10, 10)}) for (u, v) in G.edges() if u<v])
        if len(DAG.nodes)==numNodes-2:
            break
    
    sources = []
    sinks = []
    for node in DAG.nodes:
        hasPredecessor = False
        for otherNode in DAG.nodes:
            if DAG.has_predecessor(node, otherNode):
                hasPredecessor = True
        if not hasPredecessor:
            sources.append(node)
            
        hasSuccessor = False
        for otherNode in DAG.nodes:
            if DAG.has_successor(node, otherNode):
                hasSuccessor = True
        if not hasSuccessor:
            sinks.append(node)

    edges = [[0, source+1] for source in sources] + [[e1 + 1, e2 + 1] for (e1, e2) in DAG.edges] + [[sink+1, numNodes-1] for sink in sinks]
    return edges

def generateRandomArchitecture(inputDim, outputDim, sampleInput, sampleOutput, memoryLimit=4*1024, timeLimit=180):
    num_nodes = random.randint(4, 7)
    nodes = [
        {"type": "Input", "params": {"input_dim": inputDim}}
    ]
    for i in range(num_nodes-1):
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

        node_params = generateRandomNodeParams(node_type, outputDim)
        nodes.append({"type": node_type, "params": node_params})

    # Adding the readout node
    ipExists = False
    for node in nodes:
        if node["type"]=="IPReservoir":
            ipExists = True
    if ipExists:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)}
        ]
    else:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)},
            {"type": "LMS", "params": generateRandomNodeParams("LMS", outputDim)},
            {"type": "RLS", "params": generateRandomNodeParams("RLS", outputDim)}
        ]
    nodes.append(random.choice(readouts))

    # Create a graph with num_nodes + 1 nodes (+1 is for the input node)
    edges = createRandomGraph(num_nodes+1)

    architecture = {"nodes": nodes, "edges": edges}

    if isValidArchitecture(architecture, sampleInput, sampleOutput, memoryLimit, timeLimit):
        return architecture
    else:
        return generateRandomArchitecture(inputDim, outputDim, sampleInput, sampleOutput, memoryLimit, timeLimit)


def generateRandomArchitectureOld(inputDim, outputDim, sampleInput, sampleOutput, memoryLimit=4*1024, timeLimit=180):
    num_nodes = random.randint(2, 6)

    nodes = [
        {"type": "Input", "params": {"input_dim": inputDim}}
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

        node_params = generateRandomNodeParams(node_type, outputDim)
        nodes.append({"type": node_type, "params": node_params})

    edges = []
    connected_nodes = {0}  # start with the first node being "connected"
    
    for i in range(1, len(nodes)):
        while (True):
            source = random.choice([node for node in list(connected_nodes) if node != i])
            if (nodes[source]['type']=="Reservoir" or nodes[source]['type']=="IPReservoir" or nodes[source]['type']=="NVAR") and nodes[i]["type"]=="NVAR":
                continue
            if nodes[source]['type']=="IPReservoir" and (nodes[i]["type"]=="RLS" or nodes[i]["type"]=="LMS"):
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
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)}
        ]
    else:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)},
            {"type": "LMS", "params": generateRandomNodeParams("LMS", outputDim)},
            {"type": "RLS", "params": generateRandomNodeParams("RLS", outputDim)}
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

    if isValidArchitecture(architecture, sampleInput, sampleOutput, memoryLimit, timeLimit):
        return architecture
    else:
        return generateRandomArchitectureOld(inputDim, outputDim, sampleInput, sampleOutput, memoryLimit, timeLimit)
    
  
def evaluateArchitecture(individual, trainX, trainY, valX, valY, errorMetrics=[nrmse], defaultErrors=[np.inf]):
    """
    Instantiate random models using given architecture, then train and evaluate them
    on one step ahead prediction using errorMetrics on valX and valY.
    """

    model = constructModel(individual)
    model = trainModel(model, trainX, trainY)
    preds = runModel(model, valX)
    modelErrors = [metric(valY, preds) for metric in errorMetrics]
        
    return modelErrors[0]
  
def evaluateArchitectureAutoRegressive(individual, trainX, trainY, valX, valY, errorMetrics=[nrmse], defaultErrors=[np.inf]):
    """
    Instantiate random models using given architecture, then train and evaluate them
    on one step ahead prediction using errorMetrics on valX and valY.
    """

    model = constructModel(individual)
    model = trainModel(model, trainX, trainY)
    prevOutput = valX[0]
    preds = []
    for _ in range(len(valX)):
        pred = runModel(model, prevOutput)
        prevOutput = pred
        preds.append(pred[0])
    preds = np.array(preds)
    modelErrors = [metric(valY, preds) for metric in errorMetrics]
        
    return modelErrors[0]

def constructModel(architecture):
    nodes = [nodeConstructors[nodeData['type']](**nodeData['params']) for nodeData in architecture['nodes']]

    # Start with the first connection
    model = nodes[architecture['edges'][0][0]] >> nodes[architecture['edges'][0][1]]

    # Continue with the rest of the edges
    for edge in architecture['edges'][1:]:
        model = model & (nodes[edge[0]] >> nodes[edge[1]])

    return model
    
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
    
def smape(yTrue, preds):
    tmp = 2 * np.abs(preds - yTrue) / (np.abs(yTrue) + np.abs(preds))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)

def printArchitecture(architecture):
    for node in architecture["nodes"]:
        print("{}({})".format(node["type"], list(node["params"].values())))
    print(architecture["edges"])

def printArchitectures(architectures):
    [printArchitecture(architecture) for architecture in architectures]