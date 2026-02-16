from reservoirpy.jax.nodes import Reservoir, NVAR, RLS, LMS, Ridge, Input
from src.nodes.jax.ipreservoir import IPReservoir
from reservoirpy.observables import nrmse
import numpy as np
import random
import networkx as nx
import copy
import traceback
from .memory_estimator import estimateMemory

nodeConstructors = {
    "Input": Input,
    "Reservoir": Reservoir,
    "IPReservoir": IPReservoir,
    "NVAR": NVAR,
    "Ridge": Ridge,
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
        "strides": {"lower": 1, "upper": 5, "intOnly": True},
    },
    "Ridge": {"ridge": {"lower": 0, "upper": 0.0001, "intOnly": False}},
}


def generateRandomNodeParams(nodeType, output_dim):
    params = {}
    if nodeType == "Ridge" or nodeType == "LMS" or nodeType == "RLS":
        params["output_dim"] = output_dim
    parameterRanges = nodeParameterRanges[nodeType]
    for parameterName in parameterRanges:
        parameterRange = parameterRanges[parameterName]
        if parameterRange["intOnly"]:
            params[parameterName] = random.randint(
                parameterRange["lower"], parameterRange["upper"]
            )
        else:
            params[parameterName] = (
                random.random() * (parameterRange["upper"] - parameterRange["lower"])
                + parameterRange["lower"]
            )
    return params


def isValidArchitecture(
    architecture,
    sampleInput,
    memoryLimit,
):
    ipExists = False
    forceExists = False
    for i, node in enumerate(architecture["nodes"]):
        if node["type"] == "IPReservoir":
            ipExists = True
        if node["type"] == "LMS" or node["type"] == "RLS":
            forceExists = True
    if ipExists and forceExists:
        return False
    memoryEstimate = estimateMemory(architecture, len(sampleInput))
    if memoryEstimate > memoryLimit:
        return False
    
    # try:
    #     model = constructModel(architecture)
    #     runModel(model, sampleInput[:1])
    #     for node in model.nodes:
    #         if "Ridge" in node.name and node.input_dim > 2200:
    #             return False
    #         if "RLS" in node.name and node.input_dim > 400:
    #             return False
    #         if "LMS" in node.name and node.input_dim > 3000:
    #             return False
    # except Exception:
    #     return False
    return True


def createRandomGraph(numNodes):
    while True:
        G = nx.gnp_random_graph(numNodes - 2, 0.7, directed=True)
        DAG = nx.DiGraph(
            [
                (u, v, {"weight": random.randint(-10, 10)})
                for (u, v) in G.edges()
                if u < v
            ]
        )
        if len(DAG.nodes) == numNodes - 2:
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

    edges = (
        [[0, source + 1] for source in sources]
        + [[e1 + 1, e2 + 1] for (e1, e2) in DAG.edges]
        + [[sink + 1, numNodes - 1] for sink in sinks]
    )
    return edges


def generateRandomArchitecture(
    outputDim, sampleInput, memoryLimit=4 * 1024
):
    num_nodes = random.randint(4, 7)
    nodes = [{"type": "Input", "params": {"input_dim": sampleInput.shape[-1]}}]
    for i in range(num_nodes - 1):
        available_node_types = list(nodeConstructors.keys())
        if i == 0:
            available_node_types.remove("Ridge")
        available_node_types.remove("Input")
        for node in nodes:
            if node["type"] == "IPReservoir":
                if "LMS" in available_node_types:
                    available_node_types.remove("LMS")
                if "RLS" in available_node_types:
                    available_node_types.remove("RLS")
            if (
                node["type"] == "LMS" or node["type"] == "RLS"
            ) and "IPReservoir" in available_node_types:
                available_node_types.remove("IPReservoir")

        node_type = random.choice(available_node_types)

        node_params = generateRandomNodeParams(node_type, outputDim)
        nodes.append({"type": node_type, "params": node_params})

    # Adding the readout node
    ipExists = False
    for node in nodes:
        if node["type"] == "IPReservoir":
            ipExists = True
    if ipExists:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)}
        ]
    else:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)},
        ]
    nodes.append(random.choice(readouts))

    # Create a graph with num_nodes + 1 nodes (+1 is for the input node)
    edges = createRandomGraph(num_nodes + 1)

    architecture = {"nodes": nodes, "edges": edges}

    if isValidArchitecture(
        architecture, sampleInput, memoryLimit
    ):
        return architecture
    else:
        return generateRandomArchitecture(
            outputDim, sampleInput, memoryLimit
        )

def generateRandomArchitectureOld(
    outputDim, sampleInput, memoryLimit=4 * 1024, num_nodes_range=(2, 4)
):
    num_nodes = random.randint(num_nodes_range[0], num_nodes_range[1])

    nodes = [{"type": "Input", "params": {"input_dim": sampleInput.shape[-1]}}]

    for i in range(num_nodes):
        available_node_types = list(nodeConstructors.keys())
        if i == 0:
            available_node_types.remove("Ridge")
        available_node_types.remove("Input")
        for node in nodes:
            if node["type"] == "IPReservoir":
                if "LMS" in available_node_types:
                    available_node_types.remove("LMS")
                if "RLS" in available_node_types:
                    available_node_types.remove("RLS")
            if (
                node["type"] == "LMS" or node["type"] == "RLS"
            ) and "IPReservoir" in available_node_types:
                available_node_types.remove("IPReservoir")

        node_type = random.choice(available_node_types)

        node_params = generateRandomNodeParams(node_type, outputDim)
        nodes.append({"type": node_type, "params": node_params})

    edges = []
    connected_nodes = {0}  # start with the first node being "connected"

    for i in range(1, len(nodes)):
        while True:
            source = random.choice(
                [node for node in list(connected_nodes) if node != i]
            )
            if (
                nodes[source]["type"] == "Reservoir"
                or nodes[source]["type"] == "IPReservoir"
                or nodes[source]["type"] == "NVAR"
            ) and nodes[i]["type"] == "NVAR":
                continue
            if nodes[source]["type"] == "IPReservoir" and (
                nodes[i]["type"] == "RLS" or nodes[i]["type"] == "LMS"
            ):
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
        if node["type"] == "IPReservoir":
            ipExists = True
    if ipExists:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)}
        ]
    else:
        readouts = [
            {"type": "Ridge", "params": generateRandomNodeParams("Ridge", outputDim)},
        ]
    nodes.append(random.choice(readouts))

    final_node_index = len(nodes) - 1
    for i in range(final_node_index):
        isOutputNode = True
        for edge in edges:
            if edge[0] == i:
                isOutputNode = False
        if isOutputNode:
            edges.append([i, final_node_index])

    architecture = {"nodes": nodes, "edges": edges}
    if isValidArchitecture(
        architecture, sampleInput, memoryLimit
    ):
        return architecture
    else:
        return generateRandomArchitectureOld(
            outputDim, sampleInput, memoryLimit
        )

def evaluateArchitecture(
    individual,
    trainX,
    trainY,
    valX,
    valY,
    numEvals=1,
    errorMetrics=[nrmse],
    defaultErrors=[10000],
    isAutoRegressive=False,
):
    """Standalone function executed in a worker process.

    Returns ``(individual, bestErrors, bestModel_or_None)``.  The model
    object is returned *only if it can be pickled*; otherwise ``None`` is
    sent back.
    """
    import os
    import jax
    jax.config.update("jax_enable_x64", True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    errors = []
    models = []

    try:
        for _ in range(numEvals):
            try:
                model = constructModel(individual)
                model = trainModel(model, trainX, trainY)
                model_copy = copy.deepcopy(model)

                if isAutoRegressive:
                    prevOutput = valX[0]
                    preds = []
                    for _ in range(len(valX)):
                        pred = runModel(model, prevOutput)
                        prevOutput = pred
                        preds.append(pred[0])
                    preds = np.array(preds)
                else:
                    preds = runModel(model, valX)

                modelErrors = [metric(valY, preds) for metric in errorMetrics]
                errors.append(modelErrors)
                models.append(model_copy)
            except Exception:
                errors.append(defaultErrors)
                models.append(None)

        # Select best attempt w.r.t. the first error metric
        error0 = [e[0] for e in errors]
        best_idx = error0.index(min(error0)) if defaultErrors[0] != 0 else error0.index(max(error0))

        return individual, errors[best_idx], models[best_idx]

    except Exception:
        # Any unexpected failure -> signal default error
        return individual, defaultErrors, None

def constructModel(architecture):
    nodes = []
    node_type_counts = {}  # Track how many of each type for unique naming
    
    for nodeData in architecture["nodes"]:
        node_type = nodeData["type"]
        node_params = nodeData["params"].copy()
        
        # In reservoirpy v0.4.1+, Input node doesn't accept input_dim as constructor arg
        # It's inferred during initialization from data
        if node_type == "Input":
            node_params.pop("input_dim", None)
        
        # Assign unique names to nodes (required in v0.4 for models with multiple trainable nodes)
        if node_type not in node_type_counts:
            node_type_counts[node_type] = 0
        node_type_counts[node_type] += 1
        node_params["name"] = f"{node_type}_{node_type_counts[node_type]}"
        
        nodes.append(nodeConstructors[node_type](**node_params))

    # Start with the first connection
    model = nodes[architecture["edges"][0][0]] >> nodes[architecture["edges"][0][1]]

    # Continue with the rest of the edges
    for edge in architecture["edges"][1:]:
        model = model & (nodes[edge[0]] >> nodes[edge[1]])

    return model


def _get_node_type_name(node):
    """Get a string identifier for a node's type."""
    # In v0.4.1, node.name might be None, so use class name as fallback
    if node.name is not None:
        return node.name
    return type(node).__name__


def trainModel(model, trainX, trainY, warmup=100):
    model.fit(trainX, trainY, warmup=warmup)
    return model


def runModel(model, x):
    node_names = [_get_node_type_name(node) for node in model.nodes]
    notLastNodes = []
    for edge in model.edges:
        edge_name = _get_node_type_name(edge[0])
        if edge_name not in notLastNodes:
            notLastNodes.append(edge_name)
    
    output_nodes = list(set(node_names) - set(notLastNodes))
    nodePreds = model.run(x)
    
    # Check if nodePreds is an array (numpy or JAX) or a dict
    # In v0.4.1 with JAX, model.run() may return a JAX array directly
    if isinstance(nodePreds, dict):
        return nodePreds[output_nodes[-1]] if output_nodes else list(nodePreds.values())[-1]
    else:
        # It's an array (numpy or JAX), return it directly
        return np.asarray(nodePreds)


def smape(yTrue, preds):
    tmp = 2 * np.abs(preds - yTrue) / (np.abs(yTrue) + np.abs(preds))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0:  # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)


def printArchitecture(architecture):
    for node in architecture["nodes"]:
        print("{}({})".format(node["type"], list(node["params"].values())))
    print(architecture["edges"])


def printArchitectures(architectures):
    [printArchitecture(architecture) for architecture in architectures]
