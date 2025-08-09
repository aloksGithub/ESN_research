import random
import time
import sys
import os
import numpy as np
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.utils import constructModel, evaluateArchitecture, generateRandomArchitectureOld, generateRandomNodeParams, nodeConstructors, runModel
from NAS.parallel_processing import executeParallelBatch
from utils import getDataMGS
from NAS.memory_estimator import estimateMemory

trainX, trainY, valX, valY, testX, testY = getDataMGS()

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

    model = constructModel(architecture)
    runModel(model, sampleInput[:1])
    for node in model.nodes:
        if "Ridge" in node.name and node.input_dim >1200:
            return False
        if "RLS" in node.name and node.input_dim >400:
            return False
        if "LMS" in node.name and node.input_dim >3000:
            return False
    return True

def generateRandomArchitectureOld(
    inputDim, outputDim, sampleInput, sampleOutput, memoryLimit=4 * 1024, timeLimit=180
):
    num_nodes = random.randint(2, 4)

    nodes = [{"type": "Input", "params": {"input_dim": inputDim}}]

    for i in range(num_nodes):
        available_node_types = list(nodeConstructors.keys())
        if i == 0:
            available_node_types.remove("LMS")
            available_node_types.remove("RLS")
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
            {"type": "LMS", "params": generateRandomNodeParams("LMS", outputDim)},
            {"type": "RLS", "params": generateRandomNodeParams("RLS", outputDim)},
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
    if isValidArchitecture(architecture, sampleInput, memoryLimit):
        return architecture
    else:
        return generateRandomArchitectureOld(
            inputDim, outputDim, sampleInput, sampleOutput, memoryLimit, timeLimit
        )

def eval_func(architecture):
    start = time.time()
    _, _, model = evaluateArchitecture(architecture, trainX, trainY, valX, valY, 3)
    print(time.time() - start)
    return model, time.time() - start

def main():
    architectures = []
    for _ in range(20):
        architectures.append(generateRandomArchitectureOld(trainX.shape[1], trainY.shape[1], trainX, trainY, 756, 60))
    results = executeParallelBatch(eval_func, [(individual,) for individual in architectures], 20, 600 * 3)
    for result in results:
        if result is not None:
            model, timeTaken = result
            print("===============================================")
            print(timeTaken)
            if model is not None:
                for node in model.nodes:
                    print(node)

if __name__ == "__main__":
    main()