import networkx as nx
import matplotlib.pyplot as plt
from NAS.ESN_NAS import ESN_NAS
from NAS.error_metrics import nrmse, r_squared
from utils import readSavedExperiment
import numpy as np

def visualize_esn_architecture(architecture):
    lastNode = len(architecture['nodes']) - 1
    architecture['nodes'].append({'type': 'Output'})
    architecture['edges'].append([lastNode, lastNode+1])
    G = nx.DiGraph()
    
    # Add nodes
    for i, node in enumerate(architecture['nodes']):
        G.add_node(i, **node)
    
    G.add_edges_from(architecture['edges'])
    
    # Compute node levels (x-coordinates)
    levels = {}
    def dfs(node, level):
        if node not in levels:
            levels[node] = level
            for neighbor in G.neighbors(node):
                dfs(neighbor, level + 1)
    
    # Start DFS from input nodes (nodes with in_degree 0)
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    for input_node in input_nodes:
        dfs(input_node, 0)
    
    # Compute node positions
    pos = {}
    level_counts = {}
    for node, level in levels.items():
        if level not in level_counts:
            level_counts[level] = 0
        y = level_counts[level]
        level_counts[level] += 1
        pos[node] = (level, -y)  # Negative y to have top-to-bottom layout
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                           arrowsize=20, arrowstyle='wedge', connectionstyle='arc3,rad=0')
    
    # Add labels (only node type)
    labels = {i: data['type'] for i, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")
    
    # Set title and remove axis
    plt.title("Echo State Network Architecture", fontsize=16)
    plt.axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def findBestGaArchitecture(ga: ESN_NAS):
    errors = [errors[0] for errors in ga.fitnesses]
    gaBestError = min(errors)
    gaBestErrorIndex = errors.index(gaBestError)
    gaBestModel = ga.architectures[gaBestErrorIndex]
    return gaBestModel, gaBestError

def findBestGasArchitecture(gas: list[ESN_NAS]):
    bestError = np.inf
    bestArchitecture = None
    for ga in gas:
        architecture, error = findBestGaArchitecture(ga)
        if error<bestError:
            bestArchitecture = architecture
    return bestArchitecture

if __name__ == "__main__":
    gas = [readSavedExperiment(f'./backup_50/laser/backup_{i}.obj') for i in range(5)]
    architecture = findBestGasArchitecture(gas)
    architectures = [findBestGaArchitecture(ga)[0] for ga in gas]
    print(architectures[0])
    visualize_esn_architecture(architectures[0])
    # [visualize_esn_architecture(architecture) for architecture in architectures]