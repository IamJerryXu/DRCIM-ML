import numpy as np
import networkx as nx
from generate_edges import generate_network_edges
def generate_network(nodes, layer, layer_nodes_cover, avg_degree, graph_type):
    """
        根据策略构建多层网络
        参数:
            多层网络生成参数
        返回:
            单个构建好的多层网络
    """
    nodes_num = np.random.randint(nodes[0], nodes[1])
    nodes_list = list(range(nodes_num))
    nodes_layer = []
    unselected_nodes = set(nodes_list)
    layer_num = np.random.randint(layer[0], layer[1])
    Gm = []

    for l in range(layer_num):
        cover_rate = np.random.uniform(layer_nodes_cover[0], layer_nodes_cover[1])
        nodes_layer.append(list(np.random.choice(nodes_list, int(nodes_num * cover_rate), replace=False)))
        unselected_nodes = unselected_nodes - set(nodes_layer[-1])
    for node in unselected_nodes:
        i = np.random.randint(0, layer_num)
        nodes_layer[i].append(node)
    for l in range(layer_num):
        G = nx.Graph()
        G.add_nodes_from(sorted(nodes_layer[l]))
        ad = np.random.uniform(avg_degree[0], avg_degree[1])
        G = generate_network_edges(G, ad, graph_type)
        Gm.append(G)

    edges_num = sum([G.number_of_edges() for G in Gm])
    return Gm,nodes_num,layer_num,edges_num,graph_type