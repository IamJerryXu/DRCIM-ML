import networkx as nx
import random
import numpy as np
import edges_generation_fun as gf

def generate_network_edges(G, avg_degree, graph_type):
    """
    为给定的网络生成连接

    参数:
    G: NetworkX图对象，包含节点但没有边
    avg_degree: float, 预期平均度
        高密度 (平均度 >8 && <12)
        中等密度 (平均度 3~8)
        低密度 (平均度 1~3)
    graph_type: string, 网络类型
        'small_world' - 小世界类型,,
        'scale_free' - 无标度类型
        'hybrid' - 混合类型
        3 - 社区聚集类型

    返回:
    NetworkX图对象，已添加相应的边
    """
    n = G.number_of_nodes()

    # 计算目标边数
    target_edges = int(avg_degree * n / 2)

    # 根据网络类型生成连接
    if graph_type == 'small_world':  # 小世界类型
        G = gf.generate_sw(G, target_edges)
    elif graph_type == 'scale_free':  # 无标度类型
        G = gf.generate_ba(G, target_edges)
    elif graph_type == 'hybrid':  # 社区聚集类型
        G = gf.generate_hybrid(G, target_edges)
    elif graph_type == 'er':
        G = gf.generate_er(G, target_edges)

    return G