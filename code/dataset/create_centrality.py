import math

import networkx as nx
import numpy as np
from warnings import simplefilter

simplefilter(action="ignore",category=FutureWarning)
# 1.中心性特征矩阵---DC计算
def create_DC(G):
    """计算归一化的节点中心性

    Args:
        G (networkx.Graph): 输入图

    Returns:
        list: 节点的归一化list
    """
    # adj_matrix = nx.adjacency_matrix(G).todense()
    # adj_matrix = np.array(adj_matrix, dtype=np.float32).reshape(len(adj_matrix), len(adj_matrix))
    # dc = np.array(adj_matrix.sum(axis=0))

    dc = []
    for node in range(G.number_of_nodes()):
        dc.append(G.degree(node))
    return np.array(dc)


# 1.中心性特征矩阵---HI计算
def create_HI(G):
    """计算归一化的节点H-Index

    Args:
        G (networkx.Graph): 输入图

    Returns:
        list: 节点的归一化list
    """
    H_index = []
    for node in range(G.number_of_nodes()):
        citations = []
        for neighbor in G.neighbors(node):
            citations.append(G.degree(neighbor))
        citations.sort(reverse=True)
        h_index = 0
        for i, citation in enumerate(citations):
            if citation >= i + 1:
                h_index = i + 1
            else:
                break
        H_index.append(h_index)

    return np.array(H_index)

# 1.中心性特征矩阵---KS计算
def create_kshell(G):
    """计算节点归一化的节点KShell

    Args:
        G (networkx.Graph): 输入图

    Returns:
        list: 节点的归一化list
    """
    kshell = nx.core_number(G)
    kshell = sorted(kshell.items(), key=lambda x: x[0])
    return np.array([v for _,v in kshell])

def create_pagerank(G):
    pr = nx.pagerank(G)
    pr = sorted(pr.items(), key=lambda x: x[0])
    return np.array([v for _,v in pr])

# 1.中心性特征矩阵---Clustering Coefficient计算
def create_cc(G):
    cc = nx.clustering(G)
    cc = sorted(cc.items(), key=lambda x: x[0])
    return np.array([v for _,v in cc])

def create_degree_centrality(G):
    dc = nx.degree_centrality(G)
    dc = sorted(dc.items(), key=lambda x: x[0])
    return np.array([v for _, v in dc])

def create_closeness(G):
    cc = nx.closeness_centrality(G)
    cc = sorted(cc.items(), key=lambda x: x[0])
    return np.array([v for _, v in cc])

def create_harmonic(G):
    hc = nx.harmonic_centrality(G)
    hc = sorted(hc.items(), key=lambda x: x[0])
    return np.array([v for _, v in hc])

def create_avg_neighbor_degree(G):
    nd = nx.average_neighbor_degree(G)
    nd = sorted(nd.items(), key=lambda x: x[0])
    return np.array([v for _, v in nd])

def create_eigenvector(G):
    try:
        ec = nx.eigenvector_centrality_numpy(G)
    except Exception:
        ec = {n: 0.0 for n in G.nodes()}
    ec = sorted(ec.items(), key=lambda x: x[0])
    return np.array([v for _, v in ec])

# 1.中心性特征矩阵---Node Influence Power计算
def create_nip(G):
    nip = []
    # tran-based 论文中描述的计算方法
    for node in range(G.number_of_nodes()):
        d = G.degree(node)
        cc = nx.clustering(G, node)
        cc = 2/(cc+1)
        neighbors = list(G.neighbors(node))
        s = 0
        if d > 0:
            for i in range(d):
                du = G.degree(neighbors[i])
                ccu = nx.clustering(G, neighbors[i])
                s += du / (ccu+1)
            s = s*cc + d
            nip.append(s)
        else:
            nip.append(d)

    # 原论文描述的公式
    # for node in G.nodes():
    #     d = G.degree(node)
    #     cc = nx.clustering(G, node)
    #     neighbors = list(G.neighbors(node))
    #     if d > 0:
    #         max_d = 0
    #         max_node = 0
    #         for i in range(d):
    #             du = G.degree(neighbors[i])
    #             if du > max_d:
    #                 max_d = du
    #                 max_node = neighbors[i]
    #         neighbors_u = set(G.neighbors(max_node))
    #         nip.append(d-len(neighbors_u & set(neighbors))*(1-cc))
    #     else:
    #         nip.append(d)
    return np.array(nip)

def create_SI(G):
    si = []
    for node in range(G.number_of_nodes()):
        d = G.degree(node)
        s = 0
        if d > 0:
            for n in G.neighbors(node):
                du = G.degree(n)
                s += math.sqrt(du**2+d**2)
            si.append(s/d)
        else:
            si.append(d)
    return np.array(si)

# 1.中心性特征矩阵---综合
def create_centrality(G, f_path, c_num):
    """ 根据需要产生对应的的节点的中心性特征并保存到npy文件中

    Args:
        index:
        f_path:
        G (networkx.Graph): 输入图
        centralities (list): 需要计算的中心性指标

    Returns:
        np.array: 节点中心性
    """
    try:
        c_num_int = int(c_num)
    except Exception:
        c_num_int = 3

    centrality_list = [
        create_DC(G),
        create_HI(G),
        create_kshell(G),
    ]

    if c_num_int >= 6:
        centrality_list.extend([
            create_pagerank(G),
            create_nip(G),
            create_SI(G),
        ])

    if c_num_int >= 12:
        centrality_list.extend([
            create_cc(G),
            create_degree_centrality(G),
            create_closeness(G),
            create_harmonic(G),
            create_avg_neighbor_degree(G),
            create_eigenvector(G),
        ])

    # 补齐或裁剪到指定维度
    if len(centrality_list) < c_num_int:
        pad_num = c_num_int - len(centrality_list)
        zeros = np.zeros_like(centrality_list[0])
        centrality_list.extend([zeros for _ in range(pad_num)])
    elif len(centrality_list) > c_num_int:
        centrality_list = centrality_list[:c_num_int]

    centrality_matrix = np.array(centrality_list).T
    np.save(f_path, centrality_matrix)
    return centrality_matrix
