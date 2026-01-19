import math
from copy import deepcopy

import numpy as np
from collections import Counter


# spatial matrix 抽样实现

def alias_setup(probs):
    """Alias Sampling"""
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)
    smaller = []
    larger = []

    # 将概率分成两组，一组概率大于1，一组概率小于1
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # 使用贪心算法，将概率 < 1 的填满
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < kk:
        return kk
    else:
        return J[kk]


def get_alias_edge(G, src, dst, args):
    p = args.p_f
    q = args.q_f

    unnormalized_probs = []

    # 论文 3.2.2 算法，计算每条边的转移概率
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
        elif G.has_edge(dst_nbr, src):
            unnormalized_probs.append(G[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
    # 归一化转移概率
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    # 执行 Alias Sampling
    return alias_setup(normalized_probs)


def node2vec_walk(G, walk_length: int, start_node: int, alias_nodes, alias_edges):
    """根据表生成随机游走序列"""
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next_node = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                walk.append(next_node)
        else:
            break
    return walk


def process_node(node, G, args, alias_nodes, alias_edges):
    walks = []
    for _ in range(args.num_walks_f):
        walks.append(node2vec_walk(G, args.walk_length_f, node, alias_nodes, alias_edges))
    return walks


def create_walks(G, f_path, args, is_weight: bool = False):
    """采样得到所有随机游走序列"""

    walks = list()
    alias_nodes = {}
    alias_edges = {}
    if not is_weight:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    nodes = sorted([node for node in G.nodes()])
    assert nodes == [i for i in range(len(G.nodes()))]
    for node in nodes:
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    for edge in G.edges():
        alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], args)
        alias_edges[(edge[1], edge[0])] = get_alias_edge(G, edge[1], edge[0], args)
    w_length = int(math.log(len(G.nodes)) / math.log(sum(dict(G.degree()).values()) / len(G)))
    for node in nodes:
        for _ in range(args.num_walks_f):
            walks.append(node2vec_walk(G, max(w_length, args.walk_length_f), node, alias_nodes, alias_edges))
    for i in range(len(walks)):
        if len(walks[i]) == 1:
            walks[i] = walks[i] * max(w_length, args.walk_length_f)
    walks = np.array(walks)
    walks = np.reshape(walks, (len(nodes), args.num_walks_f, max(w_length, args.walk_length_f)))
    np.save(f_path, np.array(walks))
    return walks


def get_count_by_counter(walk: list):
    count = Counter(walk)
    return count.items()


def create_frequency(nw, nwf):
    """根据随机游走结果统计每个节点与其他节点的相关性"""
    walks = np.load(nw)
    walks_size = walks.shape
    baisc_key = [i for i in range(walks_size[0])]
    baisc_value = np.zeros(walks_size[0]).tolist()
    baisc_dict = dict(zip(baisc_key, baisc_value))
    frequency = list()
    temp_walks = list()
    for i in range(walks_size[0]):
        temp_walks = walks[i].reshape(walks_size[1] * walks_size[2])
        temp_dict = get_count_by_counter(temp_walks)
        frequency.append(deepcopy(baisc_dict))
        frequency[i].update(temp_dict)
        frequency[i] = np.array(list(frequency[i].values())) / np.array(list(frequency[i].values())).max()
    frequency = np.array(frequency)
    np.save(nwf, frequency)
