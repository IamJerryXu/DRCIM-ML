import numpy as np
import networkx as nx
import random
from tqdm import tqdm

def ICm(S, G, p=0.1, mc=1000):
    """ 独立级联传播模型 """
    l = len(G)
    spread = []
    for i in range(mc):
        count = {}
        final_activated = set(S)
        for g in G:
            activated = set(S)  # 初始激活的节点集合
            spreader = set(S)  # 当前传播节点

            while spreader:  # 当还有传播节点时
                new_spreader = set()
                for node in spreader:
                    neighbors = list(g.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in activated:  # 只处理未激活的节点
                            if random.random() < p:  # 以概率决定激活
                                new_spreader.add(neighbor)
                                activated.add(neighbor)
                                '''
                                if neighbor not in count.keys():
                                    count[neighbor] = 1
                                else:
                                    count[neighbor] += 1
                                    if count[neighbor] >= lmin:
                                        final_activated.add(neighbor)
                                '''
                                count[neighbor] = 1
                spreader = new_spreader
        spread.append(len(count)+len(S))
    return np.mean(spread)
def greedy_ic_m(G, k):
    node_num = G[0].number_of_nodes()
    marg_gain = [ICm([node], G) for node in range(node_num)]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(node_num), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q = Q[1:]

    for _ in tqdm(range(k - 1), leave=False):
        max_gain = (0, 0)
        max_index = 0
        for i in range(len(Q)):

            Q[i] =(Q[i][0], ICm(S + [Q[i][0]], G))
            if Q[i][1] >= max_gain[1]:
                max_gain = Q[i]
                max_index = i
        Q = Q[0:max_index] + Q[max_index + 1:]

        # Select the next node
        spread = max_gain[1]
        S.append(max_gain[0])
        SPREAD.append(spread)

    return S, spread

def celf_ic_m(G, k,p=0.1,mc=1000):
    node_num = G[0].number_of_nodes()
    marg_gain = [ICm([node], G,p,mc) for node in range(node_num)]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(node_num), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q = Q[1:]

    for _ in tqdm(range(k-1),leave=False):

        check, node_lookup = False, 0

        while not check:

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, ICm(S + [current], G,p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)


        # Remove the selected node from the list
        Q = Q[1:]

    return S, spread

def celf_optimize(G,seeds,p=0.1,mc=1000):
    marg_gain = [ICm([node], G,p,mc) for node in seeds]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(seeds, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q = Q[1:]

    for _ in tqdm(range(len(seeds)-1),leave=False):

        check, node_lookup = False, 0

        while not check:

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, ICm(S + [current], G,p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)


        # Remove the selected node from the list
        Q = Q[1:]

    return S, SPREAD


def degree(G):
    # 计算每个节点的度
    degree_list = []
    for g in G:
        degree_list.append(dict(nx.degree(g)))
    for i in range(G[0].number_of_nodes()):
        for j in range(1, len(degree_list)):
            degree_list[0][i]+=degree_list[j][i]

    # 根据度从高到低排序
    sorted_degrees = sorted(degree_list[0].items(), key=lambda x: x[1], reverse=True)
    sorted_node = [node for node, degree in sorted_degrees]
    return sorted_node

def pagerank(G):
    pagerank_list = []
    for g in G:
        pagerank_list.append(nx.pagerank(g))
    for i in range(G[0].number_of_nodes()):
        for j in range(1, len(pagerank_list)):
            pagerank_list[0][i] += pagerank_list[j][i]

    # 根据PageRank从高到低排序，并提取排序后的节点索引列表
    sorted_pagerank = sorted(pagerank_list[0].items(), key=lambda x: x[1], reverse=True)
    sorted_node = [node for node, rank in sorted_pagerank]
    return sorted_node

def tongji(name, f):
    # 转换为numpy数组以便计算
    data = f.cpu().detach().numpy()
    shape = data.shape
    n = 1
    for j in shape:
        n*=j
    # 计算基本统计量
    total_stats = {
        'name':name,
        'shape': data.shape,  # 数据形状
        'mean': round(np.mean(data),5),  # 均值
        'std': round(np.std(data),5),  # 标准差
        'min': round(np.min(data),5),  # 最小值
        'max': round(np.max(data),5),  # 最大值
        'median': round(np.median(data),5),  # 中位数
        'zero_ratio': round(np.sum(data == 0) / n,5)
    }
    print(total_stats)
    print('')
    if len(data) > 1 and len(data.shape)>2:
        l_mean = []
        l_std = []
        l_min = []
        l_max = []
        l_median = []
        for i in range(len(data)):
            l_mean.append(round(np.mean(data[i]),5))
            l_std.append(round(np.std(data[i]),5))
            l_min.append(round(np.min(data[i]),5))
            l_max.append(round(np.max(data[i]),5))
            l_median.append(round(np.median(data[i]),5))
        layer_stats = {
            'l_mean': l_mean,
            'l_std': l_std,
            'l_min': l_min,
            'l_max': l_max,
            'l_median': l_median
        }
        print(layer_stats)
        print('')
    return