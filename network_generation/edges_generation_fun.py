
import networkx as nx
import random
from collections import defaultdict
import numpy as np


def generate_sw(G, target_edges):
    """生成小世界网络"""
    n = G.number_of_nodes()
    total_nodes = list(G.nodes())

    # 首先创建一个规则环状网络
    m = target_edges // n  # 每个节点的初始邻居数
    for i in range(n):
        for j in range(1, m + 1):
            G.add_edge(total_nodes[i], total_nodes[(i + j) % n])

    # 随机重连一些边以创建小世界特性
    p = random.uniform(0.2,0.35)  # 重连概率
    edges = list(G.edges())

    for u, v in edges:
        if random.random() < p:
            G.remove_edge(u, v)
            new_v = random.choice([node for node in total_nodes if node != u and not G.has_edge(u, node)])
            G.add_edge(u, new_v)

    #生成一些高度节点
    # num_hubs = n*0.03
    # hubs = random.choice(total_nodes, num_hubs, replace=False)

    # 为这些高度节点添加额外连接（随机选取病种）
    # for hub in hubs:
    #     # 添加4-8个额外连接
    #     num_extra = np.random.randint(4, 8)
    #     for _ in range(num_extra):
    #         target = np.random.randint(0, total_patients)
    #         if target != hub and not G.has_edge(hub, target):
    #             G.add_edge(hub, target)

    # 如果边数还不够，随机添加边
    while G.number_of_edges() < target_edges:
        v = random.choice(list(total_nodes))
        u = random.choice(list(total_nodes))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G


def generate_ba(G, target_edges):
    """生成无标度网络"""
    n = G.number_of_nodes()
    total_nodes = set(G.nodes())
    # 新增节点添加的边数
    m = target_edges // n
    # 初始化：先随机选取一些节点添加少量随机连接
    initial_nodes_count = max(4,m*2)
    initial_nodes = random.sample(list(G.nodes()), initial_nodes_count)
    unconnected_nodes = list(total_nodes - set(initial_nodes))
    degrees = {}
    # 创建初始完全连接的子图
    for i in range(len(initial_nodes)):
        for j in range(i + 1, len(initial_nodes)):
            G.add_edge(initial_nodes[i], initial_nodes[j])
    for node in initial_nodes:
        degrees[node] = G.degree(node)
    total_degree = sum(degrees.values())

    for i in range(n-initial_nodes_count):
        # 随机选一个未连接的节点
        selected_node = random.choice(unconnected_nodes)
        # 选择m个与其连接的节点（不重复，按概率与度成正比）
        j=0
        while j < m:
            r = random.random() * total_degree
            cumulative = 0
            for node, degree in degrees.items():
                cumulative += degree
                if cumulative >= r:
                    u = node
                    break
            # 确保不添加自环和重复边
            if not G.has_edge(u, selected_node):
                G.add_edge(u, selected_node)
                degrees[u] += 1
                total_degree += 2
                j+=1
        degrees[selected_node] = m
        unconnected_nodes.remove(selected_node)


    while G.number_of_edges() < target_edges:
        r1 = random.random()
        if r1 < 0.5:

            v = random.choice(list(total_nodes))

            # 选择另一个节点（按概率与度成正比）
            r = random.random() * total_degree
            cumulative = 0
            for node, degree in degrees.items():
                cumulative += degree
                if cumulative >= r:
                    u = node
                    break

            # 确保不添加自环和重复边
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
        else:
            v = random.choice(list(total_nodes))
            u = random.choice(list(total_nodes))
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    return G

def generate_community(G, target_edges):
    """生成社区结构网络"""
    n = G.number_of_nodes()
    total_nodes = list(G.nodes())
    # 确定社区数量 (2-4个社区)
    num_communities = random.randint(2, min(4, n // 10))
    community_size = n // num_communities

    # 分配节点到社区
    communities = defaultdict(list)
    for i in range(n):
        community_id = i // community_size
        if community_id >= num_communities:
            community_id = num_communities - 1
        communities[community_id].append(i)

    # 计算社区内和社区间的目标边数
    # 社区内连接占70%，社区间连接占30%
    intra_edges = int(target_edges * 0.7)
    inter_edges = target_edges - intra_edges

    # 添加社区内连接 (使用小世界模型)
    for comm_id, nodes in communities.items():
        if len(nodes) < 2:
            continue

        # 计算这个社区应该有多少边
        comm_edges = int(intra_edges * len(nodes) / n)
        subG = nx.Graph()
        subG.add_nodes_from(nodes)
        subG = generate_sw(subG, comm_edges)

        # 将子图的边添加到主图
        for u, v in subG.edges():
            G.add_edge(u, v)

    # 添加社区间连接
    added_inter_edges = 0
    max_attempts = 1000  # 防止无限循环
    attempts = 0

    while added_inter_edges < inter_edges and attempts < max_attempts:
        attempts += 1

        # 随机选择两个不同的社区
        comm1, comm2 = random.sample(list(communities.keys()), 2)

        # 从每个社区随机选择一个节点
        node1 = random.choice(communities[comm1])
        node2 = random.choice(communities[comm2])

        # 如果这两个节点之间没有连接，添加边
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
            added_inter_edges += 1

    # 如果边数还不够，随机添加边
    while G.number_of_edges() < target_edges:
        v = random.choice(list(total_nodes))
        u = random.choice(list(total_nodes))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G

def generate_hybrid(G, target_edges):
    """生成混合类型网络"""

    n = G.number_of_nodes()
    total_nodes = list(G.nodes())

    # 首先创建一个规则环状网络
    m = target_edges // n  # 每个节点的初始邻居数
    for i in range(n):
        for j in range(1, m + 1):
            G.add_edge(total_nodes[i], total_nodes[(i + j) % n])

    # 随机重连一些边以创建小世界特性
    p = random.uniform(0.4, 0.7)  # 重连概率
    edges = list(G.edges())

    for u, v in edges:
        if random.random() < p:
            G.remove_edge(u, v)
            new_v = random.choice([node for node in total_nodes if node != u and not G.has_edge(u, node)])
            G.add_edge(u, new_v)

    # 生成一些高度节点
    num_hubs = random.randint(int(n*0.02), int(n*0.05))
    hubs = np.random.choice(total_nodes, num_hubs, replace=False)

    # 为这些高度节点添加额外连接
    for hub in hubs:
        # 添加额外连接
        if n<300:
            num_extra = np.random.randint(7, 15)
        else:
            num_extra = np.random.randint(8, int(n*0.04))
        for _ in range(num_extra):
            target = random.choice(total_nodes)
            if target != hub and not G.has_edge(hub, target):
                G.add_edge(hub, target)

    # 如果边数还不够，随机添加边
    p = random.uniform(0.3, 0.7)
    while G.number_of_edges() < target_edges:
        r1 = random.random()
        if r1 < p:
            # 获取当前所有节点和度数
            degrees = dict(G.degree())
            total_degree = sum(degrees.values())

            v = random.choice(list(total_nodes))

            # 选择另一个节点（按概率与度成正比）
            r = random.random() * total_degree
            cumulative = 0
            for node, degree in degrees.items():
                cumulative += degree
                if cumulative >= r:
                    u = node
                    break

            # 确保不添加自环和重复边
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
        else:
            v = random.choice(list(total_nodes))
            u = random.choice(list(total_nodes))
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)

    return G


def generate_er(G, target_edges):
    total_nodes = list(G.nodes())

    while target_edges:
        u = random.choice(total_nodes)
        v = random.choice(total_nodes)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            target_edges -= 1

    return G