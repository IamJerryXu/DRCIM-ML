import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from ..dataset.create_rrwp import compute_rrwp


def _parse_nodes_num_from_name(name):
    base = os.path.basename(name)
    if base.endswith(".txt"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[-3])
        except ValueError:
            return None
    return None


def read_multilayer_edgelist(file_path, nodes_num=None):
    """
    读取 MIM 风格的多层 edgelist 文件。

    格式示例:
    layer:1 nodes=375 edges=949
    0 299
    0 326
    layer:2 nodes=375 edges=850
    1 370
    1 401

    返回:
        networks: list[networkx.Graph]
    """
    networks = []
    current_graph = None
    if nodes_num is None:
        nodes_num = _parse_nodes_num_from_name(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("layer:"):
                if "nodes=" in line:
                    try:
                        if nodes_num is None:
                            nodes_num = int(line.split("nodes=")[1].split()[0])
                    except Exception:
                        nodes_num = nodes_num
                current_graph = nx.Graph()
                if nodes_num is not None:
                    current_graph.add_nodes_from(range(nodes_num))
                networks.append(current_graph)
                continue
            if current_graph is None:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            current_graph.add_edge(u, v)

    if networks and nodes_num is None:
        max_node = max(max(g.nodes()) for g in networks if g.nodes())
        for g in networks:
            g.add_nodes_from(range(max_node + 1))

    return networks


def read_graph_dir(path):
    """
    读取 MIM 风格的 train/test 目录，返回多层图列表。
    """
    graph_files = []
    for name in os.listdir(path):
        if not name.endswith(".txt"):
            continue
        graph_files.append(name)

    def _sort_key(fname):
        try:
            return int(fname.split("_")[0])
        except Exception:
            return fname

    graph_files = sorted(graph_files, key=_sort_key)

    graphs = []
    for name in graph_files:
        full_path = os.path.join(path, name)
        nodes_num = _parse_nodes_num_from_name(name)
        networks = read_multilayer_edgelist(full_path, nodes_num=nodes_num)
        graphs.append({"name": name[: name.rfind(".txt")], "networks": networks})

    return graphs


def read_graph_file(file_path):
    """
    读取单个多层 edgelist 文件，返回与 read_graph_dir 一致的结构。
    """
    nodes_num = _parse_nodes_num_from_name(file_path)
    networks = read_multilayer_edgelist(file_path, nodes_num=nodes_num)
    name = os.path.basename(file_path)
    if name.endswith(".txt"):
        name = name[: name.rfind(".txt")]
    return [{"name": name, "networks": networks}]


def build_pyg_data_from_entry(entry, layer_idx, rrwp_k_steps=0):
    """
    将 create_dataset 输出的 entry 转成 PyG Data。

    Args:
        entry (dict): create_dataset 返回的单个图样本
        layer_idx (int): 层索引
        rrwp_k_steps (int): RRWP 步数
    """
    edge_index = torch.tensor(entry["edge_index"][layer_idx], dtype=torch.long)
    if "x" in entry and len(entry["x"]) > 0:
        x = torch.tensor(entry["x"][layer_idx], dtype=torch.float32)
    else:
        # 兜底: 使用中心性特征作为 x
        x = torch.tensor(entry["centrality_feature"][layer_idx], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)

    if "rrwp_feature" in entry and len(entry["rrwp_feature"]) > 0:
        data.rrwp = torch.tensor(entry["rrwp_feature"][layer_idx], dtype=torch.float32)
    elif rrwp_k_steps > 0:
        rrwp_feat = compute_rrwp(edge_index, x.size(0), k_steps=rrwp_k_steps)
        data.rrwp = rrwp_feat

    return data


def build_pyg_data_list(entry, rrwp_k_steps=0):
    """
    将单个样本 entry 转成多层 PyG Data 列表。
    """
    data_list = []
    for i in range(len(entry["edge_index"])):
        data_list.append(build_pyg_data_from_entry(entry, i, rrwp_k_steps=rrwp_k_steps))
    return data_list
