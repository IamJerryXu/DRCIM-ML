import sys
import numpy as np
import os
import networkx as nx
from tqdm import tqdm

# Fix paths using absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from .create_centrality import create_centrality
from .create_embedding import node2vec
from .create_edge_index import get_edge_index
from .create_pe import create_walks, create_frequency
from .create_laplacian import normalized_laplacian_matrix, compute_lap_pe
from .create_rrwp import compute_rrwp
from .create_label import celf
from tools.normalize import normalize_labels

def _pad_rows(arr, target_rows):
    """将二维或三维特征按行补零到统一节点数。"""
    if arr is None:
        return None
    if arr.shape[0] == target_rows:
        return arr
    if arr.shape[0] > target_rows:
        return arr[:target_rows]
    pad_rows = target_rows - arr.shape[0]
    pad_shape = (pad_rows,) + arr.shape[1:]
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def process_graph(
    g,
    f_path,
    index,
    args,
    option,
    option_nv,
    option_c7,
    option_lpe,
    lap_pe_dim,
    option_rrwp,
    rrwp_k_steps,
    max_nodes,
):
    data = {}
    index = str(index)
    c_num = str(args.gcn_inchannel)
    c = f_path + "/centrality" + c_num + "/" + index +  "_c" + c_num + ".npy"
    nv = f_path + "/embedding/" + str(args.graph_embedding) + '/' + index + "_" + str(args.p) + "_" + str(args.q) + "_nv.npy"
    ei = f_path + "/edge_index/" + index +  "_ei.npy"
    lp = f_path + "/lp/" + index + "_lp.npy"
    nw = f_path + "/nodewalk/" + index + "_nw.npy"
    nwf = f_path + "/nodewalkf/" + index + "_nwf.npy"
    lpe = f_path + "/lap_pe/" + index + f"_lpe_{lap_pe_dim}.npy"
    rrwp = f_path + "/rrwp/" + index + f"_rrwp_{rrwp_k_steps}.npy"

    # 目录兜底，避免缓存目录缺失导致保存失败
    os.makedirs(os.path.dirname(c), exist_ok=True)
    os.makedirs(os.path.dirname(ei), exist_ok=True)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    os.makedirs(os.path.dirname(nw), exist_ok=True)
    os.makedirs(os.path.dirname(nwf), exist_ok=True)
    if lap_pe_dim > 0:
        os.makedirs(os.path.dirname(lpe), exist_ok=True)
    if rrwp_k_steps > 0:
        os.makedirs(os.path.dirname(rrwp), exist_ok=True)
    use_node2vec = getattr(args, "use_node2vec", False)
    if use_node2vec:
        os.makedirs(os.path.dirname(nv), exist_ok=True)

    # 统一检查文件是否存在，不存在就生成，避免历史缓存不完整导致报错
    if not os.path.exists(ei):
        get_edge_index(g, ei)
    if use_node2vec and not os.path.exists(nv):
        node2vec(g, nv, args)
    if not os.path.exists(c):
        create_centrality(g, c, c_num)
    if args.is_pe:
        if not os.path.exists(lp):
            normalized_laplacian_matrix(g, lp)
        if not os.path.exists(nw):
            create_walks(g, nw, args)
        if not os.path.exists(nwf):
            create_frequency(nw, nwf)
        data["spatial_feature"] = _pad_rows(np.load(nwf), max_nodes)
        data["lp_feature"] = _pad_rows(np.load(lp), max_nodes)
    if lap_pe_dim > 0:
        if not os.path.exists(lpe):
            os.makedirs(os.path.dirname(lpe), exist_ok=True)
            compute_lap_pe(g, lap_pe_dim, lpe)
        elif option_lpe or option:
            compute_lap_pe(g, lap_pe_dim, lpe)
        lap_pe = np.load(lpe)
        # 若缓存维度不一致则重算
        if lap_pe.ndim != 2 or lap_pe.shape[1] != lap_pe_dim:
            compute_lap_pe(g, lap_pe_dim, lpe)
            lap_pe = np.load(lpe)
        data["lap_pe_feature"] = _pad_rows(lap_pe, max_nodes)

    centrality = np.load(c)
    # 若中心性维度不一致则重算
    try:
        c_num_int = int(c_num)
    except Exception:
        c_num_int = centrality.shape[1]
    if centrality.ndim != 2 or centrality.shape[1] != c_num_int:
        create_centrality(g, c, c_num)
        centrality = np.load(c)
    data["centrality_feature"] = _pad_rows(centrality, max_nodes)
    if use_node2vec:
        embedding = np.load(nv)
        if embedding.ndim != 2 or embedding.shape[1] != args.graph_embedding:
            node2vec(g, nv, args)
            embedding = np.load(nv)
        data["embedding_feature"] = _pad_rows(embedding, max_nodes)
    else:
        data["embedding_feature"] = None
    data["edge_index"] = np.load(ei)
    if rrwp_k_steps > 0:
        # RRWP 以边为单位保存，形状 [E, K]
        if not os.path.exists(rrwp):
            rrwp_feat = compute_rrwp(data["edge_index"], g.number_of_nodes(), k_steps=rrwp_k_steps)
            np.save(rrwp, rrwp_feat.cpu().numpy())
        rrwp_feat = np.load(rrwp)
        if rrwp_feat.ndim != 2 or rrwp_feat.shape[1] != rrwp_k_steps:
            rrwp_feat = compute_rrwp(data["edge_index"], g.number_of_nodes(), k_steps=rrwp_k_steps)
            np.save(rrwp, rrwp_feat.cpu().numpy())
            rrwp_feat = np.load(rrwp)
        data["rrwp_feature"] = rrwp_feat
    return data

def _process_graph_entry(Gm, args, base, lap_pe_dim, rrwp_k_steps, use_lap_pe, use_rrwp):
    if args.is_pe:
        data_l = {"name": [], "graph": [], "centrality_feature": [], "embedding_feature": [], "spatial_feature": [],
                  "lp_feature": [],
                  "edge_index": []}
    else:
        data_l = {"name": [], "graph": [], "centrality_feature": [], "embedding_feature": [], "edge_index": []}
    if use_lap_pe:
        data_l["lap_pe_feature"] = []
    if use_lap_pe or getattr(args, "use_node2vec", False):
        data_l["x"] = []
    if use_rrwp:
        data_l["rrwp_feature"] = []

    networks = Gm['networks']
    max_nodes = 0
    for g in networks:
        max_nodes = max(max_nodes, g.number_of_nodes())
    name = Gm['name']
    f_path = base + name
    data_l['name'] = name
    option = 0
    option_nv = 0
    option_c7 = 0
    option_lpe = 0
    option_rrwp = 0
    c = "/centrality"+str(args.gcn_inchannel)+"/"
    if not os.path.exists(f_path):
        os.makedirs(f_path, exist_ok=True)
        os.makedirs(f_path + c, exist_ok=True)
        if getattr(args, "use_node2vec", False):
            os.makedirs(f_path + "/embedding/" + str(args.graph_embedding), exist_ok=True)
        os.makedirs(f_path + "/lp/", exist_ok=True)
        os.makedirs(f_path + "/nodewalkf/", exist_ok=True)
        os.makedirs(f_path + "/nodewalk/", exist_ok=True)
        os.makedirs(f_path + "/edge_index/", exist_ok=True)
        if use_lap_pe:
            os.makedirs(f_path + "/lap_pe/", exist_ok=True)
        if use_rrwp:
            os.makedirs(f_path + "/rrwp/", exist_ok=True)
        option = 1
    if getattr(args, "use_node2vec", False) and not os.path.exists(f_path + "/embedding/" + str(args.graph_embedding)):
        os.makedirs(f_path + "/embedding/" + str(args.graph_embedding), exist_ok=True)
        option_nv = 1
    if not os.path.exists(f_path + c):
        os.makedirs(f_path + c, exist_ok=True)
        option_c7 = 1
    if use_lap_pe and not os.path.exists(f_path + "/lap_pe/"):
        os.makedirs(f_path + "/lap_pe/", exist_ok=True)
        option_lpe = 1
    if use_rrwp and not os.path.exists(f_path + "/rrwp/"):
        os.makedirs(f_path + "/rrwp/", exist_ok=True)
        option_rrwp = 1

    for index, g in enumerate(networks):
        # 统一重映射为 0..N-1，避免节点编号不连续导致矩阵构造报错
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        if g.number_of_nodes() < max_nodes:
            g.add_nodes_from(range(max_nodes))
        data = process_graph(
            g,
            f_path,
            index + 1,
            args,
            option,
            option_nv,
            option_c7,
            option_lpe,
            lap_pe_dim,
            option_rrwp,
            rrwp_k_steps,
            max_nodes,
        )
        data_l["graph"].append(g)
        data_l["centrality_feature"].append(data["centrality_feature"])
        data_l["embedding_feature"].append(data["embedding_feature"])
        data_l["edge_index"].append(data["edge_index"])
        if args.is_pe:
            data_l["spatial_feature"].append(data["spatial_feature"])
            data_l["lp_feature"].append(data["lp_feature"])
        if use_lap_pe:
            data_l["lap_pe_feature"].append(data["lap_pe_feature"])
        if use_rrwp:
            data_l["rrwp_feature"].append(data["rrwp_feature"])

    # 中心性特征归一化
    cf = np.array(data_l["centrality_feature"])
    ma = np.max(cf, axis=1)  # 各层的最大值
    # 避免除零导致 NaN
    ma_safe = np.where(ma <= 0, 1.0, ma)
    data_l["centrality_feature"] = cf / ma_safe.reshape(ma.shape[0], 1, ma.shape[1])
    lap_pe = np.array(data_l["lap_pe_feature"]) if use_lap_pe else None
    if use_lap_pe or getattr(args, "use_node2vec", False):
        x_list = []
        for i in range(data_l["centrality_feature"].shape[0]):
            parts = [data_l["centrality_feature"][i]]
            if getattr(args, "use_node2vec", False) and data_l["embedding_feature"][i] is not None:
                parts.append(data_l["embedding_feature"][i])
            if use_lap_pe and lap_pe is not None:
                parts.append(lap_pe[i])
            # x = [中心性特征 + Node2Vec(可选) + LapPE(可选)]
            x = np.concatenate(parts, axis=1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x_list.append(x)
        data_l["x"] = x_list

    if not getattr(args, "skip_label", False):
        save_path = args.label_path + Gm['name'] + "_lb.npy"
        if not os.path.exists(save_path) or args.is_cover_label:
            celf(Gm['networks'], save_path, p=0.1, mc=1000)
        # minmax,standard,quantile
        label = normalize_labels(np.load(save_path), method='quantile')
        data_l["label"] = [label]

    return data_l


def create_dataset(Gms, args,is_train=True):
    dataset = []
    if is_train:
        base = 'dataset/feature/train/'
    else:
        base = 'dataset/feature/eval/'

    # LapPE 维度配置，>0 时才启用
    lap_pe_dim = getattr(args, "lap_pe_dim", 0)
    use_lap_pe = lap_pe_dim is not None and lap_pe_dim > 0
    # RRWP 步数配置，>0 时才启用
    rrwp_k_steps = getattr(args, "rrwp_k_steps", 0)
    use_rrwp = rrwp_k_steps is not None and rrwp_k_steps > 0
    # 是否跳过标签生成（用于 GMA 预训练）
    skip_label = getattr(args, "skip_label", False)
    # 是否启用 Node2Vec
    use_node2vec = getattr(args, "use_node2vec", False)

    for Gm in tqdm(Gms):
        dataset.append(
            _process_graph_entry(
                Gm,
                args,
                base,
                lap_pe_dim,
                rrwp_k_steps,
                use_lap_pe,
                use_rrwp,
            )
        )

    return dataset
