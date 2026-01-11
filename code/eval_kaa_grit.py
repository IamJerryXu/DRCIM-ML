import os
import sys
import math
import yaml
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from code.gma_core.kaa_grit.kaa_grit_encoder import KaaGritEncoder
from code.dataset.create_dataset import create_dataset
from code.utils.data_loader import read_graph_dir, build_pyg_data_list


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_args(config):
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")

    class _Args:
        pass

    args = _Args()
    args.gcn_inchannel = dataset_cfg.get("gcn_inchannel", gma_cfg.get("centrality_dim", 6))
    args.graph_embedding = dataset_cfg.get("graph_embedding", 64)
    args.p = dataset_cfg.get("p", 1.5)
    args.q = dataset_cfg.get("q", 0.5)
    args.walk_length = dataset_cfg.get("walk_length", 6)
    args.num_walks = dataset_cfg.get("num_walks", 100)
    args.window_size = dataset_cfg.get("window_size", 10)
    args.workers = dataset_cfg.get("workers", 4)
    args.walk_length_f = dataset_cfg.get("walk_length_f", 6)
    args.num_walks_f = dataset_cfg.get("num_walks_f", 100)
    args.p_f = dataset_cfg.get("p_f", 1.0)
    args.q_f = dataset_cfg.get("q_f", 1.0)
    args.is_pe = dataset_cfg.get("is_pe", False)
    args.lap_pe_dim = dataset_cfg.get("lap_pe_dim", gma_cfg.get("lap_pe_dim", 10))
    args.rrwp_k_steps = dataset_cfg.get("rrwp_k_steps", gma_cfg.get("rrwp_k_steps", 5))
    args.use_node2vec = dataset_cfg.get("use_node2vec", False)
    args.label_path = dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/")
    args.is_cover_label = dataset_cfg.get("is_cover_label", False)
    args.skip_label = True
    return args


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    pos_ranks = ranks[y_true == 1]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1)
    ap = (precision * y_sorted).sum() / n_pos
    return float(ap)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _edge_pred_metrics(z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> Tuple[float, float]:
    if edge_index.numel() == 0 or num_nodes <= 1:
        return float("nan"), float("nan")
    num_pos = edge_index.size(1)
    neg_edge = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_pos)
    if neg_edge.numel() == 0:
        return float("nan"), float("nan")
    pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)
    y_score = torch.cat([pos_score, neg_score], dim=0).detach().cpu().numpy()
    y_true = np.concatenate([np.ones(num_pos, dtype=np.int32), np.zeros(neg_score.size(0), dtype=np.int32)])
    auc = _auc_score(y_true, y_score)
    ap = _average_precision(y_true, y_score)
    return auc, ap


def _topk_recall(z_a: torch.Tensor, z_b: torch.Tensor, ks: List[int], chunk_size: int = 256) -> Dict[int, float]:
    n = min(z_a.size(0), z_b.size(0))
    if n == 0:
        return {k: float("nan") for k in ks}
    z_a = F.normalize(z_a[:n], dim=1)
    z_b = F.normalize(z_b[:n], dim=1)
    max_k = max(ks)
    correct = {k: 0 for k in ks}
    device = z_a.device
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sim = torch.matmul(z_a[start:end], z_b.t())
        topk = torch.topk(sim, k=max_k, dim=1).indices
        target = torch.arange(start, end, device=device).unsqueeze(1)
        for k in ks:
            hit = (topk[:, :k] == target).any(dim=1)
            correct[k] += int(hit.sum().item())
    return {k: correct[k] / n for k in ks}


def _knn_metrics(z_a: torch.Tensor, z_b: torch.Tensor, topk: int = 5, chunk_size: int = 256) -> Dict[str, float]:
    n_a = z_a.size(0)
    n_b = z_b.size(0)
    if n_a == 0 or n_b == 0:
        return {"top1_sim": float("nan"), "topk_sim": float("nan"), "mutual_rate": float("nan")}

    k_ab = min(topk, n_b)
    k_ba = min(topk, n_a)

    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    top1_idx = []
    top1_sim_sum = 0.0
    topk_sim_sum = 0.0

    for start in range(0, n_a, chunk_size):
        end = min(n_a, start + chunk_size)
        sim = torch.matmul(z_a[start:end], z_b.t())
        vals, idx = torch.topk(sim, k=k_ab, dim=1)
        top1_sim_sum += vals[:, 0].sum().item()
        topk_sim_sum += vals.mean(dim=1).sum().item()
        top1_idx.append(idx[:, 0].detach().cpu())

    top1_idx = torch.cat(top1_idx, dim=0).numpy()

    # B -> A top-k indices for mutual check
    topk_ba = []
    for start in range(0, n_b, chunk_size):
        end = min(n_b, start + chunk_size)
        sim = torch.matmul(z_b[start:end], z_a.t())
        idx = torch.topk(sim, k=k_ba, dim=1).indices
        topk_ba.append(idx.detach().cpu())
    topk_ba = torch.cat(topk_ba, dim=0).numpy()

    mutual = 0
    for i in range(n_a):
        j = int(top1_idx[i])
        if j < 0 or j >= topk_ba.shape[0]:
            continue
        if i in set(topk_ba[j].tolist()):
            mutual += 1

    return {
        "top1_sim": top1_sim_sum / n_a,
        "topk_sim": topk_sim_sum / n_a,
        "mutual_rate": mutual / n_a,
    }


def _alpha_degree_corr(alpha: torch.Tensor, graph) -> float:
    if alpha.numel() == 0:
        return float("nan")
    alpha_np = alpha.detach().cpu().numpy().reshape(-1)
    nodes = list(graph.nodes())
    if not nodes:
        return float("nan")
    max_node = max(nodes)
    deg = np.zeros(max_node + 1, dtype=np.float32)
    for n, d in graph.degree():
        if n <= max_node:
            deg[n] = float(d)
    n = min(len(alpha_np), len(deg))
    return _spearmanr(alpha_np[:n], deg[:n])


def main():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    config = _load_config(config_path)

    device = torch.device(config.get("project", {}).get("device", "cpu"))
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(ROOT_DIR, dataset_root)

    eval_samples = dataset_cfg.get("eval_samples", 10)
    contrastive_pos_mode = gma_cfg.get("contrastive_pos_mode", "id")
    contrastive_topk = gma_cfg.get("contrastive_topk", 5)
    seed = config.get("project", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    encoder_a = KaaGritEncoder(
        in_dim=gma_cfg.get("input_dim", 16),
        hidden_dim=gma_cfg.get("hidden_dim", 64),
        out_dim=gma_cfg.get("output_dim", 32),
        num_heads=gma_cfg.get("heads", 4),
        num_layers=gma_cfg.get("layers", 2),
        k_steps=gma_cfg.get("k_steps", 5),
        dropout=gma_cfg.get("dropout", 0.1),
        num_groups=gma_cfg.get("num_groups", 8),
        groupkan_backend=gma_cfg.get("groupkan_backend", "torch"),
        ffn_mult=gma_cfg.get("ffn_mult", 4),
        attn_act=gma_cfg.get("attn_act", "relu"),
        attn_clamp=gma_cfg.get("attn_clamp", 0.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", False),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=str(device).split(":")[0],
    ).to(device)

    encoder_b = KaaGritEncoder(
        in_dim=gma_cfg.get("input_dim", 16),
        hidden_dim=gma_cfg.get("hidden_dim", 64),
        out_dim=gma_cfg.get("output_dim", 32),
        num_heads=gma_cfg.get("heads", 4),
        num_layers=gma_cfg.get("layers", 2),
        k_steps=gma_cfg.get("k_steps", 5),
        dropout=gma_cfg.get("dropout", 0.1),
        num_groups=gma_cfg.get("num_groups", 8),
        groupkan_backend=gma_cfg.get("groupkan_backend", "torch"),
        ffn_mult=gma_cfg.get("ffn_mult", 4),
        attn_act=gma_cfg.get("attn_act", "relu"),
        attn_clamp=gma_cfg.get("attn_clamp", 0.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", False),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=str(device).split(":")[0],
    ).to(device)

    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit")
    ckpt_a = os.path.join(ckpt_dir, f"{prefix}_encoder_a.pt")
    ckpt_b = os.path.join(ckpt_dir, f"{prefix}_encoder_b.pt")
    if not (os.path.isfile(ckpt_a) and os.path.isfile(ckpt_b)):
        raise FileNotFoundError("Checkpoint not found. Please train and save weights first.")
    encoder_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    encoder_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    encoder_a.eval()
    encoder_b.eval()

    test_dir = os.path.join(dataset_root, "test")
    graphs = read_graph_dir(test_dir)
    if not graphs:
        raise RuntimeError("No test graphs found.")

    args = _build_args(config)
    test_dataset = create_dataset(graphs, args, is_train=False)

    pairs = []
    for entry, data_entry in zip(graphs, test_dataset):
        data_list = build_pyg_data_list(data_entry, rrwp_k_steps=args.rrwp_k_steps)
        if len(data_list) < 2:
            continue
        pairs.append((entry["networks"][0], entry["networks"][1], data_list[0], data_list[1]))

    if eval_samples > 0:
        pairs = pairs[:eval_samples]
    if not pairs:
        raise RuntimeError("No valid layer pairs found.")

    auc_a_list = []
    ap_a_list = []
    auc_b_list = []
    ap_b_list = []
    topk_hits = {1: [], 5: [], 10: []}
    knn_top1_sim = []
    knn_topk_sim = []
    knn_mutual = []
    alpha_corr_a = []
    alpha_corr_b = []

    with torch.no_grad():
        for graph_a, graph_b, data_a, data_b in pairs:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)

            auc_a, ap_a = _edge_pred_metrics(z_a, data_a.edge_index, z_a.size(0))
            auc_b, ap_b = _edge_pred_metrics(z_b, data_b.edge_index, z_b.size(0))
            auc_a_list.append(auc_a)
            ap_a_list.append(ap_a)
            auc_b_list.append(auc_b)
            ap_b_list.append(ap_b)

            if contrastive_pos_mode == "id":
                recalls = _topk_recall(z_a, z_b, ks=[1, 5, 10])
                for k, val in recalls.items():
                    topk_hits[k].append(val)
            else:
                knn = _knn_metrics(z_a, z_b, topk=contrastive_topk)
                knn_top1_sim.append(knn["top1_sim"])
                knn_topk_sim.append(knn["topk_sim"])
                knn_mutual.append(knn["mutual_rate"])

            alpha_corr_a.append(_alpha_degree_corr(alpha_a, graph_a))
            alpha_corr_b.append(_alpha_degree_corr(alpha_b, graph_b))

    def _mean(values):
        clean = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        return float(np.mean(clean)) if clean else float("nan")

    print("=== KAA-GRIT 自监督评估 ===")
    print(f"样本数: {len(pairs)}")
    print("=== 结构重建 (Edge Prediction) ===")
    print(f"Layer1 AUC: {_mean(auc_a_list):.6f}, AP: {_mean(ap_a_list):.6f}")
    print(f"Layer2 AUC: {_mean(auc_b_list):.6f}, AP: {_mean(ap_b_list):.6f}")
    if contrastive_pos_mode == "id":
        print("=== 跨层检索 (Top-K Recall) ===")
        print(f"Top-1: {_mean(topk_hits[1]):.6f}, Top-5: {_mean(topk_hits[5]):.6f}, Top-10: {_mean(topk_hits[10]):.6f}")
    else:
        print("=== 跨层检索 (Top-K Similarity) ===")
        print(
            f"Top-1 sim: {_mean(knn_top1_sim):.6f}, "
            f"Top-{contrastive_topk} sim: {_mean(knn_topk_sim):.6f}, "
            f"Mutual@{contrastive_topk}: {_mean(knn_mutual):.6f}"
        )
    print("=== 注意力一致性 (alpha vs degree, Spearman) ===")
    print(f"Layer1: {_mean(alpha_corr_a):.6f}, Layer2: {_mean(alpha_corr_b):.6f}")


if __name__ == "__main__":
    main()
