#!/usr/bin/env python
"""
GMA-MFEA 两阶段模型测试
======================
运行: python test_twostage.py
"""

import os
import sys
import yaml
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from types import SimpleNamespace

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from code.gma_core.kaa_grit.kaa_grit_encoder import KaaGritEncoder
from code.dataset.create_dataset import create_dataset
from code.utils.data_loader import read_graph_dir, build_pyg_data_list


def compute_reconstruction_metrics(z, edge_index, num_nodes, k=50):
    """计算重建指标: AUC, Precision@K"""
    z_norm = torch.nn.functional.normalize(z, p=2, dim=-1)
    
    src, dst = edge_index[0], edge_index[1]
    num_edges = src.size(0)
    pos_scores = (z_norm[src] * z_norm[dst]).sum(dim=-1)
    
    num_neg = min(num_edges * 3, 10000)
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=z.device)
    neg_scores = (z_norm[neg_src] * z_norm[neg_dst]).sum(dim=-1)
    
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])
    try:
        auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    except:
        auc = 0.5
    
    k = min(k, num_edges)
    all_pairs_scores = torch.mm(z_norm, z_norm.t())
    all_pairs_scores.fill_diagonal_(-float('inf'))
    
    edge_set = set()
    for i in range(num_edges):
        edge_set.add((src[i].item(), dst[i].item()))
    
    topk_scores, topk_indices = torch.topk(all_pairs_scores.flatten(), k)
    rows = topk_indices // num_nodes
    cols = topk_indices % num_nodes
    
    correct = sum(1 for r, c in zip(rows.tolist(), cols.tolist()) 
                  if (r, c) in edge_set or (c, r) in edge_set)
    precision_at_k = correct / k
    
    return auc, precision_at_k


def compute_role_alignment_metrics(z_a, z_b, alpha_a, alpha_b, topk_ratio=0.2):
    """计算跨层角色对齐指标"""
    z_a_norm = torch.nn.functional.normalize(z_a, p=2, dim=-1)
    z_b_norm = torch.nn.functional.normalize(z_b, p=2, dim=-1)
    
    n_a, n_b = z_a.size(0), z_b.size(0)
    k_a = max(3, int(n_a * topk_ratio))
    k_b = max(3, int(n_b * topk_ratio))
    
    hub_idx_a = torch.topk(alpha_a, k=k_a).indices
    hub_idx_b = torch.topk(alpha_b, k=k_b).indices
    periph_idx_a = torch.topk(-alpha_a, k=k_a).indices
    periph_idx_b = torch.topk(-alpha_b, k=k_b).indices
    
    z_hub_a = z_a_norm[hub_idx_a]
    z_hub_b = z_b_norm[hub_idx_b]
    z_periph_a = z_a_norm[periph_idx_a]
    z_periph_b = z_b_norm[periph_idx_b]
    
    # Hub质心相似度
    center_a = z_hub_a.mean(dim=0)
    center_b = z_hub_b.mean(dim=0)
    hub_center_sim = torch.nn.functional.cosine_similarity(
        center_a.unsqueeze(0), center_b.unsqueeze(0)
    ).item()
    
    # Hub最大匹配相似度
    sim_matrix = torch.mm(z_hub_a, z_hub_b.t())
    hub_max_sim = sim_matrix.max(dim=1).values.mean().item()
    
    # Peripheral质心相似度
    periph_center_a = z_periph_a.mean(dim=0)
    periph_center_b = z_periph_b.mean(dim=0)
    periph_center_sim = torch.nn.functional.cosine_similarity(
        periph_center_a.unsqueeze(0), periph_center_b.unsqueeze(0)
    ).item()
    
    # 角色分离度
    hub_periph_sim_a = torch.nn.functional.cosine_similarity(
        center_a.unsqueeze(0), periph_center_a.unsqueeze(0)
    ).item()
    role_separation = hub_center_sim - hub_periph_sim_a
    
    # α排名相关性
    full_sim = torch.mm(z_a_norm, z_b_norm.t())
    best_match_b = full_sim.argmax(dim=1)
    matched_alpha_b = alpha_b[best_match_b]
    
    try:
        corr, _ = spearmanr(alpha_a.cpu().numpy(), matched_alpha_b.cpu().numpy())
        if np.isnan(corr):
            corr = 0.0
    except:
        corr = 0.0
    
    return {
        "hub_center_sim": hub_center_sim,
        "hub_max_sim": hub_max_sim,
        "periph_center_sim": periph_center_sim,
        "role_separation": role_separation,
        "alpha_rank_corr": corr,
    }


def main():
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "config_twostage.yaml")
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    device = torch.device(config.get("project", {}).get("device", "cuda:4"))
    
    checkpoint_prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit_twostage")
    
    print("=" * 60)
    print("GMA-MFEA 两阶段模型评估")
    print("=" * 60)
    
    # 创建编码器
    encoder_a = KaaGritEncoder(
        in_dim=gma_cfg.get("input_dim", 304),
        hidden_dim=gma_cfg.get("hidden_dim", 512),
        out_dim=gma_cfg.get("output_dim", 256),
        num_heads=gma_cfg.get("heads", 16),
        num_layers=gma_cfg.get("layers", 4),
        k_steps=gma_cfg.get("k_steps", 5),
        dropout=gma_cfg.get("dropout", 0.15),
        num_groups=gma_cfg.get("num_groups", 8),
        groupkan_backend=gma_cfg.get("groupkan_backend", "torch"),
        ffn_mult=gma_cfg.get("ffn_mult", 4),
        attn_act=gma_cfg.get("attn_act", "relu"),
        attn_clamp=gma_cfg.get("attn_clamp", 5.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", True),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=device.type,
    ).to(device)
    
    encoder_b = KaaGritEncoder(
        in_dim=gma_cfg.get("input_dim", 304),
        hidden_dim=gma_cfg.get("hidden_dim", 512),
        out_dim=gma_cfg.get("output_dim", 256),
        num_heads=gma_cfg.get("heads", 16),
        num_layers=gma_cfg.get("layers", 4),
        k_steps=gma_cfg.get("k_steps", 5),
        dropout=gma_cfg.get("dropout", 0.15),
        num_groups=gma_cfg.get("num_groups", 8),
        groupkan_backend=gma_cfg.get("groupkan_backend", "torch"),
        ffn_mult=gma_cfg.get("ffn_mult", 4),
        attn_act=gma_cfg.get("attn_act", "relu"),
        attn_clamp=gma_cfg.get("attn_clamp", 5.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", True),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=device.type,
    ).to(device)
    
    # 加载权重
    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    ckpt_a = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_a.pt")
    ckpt_b = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_b.pt")
    
    if not os.path.exists(ckpt_a):
        print(f"权重文件不存在: {ckpt_a}")
        print("请先运行训练: python train_twostage.py")
        return
    
    encoder_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    encoder_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    encoder_a.eval()
    encoder_b.eval()
    
    print(f"✓ 已加载权重: {checkpoint_prefix}")
    
    # 加载测试数据
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    test_path = os.path.join(dataset_root, "test")
    
    if not os.path.isdir(test_path):
        print(f"测试目录不存在: {test_path}")
        return
    
    graphs = read_graph_dir(test_path)
    
    args = SimpleNamespace(
        gcn_inchannel=dataset_cfg.get("gcn_inchannel", 12),
        graph_embedding=dataset_cfg.get("graph_embedding", 256),
        p=dataset_cfg.get("p", 1.5),
        q=dataset_cfg.get("q", 0.5),
        walk_length=dataset_cfg.get("walk_length", 6),
        num_walks=dataset_cfg.get("num_walks", 100),
        window_size=dataset_cfg.get("window_size", 10),
        workers=dataset_cfg.get("workers", 4),
        walk_length_f=dataset_cfg.get("walk_length_f", 6),
        num_walks_f=dataset_cfg.get("num_walks_f", 100),
        p_f=dataset_cfg.get("p_f", 1.0),
        q_f=dataset_cfg.get("q_f", 1.0),
        is_pe=dataset_cfg.get("is_pe", False),
        lap_pe_dim=dataset_cfg.get("lap_pe_dim", 36),
        rrwp_k_steps=dataset_cfg.get("rrwp_k_steps", 5),
        use_node2vec=dataset_cfg.get("use_node2vec", True),
        preprocess_workers=dataset_cfg.get("preprocess_workers", 4),
        label_path=dataset_cfg.get("label_path", ""),
        is_cover_label=False,
        skip_label=True,
    )
    
    dataset = create_dataset(graphs, args, is_train=False)
    
    eval_samples = dataset_cfg.get("eval_samples", 30)
    
    metrics = {
        "auc_a": [], "auc_b": [], "prec_a": [], "prec_b": [],
        "hub_center": [], "hub_max": [], "periph_center": [],
        "role_sep": [], "alpha_corr": [],
    }
    
    print(f"\n📊 测试样本数: {min(eval_samples, len(dataset))}\n")
    
    with torch.no_grad():
        for i, entry in enumerate(dataset[:eval_samples]):
            data_list = build_pyg_data_list(entry, rrwp_k_steps=args.rrwp_k_steps)
            if len(data_list) < 2:
                continue
            
            data_a, data_b = data_list[0].to(device), data_list[1].to(device)
            
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            
            auc_a, prec_a = compute_reconstruction_metrics(z_a, data_a.edge_index, z_a.size(0))
            auc_b, prec_b = compute_reconstruction_metrics(z_b, data_b.edge_index, z_b.size(0))
            
            metrics["auc_a"].append(auc_a)
            metrics["auc_b"].append(auc_b)
            metrics["prec_a"].append(prec_a)
            metrics["prec_b"].append(prec_b)
            
            role_metrics = compute_role_alignment_metrics(z_a, z_b, alpha_a, alpha_b)
            
            metrics["hub_center"].append(role_metrics["hub_center_sim"])
            metrics["hub_max"].append(role_metrics["hub_max_sim"])
            metrics["periph_center"].append(role_metrics["periph_center_sim"])
            metrics["role_sep"].append(role_metrics["role_separation"])
            metrics["alpha_corr"].append(role_metrics["alpha_rank_corr"])
    
    # 输出结果
    print("=" * 60)
    print("【层内重建指标】")
    print("=" * 60)
    print(f"  Layer A - AUC:         {np.mean(metrics['auc_a']):.4f} ± {np.std(metrics['auc_a']):.4f}")
    print(f"  Layer A - Precision@K: {np.mean(metrics['prec_a']):.4f} ± {np.std(metrics['prec_a']):.4f}")
    print(f"  Layer B - AUC:         {np.mean(metrics['auc_b']):.4f} ± {np.std(metrics['auc_b']):.4f}")
    print(f"  Layer B - Precision@K: {np.mean(metrics['prec_b']):.4f} ± {np.std(metrics['prec_b']):.4f}")
    avg_auc = (np.mean(metrics['auc_a']) + np.mean(metrics['auc_b'])) / 2
    print(f"  ✅ 平均重建 AUC:       {avg_auc:.4f}")
    
    print("\n" + "=" * 60)
    print("【跨层功能角色对齐指标】")
    print("=" * 60)
    print(f"  Hub质心相似度:         {np.mean(metrics['hub_center']):.4f} ± {np.std(metrics['hub_center']):.4f}")
    print(f"  Hub最大匹配相似度:     {np.mean(metrics['hub_max']):.4f} ± {np.std(metrics['hub_max']):.4f}")
    print(f"  Peripheral质心相似度:  {np.mean(metrics['periph_center']):.4f} ± {np.std(metrics['periph_center']):.4f}")
    print(f"  角色分离度:            {np.mean(metrics['role_sep']):.4f} ± {np.std(metrics['role_sep']):.4f}")
    print(f"  α排名相关性:           {np.mean(metrics['alpha_corr']):.4f} ± {np.std(metrics['alpha_corr']):.4f}")
    
    hub_score = (np.mean(metrics['hub_center']) + np.mean(metrics['hub_max'])) / 2
    print(f"\n  ✅ Hub对齐综合得分:    {hub_score:.4f}")


if __name__ == "__main__":
    main()
