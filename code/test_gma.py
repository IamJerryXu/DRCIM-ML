import os
import sys
from types import SimpleNamespace

import yaml
import torch
import torch.nn.functional as F
import numpy as np

from code.gma_core.kaa_grit.kaa_grit_encoder import KaaGritEncoder
from code.gma_core.loss_ammd import AttentionAwareMMDLoss
from code.gma_core.loss_recon import GraphReconstructionLoss
from code.gma_core.loss_total import AlignmentReconstructionLoss
from code.dataset.create_dataset import create_dataset
from code.utils.data_loader import read_graph_dir, build_pyg_data_list


def _build_args(config):
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")

    return SimpleNamespace(
        gcn_inchannel=dataset_cfg.get("gcn_inchannel", gma_cfg.get("centrality_dim", 6)),
        graph_embedding=dataset_cfg.get("graph_embedding", 64),
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
        lap_pe_dim=dataset_cfg.get("lap_pe_dim", gma_cfg.get("lap_pe_dim", 10)),
        rrwp_k_steps=dataset_cfg.get("rrwp_k_steps", gma_cfg.get("rrwp_k_steps", 5)),
        use_node2vec=dataset_cfg.get("use_node2vec", False),
        label_path=dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/"),
        is_cover_label=dataset_cfg.get("is_cover_label", False),
        skip_label=True,
    )


def _build_pair_list(dataset, rrwp_k_steps):
    pairs = []
    for entry in dataset:
        data_list = build_pyg_data_list(entry, rrwp_k_steps=rrwp_k_steps)
        if len(data_list) < 2:
            continue
        pairs.append((data_list[0], data_list[1]))
    return pairs


def compute_reconstruction_metrics(z, edge_index, num_nodes):
    """
    计算层内重建指标
    - AUC: 正边 vs 负边的分类准确率
    - Precision@K: Top-K预测边中真实边的比例
    """
    z_norm = F.normalize(z, p=2, dim=-1)
    
    # 正边得分
    src, dst = edge_index
    pos_scores = (z_norm[src] * z_norm[dst]).sum(dim=-1)
    
    # 采样负边
    num_pos = edge_index.size(1)
    num_neg = min(num_pos, num_nodes * 10)  # 限制负样本数量
    
    edge_set = set()
    for i in range(edge_index.size(1)):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    
    neg_src, neg_dst = [], []
    attempts = 0
    while len(neg_src) < num_neg and attempts < num_neg * 10:
        s = np.random.randint(0, num_nodes)
        d = np.random.randint(0, num_nodes)
        if s != d and (s, d) not in edge_set:
            neg_src.append(s)
            neg_dst.append(d)
        attempts += 1
    
    if len(neg_src) == 0:
        return 0.5, 0.0  # 无法计算
    
    neg_src = torch.tensor(neg_src, device=z.device)
    neg_dst = torch.tensor(neg_dst, device=z.device)
    neg_scores = (z_norm[neg_src] * z_norm[neg_dst]).sum(dim=-1)
    
    # 计算 AUC
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([
        torch.ones(pos_scores.size(0), device=z.device),
        torch.zeros(neg_scores.size(0), device=z.device)
    ])
    
    # 简单AUC：正边得分 > 负边得分的比例
    pos_scores_np = pos_scores.cpu().numpy()
    neg_scores_np = neg_scores.cpu().numpy()
    
    correct = 0
    total = 0
    # 采样比较以提高效率
    sample_size = min(1000, len(pos_scores_np), len(neg_scores_np))
    pos_sample = np.random.choice(pos_scores_np, sample_size, replace=True)
    neg_sample = np.random.choice(neg_scores_np, sample_size, replace=True)
    for p, n in zip(pos_sample, neg_sample):
        if p > n:
            correct += 1
        elif p == n:
            correct += 0.5
        total += 1
    
    auc = correct / total if total > 0 else 0.5
    
    # Precision@K
    k = min(num_pos, 100)
    topk_idx = torch.topk(all_scores, k=k).indices
    topk_labels = all_labels[topk_idx]
    precision_k = topk_labels.sum().item() / k
    
    return auc, precision_k


def compute_cross_layer_alignment_metrics(z_a, z_b, alpha_a, alpha_b, topk_ratio=0.1):
    """
    计算跨层功能角色对齐指标
    - Hub-to-Hub Similarity: 两层Top-K高α节点之间的平均相似度
    - Alpha Correlation: α分布的相关性
    - Role Matching Accuracy: 角色匹配准确率
    """
    n_a, n_b = z_a.size(0), z_b.size(0)
    n = min(n_a, n_b)
    
    z_a_norm = F.normalize(z_a[:n], p=2, dim=-1)
    z_b_norm = F.normalize(z_b[:n], p=2, dim=-1)
    alpha_a = alpha_a[:n]
    alpha_b = alpha_b[:n]
    
    # 1. Hub-to-Hub Similarity
    k = max(5, int(n * topk_ratio))
    topk_a = torch.topk(alpha_a, k=k).indices
    topk_b = torch.topk(alpha_b, k=k).indices
    
    hub_z_a = z_a_norm[topk_a]  # [K, D]
    hub_z_b = z_b_norm[topk_b]  # [K, D]
    
    # 两组hub节点的质心相似度
    center_a = hub_z_a.mean(dim=0)
    center_b = hub_z_b.mean(dim=0)
    hub_center_sim = F.cosine_similarity(center_a.unsqueeze(0), center_b.unsqueeze(0)).item()
    
    # hub节点间的平均最大相似度（每个hub_a找最相似的hub_b）
    sim_matrix = torch.mm(hub_z_a, hub_z_b.t())  # [K, K]
    hub_max_sim = sim_matrix.max(dim=1).values.mean().item()
    
    # 2. Peripheral-to-Peripheral Similarity (低α节点)
    bottomk_a = torch.topk(alpha_a, k=k, largest=False).indices
    bottomk_b = torch.topk(alpha_b, k=k, largest=False).indices
    
    periph_z_a = z_a_norm[bottomk_a]
    periph_z_b = z_b_norm[bottomk_b]
    
    periph_center_a = periph_z_a.mean(dim=0)
    periph_center_b = periph_z_b.mean(dim=0)
    periph_center_sim = F.cosine_similarity(periph_center_a.unsqueeze(0), periph_center_b.unsqueeze(0)).item()
    
    # 3. Role Separation: hub和peripheral应该分开
    # Hub中心与Peripheral中心的距离（应该远）
    hub_periph_dist_a = 1.0 - F.cosine_similarity(center_a.unsqueeze(0), periph_center_a.unsqueeze(0)).item()
    hub_periph_dist_b = 1.0 - F.cosine_similarity(center_b.unsqueeze(0), periph_center_b.unsqueeze(0)).item()
    role_separation = (hub_periph_dist_a + hub_periph_dist_b) / 2
    
    # 4. Alpha Rank Correlation (Spearman)
    # 按α排序后，检查高α节点在嵌入空间中是否也相近
    rank_a = torch.argsort(torch.argsort(alpha_a, descending=True))
    rank_b = torch.argsort(torch.argsort(alpha_b, descending=True))
    
    # 计算嵌入相似度矩阵
    full_sim = torch.mm(z_a_norm, z_b_norm.t())  # [N, N]
    
    # 对每个节点i，找其在另一层最相似的节点j的α排名
    best_match_b = full_sim.argmax(dim=1)  # 每个a节点最匹配的b节点
    matched_rank_b = rank_b[best_match_b]
    
    # Spearman相关系数近似
    rank_diff = (rank_a.float() - matched_rank_b.float()).abs()
    rank_correlation = 1.0 - (6 * (rank_diff ** 2).sum() / (n * (n**2 - 1))).item()
    rank_correlation = max(-1.0, min(1.0, rank_correlation))
    
    return {
        "hub_center_sim": hub_center_sim,
        "hub_max_sim": hub_max_sim,
        "periph_center_sim": periph_center_sim,
        "role_separation": role_separation,
        "rank_correlation": rank_correlation,
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    config_path = os.path.join(root_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    device = torch.device(config.get("project", {}).get("device", "cpu"))

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
        attn_clamp=gma_cfg.get("attn_clamp", 5.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", True),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=device.type,
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
        attn_clamp=gma_cfg.get("attn_clamp", 5.0),
        signed_sqrt=gma_cfg.get("signed_sqrt", True),
        edge_enhance=gma_cfg.get("edge_enhance", True),
        device=device.type,
    ).to(device)

    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit")
    ckpt_a = os.path.join(ckpt_dir, f"{prefix}_encoder_a.pt")
    ckpt_b = os.path.join(ckpt_dir, f"{prefix}_encoder_b.pt")
    if not os.path.isfile(ckpt_a) or not os.path.isfile(ckpt_b):
        raise FileNotFoundError("Checkpoint not found. Please train and save weights first.")

    encoder_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    encoder_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    encoder_a.eval()
    encoder_b.eval()

    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    test_path = os.path.join(dataset_root, "test")
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"test path not found: {test_path}")

    graphs = read_graph_dir(test_path)
    if not graphs:
        raise RuntimeError("No test graphs found.")

    args = _build_args(config)
    test_dataset = create_dataset(graphs, args, is_train=False)
    pairs = _build_pair_list(test_dataset, rrwp_k_steps=args.rrwp_k_steps)
    if not pairs:
        raise RuntimeError("No valid layer pairs found in test set.")

    eval_samples = dataset_cfg.get("eval_samples", 10)
    if eval_samples > 0:
        pairs = pairs[:eval_samples]

    # 累计指标
    recon_auc_a_list, recon_auc_b_list = [], []
    recon_prec_a_list, recon_prec_b_list = [], []
    hub_center_sim_list, hub_max_sim_list = [], []
    periph_center_sim_list, role_sep_list = [], []
    rank_corr_list = []

    print("=" * 60)
    print("GMA-MFEA 模型评估")
    print("=" * 60)

    with torch.no_grad():
        for idx, (data_a, data_b) in enumerate(pairs):
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            
            # 层内重建指标
            auc_a, prec_a = compute_reconstruction_metrics(z_a, data_a.edge_index, z_a.size(0))
            auc_b, prec_b = compute_reconstruction_metrics(z_b, data_b.edge_index, z_b.size(0))
            recon_auc_a_list.append(auc_a)
            recon_auc_b_list.append(auc_b)
            recon_prec_a_list.append(prec_a)
            recon_prec_b_list.append(prec_b)
            
            # 跨层对齐指标
            align_metrics = compute_cross_layer_alignment_metrics(z_a, z_b, alpha_a, alpha_b)
            hub_center_sim_list.append(align_metrics["hub_center_sim"])
            hub_max_sim_list.append(align_metrics["hub_max_sim"])
            periph_center_sim_list.append(align_metrics["periph_center_sim"])
            role_sep_list.append(align_metrics["role_separation"])
            rank_corr_list.append(align_metrics["rank_correlation"])

    # 汇总输出
    print(f"\n📊 测试样本数: {len(pairs)}")
    
    print("\n" + "=" * 60)
    print("【层内重建指标】")
    print("=" * 60)
    print(f"  Layer A - AUC:         {np.mean(recon_auc_a_list):.4f} ± {np.std(recon_auc_a_list):.4f}")
    print(f"  Layer A - Precision@K: {np.mean(recon_prec_a_list):.4f} ± {np.std(recon_prec_a_list):.4f}")
    print(f"  Layer B - AUC:         {np.mean(recon_auc_b_list):.4f} ± {np.std(recon_auc_b_list):.4f}")
    print(f"  Layer B - Precision@K: {np.mean(recon_prec_b_list):.4f} ± {np.std(recon_prec_b_list):.4f}")
    avg_recon_auc = (np.mean(recon_auc_a_list) + np.mean(recon_auc_b_list)) / 2
    print(f"  ✅ 平均重建 AUC:       {avg_recon_auc:.4f}")
    
    print("\n" + "=" * 60)
    print("【跨层功能角色对齐指标】")
    print("=" * 60)
    print(f"  Hub质心相似度:         {np.mean(hub_center_sim_list):.4f} ± {np.std(hub_center_sim_list):.4f}")
    print(f"  Hub最大匹配相似度:     {np.mean(hub_max_sim_list):.4f} ± {np.std(hub_max_sim_list):.4f}")
    print(f"  Peripheral质心相似度:  {np.mean(periph_center_sim_list):.4f} ± {np.std(periph_center_sim_list):.4f}")
    print(f"  角色分离度:            {np.mean(role_sep_list):.4f} ± {np.std(role_sep_list):.4f}")
    print(f"  α排名相关性:           {np.mean(rank_corr_list):.4f} ± {np.std(rank_corr_list):.4f}")
    
    # 综合评分
    hub_align_score = (np.mean(hub_center_sim_list) + np.mean(hub_max_sim_list)) / 2
    print(f"\n  ✅ Hub对齐综合得分:    {hub_align_score:.4f}")
    
    print("\n" + "=" * 60)
    print("【评估解读】")
    print("=" * 60)
    print("  • 重建AUC > 0.8: 层内拓扑保持良好")
    print("  • Hub质心相似度 > 0.6: 高影响力节点跨层对齐良好")
    print("  • α排名相关性 > 0.3: 功能角色跨层一致性好")
    print("  • 角色分离度 > 0.3: Hub与Peripheral区分明显")


if __name__ == "__main__":
    main()
