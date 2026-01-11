#!/usr/bin/env python
"""
GMA-MFEA 两阶段训练入口
========================
放在项目根目录，直接运行: python train_twostage.py

阶段1: 纯重建训练 - 让编码器学会保持层内拓扑结构
阶段2: 固定权重联合训练 - 在重建基础上引入角色对齐
"""

import os
import sys
import yaml
import numpy as np
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from code.gma_core.kaa_grit.kaa_grit_encoder import KaaGritEncoder
from code.gma_core.loss_ammd import AttentionAwareMMDLoss
from code.gma_core.loss_recon import GraphReconstructionLoss
from code.gma_core.loss_contrastive import CrossLayerInfoNCELoss
from code.gma_core.loss_total import AlignmentReconstructionLoss
from code.gma_core.alignment import ManifoldAlignment
from code.dataset.create_dataset import create_dataset
from code.utils.data_loader import read_graph_dir, build_pyg_data_list


def build_pair_list(dataset, rrwp_k_steps):
    """构建层对列表"""
    pairs = []
    for entry in dataset:
        data_list = build_pyg_data_list(entry, rrwp_k_steps=rrwp_k_steps)
        if len(data_list) < 2:
            continue
        pairs.append((data_list[0], data_list[1]))
    return pairs


def collate_pairs(batch):
    """批量整理层对，确保所有数据在CPU上"""
    data_a_list, data_b_list = zip(*batch)
    
    # 确保所有数据在CPU上再进行batch
    data_a_cpu = []
    data_b_cpu = []
    for da, db in zip(data_a_list, data_b_list):
        # 移动到CPU
        da_cpu = da.cpu()
        db_cpu = db.cpu()
        data_a_cpu.append(da_cpu)
        data_b_cpu.append(db_cpu)
    
    return Batch.from_data_list(data_a_cpu), Batch.from_data_list(data_b_cpu)


def compute_recon_auc(z, edge_index, num_nodes, num_neg_samples=5000):
    """快速计算重建AUC"""
    z_norm = torch.nn.functional.normalize(z, p=2, dim=-1)
    
    src, dst = edge_index[0], edge_index[1]
    pos_scores = (z_norm[src] * z_norm[dst]).sum(dim=-1)
    
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=z.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=z.device)
    neg_scores = (z_norm[neg_src] * z_norm[neg_dst]).sum(dim=-1)
    
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])
    
    try:
        auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    except:
        auc = 0.5
    return auc


def evaluate_reconstruction(encoder_a, encoder_b, eval_pairs, device, max_samples=10):
    """评估重建AUC"""
    encoder_a.eval()
    encoder_b.eval()
    
    auc_a_list, auc_b_list = [], []
    
    with torch.no_grad():
        for data_a, data_b in eval_pairs[:max_samples]:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            
            z_a, _, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, _, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            
            auc_a = compute_recon_auc(z_a, data_a.edge_index, z_a.size(0))
            auc_b = compute_recon_auc(z_b, data_b.edge_index, z_b.size(0))
            
            auc_a_list.append(auc_a)
            auc_b_list.append(auc_b)
    
    mean_auc = (np.mean(auc_a_list) + np.mean(auc_b_list)) / 2
    return mean_auc, np.mean(auc_a_list), np.mean(auc_b_list)


def train_stage1(encoder_a, encoder_b, loss_fn, pair_list, config, device):
    """
    阶段1: 纯重建训练
    """
    gma_cfg = config.get("gma", {})
    
    epochs = gma_cfg.get("stage1_epochs", 100)
    lr = gma_cfg.get("stage1_lr", 0.0008)
    lambda_recon = gma_cfg.get("stage1_lambda_recon", 2.0)
    target_auc = gma_cfg.get("stage1_target_auc", 0.78)
    batch_size = gma_cfg.get("batch_size", 8)
    grad_accum_steps = gma_cfg.get("grad_accum_steps", 4)
    log_every = gma_cfg.get("log_every", 10)
    weight_decay = gma_cfg.get("weight_decay", 0.001)
    
    print("=" * 60)
    print("【阶段1】纯重建训练")
    print(f"  epochs: {epochs}, lr: {lr}, lambda_recon: {lambda_recon}")
    print(f"  目标AUC: {target_auc}")
    print("=" * 60)
    
    loss_fn.set_lambdas(
        lambda_align=0.0,
        lambda_recon=lambda_recon,
        lambda_contrastive=0.0,
        lambda_role=0.0,
    )
    
    optimizer = torch.optim.Adam(
        list(encoder_a.parameters()) + list(encoder_b.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    encoder_a.train()
    encoder_b.train()
    
    best_auc = 0.0
    best_state_a, best_state_b = None, None
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_pairs = 0
        
        loader = DataLoader(
            pair_list,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_pairs,
        )
        
        optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0
        
        for data_a, data_b in loader:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            
            loss, loss_dict = loss_fn(
                z_a, z_b, alpha_a, alpha_b,
                data_a.edge_index, data_b.edge_index,
                batch_a=getattr(data_a, "batch", None),
                batch_b=getattr(data_b, "batch", None),
            )
            
            (loss / grad_accum_steps).backward()
            step_in_epoch += 1
            
            if step_in_epoch % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            total_recon += loss_dict["loss_recon"].item()
            total_pairs += 1
        
        if step_in_epoch % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            avg_loss = total_loss / max(1, total_pairs)
            avg_recon = total_recon / max(1, total_pairs)
            
            mean_auc, auc_a, auc_b = evaluate_reconstruction(
                encoder_a, encoder_b, pair_list[:10], device
            )
            
            print(f"[Stage1 Epoch {epoch:03d}] loss: {avg_loss:.4f}, "
                  f"recon: {avg_recon:.4f}, AUC: {mean_auc:.4f} (A:{auc_a:.3f}, B:{auc_b:.3f})")
            
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_state_a = {k: v.cpu().clone() for k, v in encoder_a.state_dict().items()}
                best_state_b = {k: v.cpu().clone() for k, v in encoder_b.state_dict().items()}
            
            encoder_a.train()
            encoder_b.train()
    
    if best_state_a is not None:
        encoder_a.load_state_dict({k: v.to(device) for k, v in best_state_a.items()})
        encoder_b.load_state_dict({k: v.to(device) for k, v in best_state_b.items()})
    
    print(f"\n【阶段1完成】最佳重建AUC: {best_auc:.4f}")
    
    if best_auc < target_auc:
        print(f"⚠️ 警告: AUC ({best_auc:.3f}) 未达到目标 ({target_auc:.3f})，但仍继续阶段2")
    else:
        print(f"✅ AUC达标，进入阶段2")
    
    return best_auc, best_state_a, best_state_b


def train_stage2(encoder_a, encoder_b, loss_fn, pair_list, config, device):
    """
    阶段2: 固定权重联合训练
    """
    gma_cfg = config.get("gma", {})
    
    epochs = gma_cfg.get("stage2_epochs", 200)
    lr = gma_cfg.get("stage2_lr", 0.0003)
    lambda_recon = gma_cfg.get("stage2_lambda_recon", 1.5)
    lambda_align = gma_cfg.get("stage2_lambda_align", 0.3)
    lambda_role = gma_cfg.get("stage2_lambda_role", 1.2)
    lambda_contrastive = gma_cfg.get("stage2_lambda_contrastive", 0.0)
    batch_size = gma_cfg.get("batch_size", 8)
    grad_accum_steps = gma_cfg.get("grad_accum_steps", 4)
    log_every = gma_cfg.get("log_every", 10)
    weight_decay = gma_cfg.get("weight_decay", 0.001)
    
    print("\n" + "=" * 60)
    print("【阶段2】联合训练（固定权重）")
    print(f"  epochs: {epochs}, lr: {lr}")
    print(f"  lambda_recon: {lambda_recon}, lambda_align: {lambda_align}, lambda_role: {lambda_role}")
    print("=" * 60)
    
    loss_fn.set_lambdas(
        lambda_align=lambda_align,
        lambda_recon=lambda_recon,
        lambda_contrastive=lambda_contrastive,
        lambda_role=lambda_role,
    )
    
    optimizer = torch.optim.Adam(
        list(encoder_a.parameters()) + list(encoder_b.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    encoder_a.train()
    encoder_b.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_align = 0.0
        total_recon = 0.0
        total_role = 0.0
        total_pairs = 0
        
        loader = DataLoader(
            pair_list,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_pairs,
        )
        
        optimizer.zero_grad(set_to_none=True)
        step_in_epoch = 0
        
        for data_a, data_b in loader:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            
            loss, loss_dict = loss_fn(
                z_a, z_b, alpha_a, alpha_b,
                data_a.edge_index, data_b.edge_index,
                batch_a=getattr(data_a, "batch", None),
                batch_b=getattr(data_b, "batch", None),
            )
            
            (loss / grad_accum_steps).backward()
            step_in_epoch += 1
            
            if step_in_epoch % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            total_align += loss_dict["loss_align"].item()
            total_recon += loss_dict["loss_recon"].item()
            total_role += loss_dict.get("loss_role", loss.new_tensor(0.0)).item()
            total_pairs += 1
        
        if step_in_epoch % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            avg_loss = total_loss / max(1, total_pairs)
            avg_align = total_align / max(1, total_pairs)
            avg_recon = total_recon / max(1, total_pairs)
            avg_role = total_role / max(1, total_pairs)
            
            print(f"[Stage2 Epoch {epoch:03d}] total: {avg_loss:.4f}, "
                  f"MMD: {avg_align:.4f}, recon: {avg_recon:.4f}, role: {avg_role:.4f}")
    
    encoder_a.eval()
    encoder_b.eval()
    
    mean_auc, auc_a, auc_b = evaluate_reconstruction(
        encoder_a, encoder_b, pair_list[:10], device
    )
    print(f"\n【阶段2完成】最终重建AUC: {mean_auc:.4f} (A:{auc_a:.3f}, B:{auc_b:.3f})")
    
    return mean_auc


def main():
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "config_twostage.yaml")
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"使用配置: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    device = config.get("project", {}).get("device", "cuda:4")
    device = torch.device(device)
    
    print("\n" + "=" * 70)
    print(" GMA-MFEA 两阶段训练")
    print("=" * 70)
    print(f"设备: {device}")
    
    # 模型参数
    in_dim = gma_cfg.get("input_dim", 304)
    hidden_dim = gma_cfg.get("hidden_dim", 512)
    out_dim = gma_cfg.get("output_dim", 256)
    num_heads = gma_cfg.get("heads", 16)
    num_layers = gma_cfg.get("layers", 4)
    k_steps = gma_cfg.get("k_steps", 5)
    dropout = gma_cfg.get("dropout", 0.15)
    num_groups = gma_cfg.get("num_groups", 8)
    groupkan_backend = gma_cfg.get("groupkan_backend", "torch")
    ffn_mult = gma_cfg.get("ffn_mult", 4)
    attn_act = gma_cfg.get("attn_act", "relu")
    attn_clamp = gma_cfg.get("attn_clamp", 5.0)
    signed_sqrt = gma_cfg.get("signed_sqrt", True)
    edge_enhance = gma_cfg.get("edge_enhance", True)
    
    # 加载数据
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    train_path = os.path.join(dataset_root, "train")
    
    if not os.path.isdir(train_path):
        raise RuntimeError(f"训练数据目录不存在: {train_path}")
    
    graphs = read_graph_dir(train_path)
    if not graphs:
        raise RuntimeError("未找到训练图数据")
    
    args = SimpleNamespace(
        gcn_inchannel=dataset_cfg.get("gcn_inchannel", gma_cfg.get("centrality_dim", 12)),
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
        lap_pe_dim=dataset_cfg.get("lap_pe_dim", gma_cfg.get("lap_pe_dim", 36)),
        rrwp_k_steps=dataset_cfg.get("rrwp_k_steps", gma_cfg.get("rrwp_k_steps", k_steps)),
        use_node2vec=dataset_cfg.get("use_node2vec", True),
        preprocess_workers=dataset_cfg.get("preprocess_workers", 4),
        label_path=dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/"),
        is_cover_label=dataset_cfg.get("is_cover_label", False),
        skip_label=True,
    )
    
    dataset = create_dataset(graphs, args, is_train=True)
    pair_list = build_pair_list(dataset, rrwp_k_steps=args.rrwp_k_steps)
    
    if not pair_list:
        raise RuntimeError("未能构建训练对")
    
    print(f"训练数据: {len(pair_list)} 对多层图")
    
    # 创建编码器
    encoder_a = KaaGritEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        k_steps=k_steps,
        dropout=dropout,
        num_groups=num_groups,
        groupkan_backend=groupkan_backend,
        ffn_mult=ffn_mult,
        attn_act=attn_act,
        attn_clamp=attn_clamp,
        signed_sqrt=signed_sqrt,
        edge_enhance=edge_enhance,
        device=device.type,
    ).to(device)
    
    encoder_b = KaaGritEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        k_steps=k_steps,
        dropout=dropout,
        num_groups=num_groups,
        groupkan_backend=groupkan_backend,
        ffn_mult=ffn_mult,
        attn_act=attn_act,
        attn_clamp=attn_clamp,
        signed_sqrt=signed_sqrt,
        edge_enhance=edge_enhance,
        device=device.type,
    ).to(device)
    
    # 创建损失函数
    recon_loss = GraphReconstructionLoss(
        neg_ratio=gma_cfg.get("recon_neg_ratio", 3.0),
        logit_scale=gma_cfg.get("recon_logit_scale", 4.0),
        use_l2_norm=gma_cfg.get("recon_use_l2_norm", True),
        two_hop_ratio=gma_cfg.get("recon_two_hop_ratio", 0.3),
        deg_norm=gma_cfg.get("recon_deg_norm", True),
        hard_neg_ratio=gma_cfg.get("recon_hard_neg_ratio", 0.2),
        hard_pool_ratio=gma_cfg.get("recon_hard_pool_ratio", 5.0),
    )
    
    ammd_loss = AttentionAwareMMDLoss(
        max_samples=gma_cfg.get("ammd_max_samples", 1024),
        sample_mode=gma_cfg.get("ammd_sample_mode", "topk"),
    )
    
    contrastive_loss = CrossLayerInfoNCELoss(
        temperature=gma_cfg.get("contrastive_temp", 0.2),
        max_samples=0,
        use_projection=False,
    ).to(device)
    
    loss_fn = AlignmentReconstructionLoss(
        lambda_align=0.0,
        lambda_recon=2.0,
        lambda_contrastive=0.0,
        lambda_role=0.0,
        ammd_loss=ammd_loss,
        recon_loss=recon_loss,
        contrastive_loss=contrastive_loss,
        role_temperature=gma_cfg.get("role_temperature", 0.07),
        role_alpha_temperature=gma_cfg.get("role_alpha_temperature", 0.3),
        role_topk_ratio=gma_cfg.get("role_topk_ratio", 0.2),
        role_use_hub_contrastive=gma_cfg.get("role_use_hub_contrastive", True),
        role_use_alpha_matching=gma_cfg.get("role_use_alpha_matching", True),
        role_use_prototype=gma_cfg.get("role_use_prototype", True),
        role_num_prototypes=gma_cfg.get("role_num_prototypes", 5),
    )
    
    # ========== 阶段1: 纯重建 ==========
    stage1_auc, state_a, state_b = train_stage1(
        encoder_a, encoder_b, loss_fn, pair_list, config, device
    )
    
    # 保存阶段1权重
    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit_twostage")
    
    torch.save(encoder_a.state_dict(), os.path.join(ckpt_dir, f"{checkpoint_prefix}_stage1_a.pt"))
    torch.save(encoder_b.state_dict(), os.path.join(ckpt_dir, f"{checkpoint_prefix}_stage1_b.pt"))
    print(f"阶段1权重已保存")
    
    # ========== 阶段2: 联合训练 ==========
    stage2_auc = train_stage2(
        encoder_a, encoder_b, loss_fn, pair_list, config, device
    )
    
    # 保存最终权重
    torch.save(encoder_a.state_dict(), os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_a.pt"))
    torch.save(encoder_b.state_dict(), os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_b.pt"))
    print(f"最终权重已保存: {checkpoint_prefix}_encoder_[a|b].pt")
    
    # 保存对齐矩阵
    data_list = build_pyg_data_list(dataset[0], rrwp_k_steps=args.rrwp_k_steps)
    data_a, data_b = data_list[0].to(device), data_list[1].to(device)
    
    with torch.no_grad():
        z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
        z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
        s_align = ManifoldAlignment.compute_similarity_matrix(z_a, z_b)
    
    data_root = config.get("paths", {}).get("data_root", "./data")
    s_align_dir = os.path.join(data_root, "alignment", "similarity_matrix")
    os.makedirs(s_align_dir, exist_ok=True)
    s_align_path = os.path.join(s_align_dir, "S_align_twostage.npy")
    np.save(s_align_path, s_align)
    
    print("\n" + "=" * 70)
    print(" 训练完成!")
    print(f" 阶段1 AUC: {stage1_auc:.4f}")
    print(f" 阶段2 AUC: {stage2_auc:.4f}")
    print(f" S_align 已保存: {s_align_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
