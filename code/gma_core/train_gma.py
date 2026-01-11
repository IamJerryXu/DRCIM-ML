import os
import numpy as np
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from .kaa_grit.kaa_grit_encoder import KaaGritEncoder
from .loss_ammd import AttentionAwareMMDLoss
from .loss_recon import GraphReconstructionLoss
from .loss_contrastive import CrossLayerInfoNCELoss
from .loss_total import AlignmentReconstructionLoss
from .alignment import ManifoldAlignment
from ..dataset.create_rrwp import compute_rrwp
from ..dataset.create_dataset import create_dataset
from ..utils.data_loader import read_graph_dir, read_graph_file, build_pyg_data_list


def _compute_lambda_schedule(epoch, epochs, gma_cfg):
    """计算当前epoch的各损失权重（支持预热和动态调度）"""
    schedule = gma_cfg.get("lambda_schedule", "fixed")
    lambda_align = gma_cfg.get("lambda_align", 1.0)
    lambda_recon = gma_cfg.get("lambda_recon", 1.0)
    lambda_contrastive = gma_cfg.get("lambda_contrastive", 0.0)
    lambda_role = gma_cfg.get("lambda_role", 0.0)
    align_start = gma_cfg.get("lambda_align_start", lambda_align * 0.1)
    align_end = gma_cfg.get("lambda_align_end", lambda_align)
    recon_start = gma_cfg.get("lambda_recon_start", lambda_recon)
    recon_end = gma_cfg.get("lambda_recon_end", lambda_recon * 0.1)
    contrastive_start = gma_cfg.get("lambda_contrastive_start", lambda_contrastive)
    contrastive_end = gma_cfg.get("lambda_contrastive_end", lambda_contrastive)
    role_start = gma_cfg.get("lambda_role_start", lambda_role * 0.1)
    role_end = gma_cfg.get("lambda_role_end", lambda_role)
    pretrain_epochs = gma_cfg.get("recon_pretrain_epochs", 0)

    if epoch <= pretrain_epochs:
        # 预训练阶段：只做重建，不做对齐
        return 0.0, recon_start, 0.0, 0.0

    remain_epochs = max(1, epochs - pretrain_epochs)
    adj_epoch = epoch - pretrain_epochs

    if schedule == "fixed" or remain_epochs <= 1:
        return lambda_align, lambda_recon, lambda_contrastive, lambda_role

    t = float(adj_epoch - 1) / max(1, remain_epochs - 1)
    if schedule == "cosine":
        t = 0.5 * (1.0 - np.cos(np.pi * t))

    align = align_start + t * (align_end - align_start)
    recon = recon_start + t * (recon_end - recon_start)
    contrastive = contrastive_start + t * (contrastive_end - contrastive_start)
    role = role_start + t * (role_end - role_start)
    return align, recon, contrastive, role


def _build_dummy_data(num_nodes, in_dim, k_steps, device):
    row = torch.arange(num_nodes, dtype=torch.long)
    col = torch.roll(row, shifts=-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.randn(num_nodes, in_dim, device=device)

    rrwp = compute_rrwp(edge_index.cpu(), num_nodes, k_steps=k_steps).to(device)
    return Data(x=x, edge_index=edge_index.to(device), rrwp=rrwp)


def _build_pair_list(dataset, rrwp_k_steps):
    pairs = []
    for entry in dataset:
        data_list = build_pyg_data_list(entry, rrwp_k_steps=rrwp_k_steps)
        if len(data_list) < 2:
            continue
        pairs.append((data_list[0], data_list[1]))
    return pairs


def _collate_pairs(batch):
    data_a_list, data_b_list = zip(*batch)
    return Batch.from_data_list(data_a_list), Batch.from_data_list(data_b_list)


def _run_eval_pairs(encoder_a, encoder_b, loss_fn, eval_pairs, device, label):
    if not eval_pairs:
        return
    eval_loss = 0.0
    eval_align = 0.0
    eval_recon = 0.0
    eval_contrast = 0.0
    encoder_a.eval()
    encoder_b.eval()
    loss_fn.eval()
    with torch.no_grad():
        for data_a, data_b in eval_pairs:
            data_a = data_a.to(device)
            data_b = data_b.to(device)
            z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
            z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
            loss, loss_dict = loss_fn(
                z_a,
                z_b,
                alpha_a,
                alpha_b,
                data_a.edge_index,
                data_b.edge_index,
                batch_a=getattr(data_a, "batch", None),
                batch_b=getattr(data_b, "batch", None),
            )
            eval_loss += loss.item()
            eval_align += loss_dict["loss_align"].item()
            eval_recon += loss_dict["loss_recon"].item()
            eval_contrast += loss_dict["loss_contrastive"].item()
    denom = max(1, len(eval_pairs))
    print(
        "{} total: {:.6f}, A-MMD: {:.6f}, recon: {:.6f}, InfoNCE: {:.6f}".format(
            label,
            eval_loss / denom,
            eval_align / denom,
            eval_recon / denom,
            eval_contrast / denom,
        )
    )


def train_gma_model(config):
    """
    Unsupervised alignment: train KAA-GRIT encoders with A-MMD + recon loss.
    """
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    device = config.get("project", {}).get("device", "cpu")
    device = torch.device(device)

    in_dim = gma_cfg.get("input_dim", 16)
    hidden_dim = gma_cfg.get("hidden_dim", 64)
    out_dim = gma_cfg.get("output_dim", 32)
    num_heads = gma_cfg.get("heads", 4)
    num_layers = gma_cfg.get("layers", 2)
    k_steps = gma_cfg.get("k_steps", 5)
    dropout = gma_cfg.get("dropout", 0.1)
    num_groups = gma_cfg.get("num_groups", 8)
    epochs = gma_cfg.get("epochs", 0)
    lr = gma_cfg.get("lr", 1e-3)
    weight_decay = gma_cfg.get("weight_decay", 0.0)
    log_every = gma_cfg.get("log_every", 10)
    lambda_align = gma_cfg.get("lambda_align", 1.0)
    lambda_recon = gma_cfg.get("lambda_recon", 1.0)
    lambda_contrastive = gma_cfg.get("lambda_contrastive", 0.0)
    recon_neg_ratio = gma_cfg.get("recon_neg_ratio", 1.0)
    recon_logit_scale = gma_cfg.get("recon_logit_scale", 1.0)
    recon_use_l2_norm = gma_cfg.get("recon_use_l2_norm", True)
    recon_two_hop_ratio = gma_cfg.get("recon_two_hop_ratio", 0.0)
    recon_deg_norm = gma_cfg.get("recon_deg_norm", True)
    recon_hard_neg_ratio = gma_cfg.get("recon_hard_neg_ratio", 0.0)
    recon_hard_pool_ratio = gma_cfg.get("recon_hard_pool_ratio", 1.0)
    lambda_schedule = gma_cfg.get("lambda_schedule", "fixed")
    batch_size = gma_cfg.get("batch_size", 1)
    contrastive_temp = gma_cfg.get("contrastive_temp", 0.2)
    contrastive_max_samples = gma_cfg.get("contrastive_max_samples", 0)
    use_projection = gma_cfg.get("use_projection", False)
    proj_hidden_dim = gma_cfg.get("proj_hidden_dim", 0)
    proj_out_dim = gma_cfg.get("proj_out_dim", 0)
    proj_dropout = gma_cfg.get("proj_dropout", 0.0)
    contrastive_pos_mode = gma_cfg.get("contrastive_pos_mode", "id")
    contrastive_topk = gma_cfg.get("contrastive_topk", 1)
    groupkan_backend = gma_cfg.get("groupkan_backend", "triton")
    ffn_mult = gma_cfg.get("ffn_mult", 4)
    attn_act = gma_cfg.get("attn_act", "relu")
    attn_clamp = gma_cfg.get("attn_clamp", 5.0)
    signed_sqrt = gma_cfg.get("signed_sqrt", True)
    edge_enhance = gma_cfg.get("edge_enhance", True)
    grad_accum_steps = gma_cfg.get("grad_accum_steps", 1)
    save_weights = gma_cfg.get("save_weights", True)
    checkpoint_prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit")
    # Role-based alignment params (NEW)
    lambda_role = gma_cfg.get("lambda_role", 0.0)
    role_temperature = gma_cfg.get("role_temperature", 0.1)
    role_alpha_temperature = gma_cfg.get("role_alpha_temperature", 0.5)
    role_topk_ratio = gma_cfg.get("role_topk_ratio", 0.1)
    role_use_hub_contrastive = gma_cfg.get("role_use_hub_contrastive", True)
    role_use_alpha_matching = gma_cfg.get("role_use_alpha_matching", True)
    role_use_prototype = gma_cfg.get("role_use_prototype", True)
    role_num_prototypes = gma_cfg.get("role_num_prototypes", 3)

    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    train_path = os.path.join(dataset_root, "train")
    single_file = dataset_cfg.get("single_file", "")
    use_real_data = os.path.isdir(train_path)

    if single_file and os.path.isfile(single_file):
        # 读取单个多层 edgelist 文件
        graphs = read_graph_file(single_file)
        use_real_data = True

    if use_real_data:
        # 读取多层图并生成特征缓存
        if "graphs" not in locals():
            graphs = read_graph_dir(train_path)
        if not graphs:
            use_real_data = False
        else:
            args = SimpleNamespace(
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
                rrwp_k_steps=dataset_cfg.get("rrwp_k_steps", gma_cfg.get("rrwp_k_steps", k_steps)),
                use_node2vec=dataset_cfg.get("use_node2vec", False),
                preprocess_workers=dataset_cfg.get("preprocess_workers", 0),
                label_path=dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/"),
                is_cover_label=dataset_cfg.get("is_cover_label", False),
                skip_label=True,
            )

            dataset = create_dataset(graphs, args, is_train=True)
            if not dataset:
                use_real_data = False
            else:
                pair_list = _build_pair_list(dataset, rrwp_k_steps=args.rrwp_k_steps)
                if not pair_list:
                    use_real_data = False

    if not use_real_data:
        data_a = _build_dummy_data(32, in_dim, k_steps, device)
        data_b = _build_dummy_data(32, in_dim, k_steps, device)
        epochs = 0

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

    ammd_loss = AttentionAwareMMDLoss(
        max_samples=gma_cfg.get("ammd_max_samples", 0),
        sample_mode=gma_cfg.get("ammd_sample_mode", "topk"),
    )
    recon_loss = GraphReconstructionLoss(
        neg_ratio=recon_neg_ratio,
        logit_scale=recon_logit_scale,
        use_l2_norm=recon_use_l2_norm,
        two_hop_ratio=recon_two_hop_ratio,
        deg_norm=recon_deg_norm,
        hard_neg_ratio=recon_hard_neg_ratio,
        hard_pool_ratio=recon_hard_pool_ratio,
    )
    contrastive_loss = CrossLayerInfoNCELoss(
        temperature=contrastive_temp,
        max_samples=contrastive_max_samples,
        use_projection=use_projection,
        in_dim=out_dim,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        proj_dropout=proj_dropout,
        pos_mode=contrastive_pos_mode,
        topk=contrastive_topk,
    ).to(device)
    loss_fn = AlignmentReconstructionLoss(
        lambda_align=lambda_align,
        lambda_recon=lambda_recon,
        lambda_contrastive=lambda_contrastive,
        lambda_role=lambda_role,
        ammd_loss=ammd_loss,
        recon_loss=recon_loss,
        contrastive_loss=contrastive_loss,
        role_temperature=role_temperature,
        role_alpha_temperature=role_alpha_temperature,
        role_topk_ratio=role_topk_ratio,
        role_use_hub_contrastive=role_use_hub_contrastive,
        role_use_alpha_matching=role_use_alpha_matching,
        role_use_prototype=role_use_prototype,
        role_num_prototypes=role_num_prototypes,
    )

    if use_real_data and epochs > 0:
        optimizer_params = list(encoder_a.parameters()) + list(encoder_b.parameters())
        if any(p.requires_grad for p in contrastive_loss.parameters()):
            optimizer_params += list(contrastive_loss.parameters())
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        encoder_a.train()
        encoder_b.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_align = 0.0
            total_recon = 0.0
            total_contrast = 0.0
            total_role = 0.0
            total_pairs = 0
            loader = DataLoader(
                pair_list,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_collate_pairs,
            )
            optimizer.zero_grad(set_to_none=True)
            step_in_epoch = 0
            for data_a, data_b in loader:
                data_a = data_a.to(device)
                data_b = data_b.to(device)

                cur_align, cur_recon, cur_contrast, cur_role = _compute_lambda_schedule(epoch, epochs, gma_cfg)
                loss_fn.set_lambdas(
                    lambda_align=cur_align,
                    lambda_recon=cur_recon,
                    lambda_contrastive=cur_contrast,
                    lambda_role=cur_role,
                )

                z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
                z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
                loss, loss_dict = loss_fn(
                    z_a,
                    z_b,
                    alpha_a,
                    alpha_b,
                    data_a.edge_index,
                    data_b.edge_index,
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
                total_contrast += loss_dict["loss_contrastive"].item()
                total_role += loss_dict.get("loss_role", loss.new_tensor(0.0)).item()
                total_pairs += 1

            if step_in_epoch % grad_accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if total_pairs == 0:
                print("No valid layer pairs for training.")
                break
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                avg_loss = total_loss / total_pairs
                avg_align = total_align / total_pairs
                avg_recon = total_recon / total_pairs
                avg_contrast = total_contrast / total_pairs
                avg_role = total_role / total_pairs
                print(
                    f"[Epoch {epoch:03d}] total: {avg_loss:.6f}, "
                    f"A-MMD: {avg_align:.6f}, recon: {avg_recon:.6f}, "
                    f"InfoNCE: {avg_contrast:.6f}, Role: {avg_role:.6f}"
                )
                if lambda_schedule != "fixed":
                    cur_align, cur_recon, cur_contrast, cur_role = _compute_lambda_schedule(epoch, epochs, gma_cfg)
                    print(
                        "  lambdas -> align: {:.4f}, recon: {:.4f}, InfoNCE: {:.4f}, role: {:.4f}".format(
                            cur_align, cur_recon, cur_contrast, cur_role
                        )
                    )

        encoder_a.eval()
        encoder_b.eval()

    # 评估与保存 S_align
    if use_real_data:
        data_list = build_pyg_data_list(dataset[0], rrwp_k_steps=args.rrwp_k_steps)
        if len(data_list) == 1:
            data_a = data_list[0]
            data_b = data_list[0]
        else:
            data_a, data_b = data_list[0], data_list[1]
        data_a = data_a.to(device)
        data_b = data_b.to(device)
        print("=== Phase-1 数据检查 ===")
        print(f"layers: {len(data_list)}")
        print(f"x shape: {tuple(data_a.x.shape)}")
        print(f"edge_index shape: {tuple(data_a.edge_index.shape)}")
        if hasattr(data_a, "rrwp"):
            print(f"rrwp shape: {tuple(data_a.rrwp.shape)}")
        print("======================")

    with torch.no_grad():
        z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
        z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
        loss, loss_dict = loss_fn(
            z_a,
            z_b,
            alpha_a,
            alpha_b,
            data_a.edge_index,
            data_b.edge_index,
            batch_a=getattr(data_a, "batch", None),
            batch_b=getattr(data_b, "batch", None),
        )
        s_align = ManifoldAlignment.compute_similarity_matrix(z_a, z_b)

    # 简单评估: 优先 eval 集，若无则 fallback 到 test 集
    eval_samples = dataset_cfg.get("eval_samples", 10)
    if eval_samples > 0:
        eval_path = os.path.join(dataset_root, "eval")
        test_path = os.path.join(dataset_root, "test")
        eval_done = False
        if os.path.isdir(eval_path):
            eval_graphs = read_graph_dir(eval_path)
            if eval_graphs:
                eval_dataset = create_dataset(eval_graphs, args, is_train=False)
                eval_pairs = _build_pair_list(eval_dataset, rrwp_k_steps=args.rrwp_k_steps)
                _run_eval_pairs(
                    encoder_a,
                    encoder_b,
                    loss_fn,
                    eval_pairs[:eval_samples],
                    device,
                    f"Eval[{eval_samples}]",
                )
                eval_done = True
        if not eval_done and os.path.isdir(test_path):
            test_graphs = read_graph_dir(test_path)
            if test_graphs:
                test_dataset = create_dataset(test_graphs, args, is_train=False)
                test_pairs = _build_pair_list(test_dataset, rrwp_k_steps=args.rrwp_k_steps)
                _run_eval_pairs(
                    encoder_a,
                    encoder_b,
                    loss_fn,
                    test_pairs[:eval_samples],
                    device,
                    f"Test[{eval_samples}]",
                )

    if save_weights:
        ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_a = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_a.pt")
        ckpt_b = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_b.pt")
        ckpt_proj = os.path.join(ckpt_dir, f"{checkpoint_prefix}_proj.pt")
        torch.save(encoder_a.state_dict(), ckpt_a)
        torch.save(encoder_b.state_dict(), ckpt_b)
        print(f"Saved encoder weights: {ckpt_a}")
        print(f"Saved encoder weights: {ckpt_b}")
        if any(p.requires_grad for p in contrastive_loss.parameters()):
            torch.save(contrastive_loss.state_dict(), ckpt_proj)
            print(f"Saved projection head: {ckpt_proj}")

    # 保存 S_align
    data_root = config.get("paths", {}).get("data_root", "./data")
    s_align_dir = os.path.join(data_root, "alignment", "similarity_matrix")
    os.makedirs(s_align_dir, exist_ok=True)
    s_align_path = os.path.join(s_align_dir, "S_align.npy")
    np.save(s_align_path, s_align)

    # 保存 alpha 和 embeddings 用于 MFEA
    align_dir = os.path.join(data_root, "alignment")
    os.makedirs(align_dir, exist_ok=True)
    
    z_a_np = z_a.detach().cpu().numpy()
    z_b_np = z_b.detach().cpu().numpy()
    alpha_a_np = alpha_a.detach().cpu().numpy()
    alpha_b_np = alpha_b.detach().cpu().numpy()
    
    np.save(os.path.join(align_dir, "embeddings_l1.npy"), z_a_np)
    np.save(os.path.join(align_dir, "embeddings_l2.npy"), z_b_np)
    np.save(os.path.join(align_dir, "alpha_l1.npy"), alpha_a_np)
    np.save(os.path.join(align_dir, "alpha_l2.npy"), alpha_b_np)
    
    print(f"Saved embeddings: L1={z_a_np.shape}, L2={z_b_np.shape}")
    print(f"Saved alpha: L1={alpha_a_np.shape}, L2={alpha_b_np.shape}")

    print("Unsupervised alignment run ok")
    print(f"z_a: {tuple(z_a.shape)}, z_b: {tuple(z_b.shape)}")
    print(f"alpha_a: {tuple(alpha_a.shape)}, alpha_b: {tuple(alpha_b.shape)}")
    print(
        "loss_total: {:.6f}, A-MMD: {:.6f}, recon: {:.6f}".format(
            loss.item(),
            loss_dict["loss_align"].item(),
            loss_dict["loss_recon"].item(),
        )
    )
    print(f"S_align saved: {s_align_path}")

    return z_a, z_b, alpha_a, alpha_b, loss, s_align_path
