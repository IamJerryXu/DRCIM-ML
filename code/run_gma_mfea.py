"""
GMA-MFEA Main Entry Point.

Complete pipeline for multiplex network robust influence maximization:
1. Phase 1 (GMA Pre-training): Train KAA-GRIT encoders to learn S_align, α, embeddings
2. Phase 2 (MFEA Evolution): Use GMA outputs to guide multi-factorial evolutionary search

Usage:
    python -m code.main --config config.yaml --mode all
    python -m code.main --config config.yaml --mode pretrain
    python -m code.main --config config.yaml --mode evolution --s_align_path data/alignment/...
"""

import os
import argparse
import yaml
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any

from code.gma_core.train_gma import train_gma_model
from code.mfea_core.run_mfea import run_mfea_solver
from code.utils.data_loader import read_graph_dir, read_graph_file
from code.utils.graph_ops import get_adjacency_list


def load_gma_outputs(
    config: Dict,
    s_align_path: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load pre-trained GMA outputs (S_align, embeddings, alpha).
    
    Args:
        config: Configuration dictionary.
        s_align_path: Path to S_align.npy (optional, will use default if None).
        checkpoint_prefix: Prefix for checkpoint files.
    
    Returns:
        Dictionary containing s_align, alpha_l1, alpha_l2, embeddings_l1, embeddings_l2.
    """
    data_root = config.get("paths", {}).get("data_root", "./data")
    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    
    if checkpoint_prefix is None:
        checkpoint_prefix = config.get("gma", {}).get("checkpoint_prefix", "kaa_grit")
    
    # Load S_align
    if s_align_path is None:
        s_align_path = os.path.join(data_root, "alignment", "similarity_matrix", "S_align.npy")
    
    s_align = None
    if os.path.exists(s_align_path):
        s_align = np.load(s_align_path)
        print(f"[GMA] Loaded S_align from {s_align_path}, shape={s_align.shape}")
    else:
        print(f"[GMA] Warning: S_align not found at {s_align_path}")
    
    # Load alpha (attention weights) if available
    alpha_l1_path = os.path.join(data_root, "alignment", "alpha_l1.npy")
    alpha_l2_path = os.path.join(data_root, "alignment", "alpha_l2.npy")
    
    alpha_l1 = None
    alpha_l2 = None
    if os.path.exists(alpha_l1_path):
        alpha_l1 = np.load(alpha_l1_path)
        print(f"[GMA] Loaded alpha_l1, shape={alpha_l1.shape}")
    if os.path.exists(alpha_l2_path):
        alpha_l2 = np.load(alpha_l2_path)
        print(f"[GMA] Loaded alpha_l2, shape={alpha_l2.shape}")
    
    # Warn if dimensions seem wrong (likely from test data)
    if alpha_l1 is not None and alpha_l1.shape[0] < 100:
        print(f"[GMA] Warning: alpha_l1 size={alpha_l1.shape[0]} seems too small!")
        print(f"[GMA] Consider re-generating with: --mode load_checkpoint --checkpoint <prefix>")
    
    # Load embeddings if available
    emb_l1_path = os.path.join(data_root, "alignment", "embeddings_l1.npy")
    emb_l2_path = os.path.join(data_root, "alignment", "embeddings_l2.npy")
    
    embeddings_l1 = None
    embeddings_l2 = None
    if os.path.exists(emb_l1_path):
        embeddings_l1 = np.load(emb_l1_path)
        print(f"[GMA] Loaded embeddings_l1, shape={embeddings_l1.shape}")
    if os.path.exists(emb_l2_path):
        embeddings_l2 = np.load(emb_l2_path)
        print(f"[GMA] Loaded embeddings_l2, shape={embeddings_l2.shape}")
    
    return {
        "s_align": s_align,
        "alpha_l1": alpha_l1,
        "alpha_l2": alpha_l2,
        "embeddings_l1": embeddings_l1,
        "embeddings_l2": embeddings_l2,
    }


def save_gma_outputs(
    config: Dict,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    alpha_a: torch.Tensor,
    alpha_b: torch.Tensor,
    s_align: np.ndarray,
):
    """
    Save GMA outputs for MFEA consumption.
    
    Args:
        config: Configuration dictionary.
        z_a, z_b: Node embeddings from encoders.
        alpha_a, alpha_b: Attention weights from encoders.
        s_align: Cross-layer similarity matrix.
    """
    data_root = config.get("paths", {}).get("data_root", "./data")
    align_dir = os.path.join(data_root, "alignment")
    os.makedirs(align_dir, exist_ok=True)
    
    # Convert to numpy
    z_a_np = z_a.detach().cpu().numpy()
    z_b_np = z_b.detach().cpu().numpy()
    alpha_a_np = alpha_a.detach().cpu().numpy()
    alpha_b_np = alpha_b.detach().cpu().numpy()
    
    # Save embeddings
    np.save(os.path.join(align_dir, "embeddings_l1.npy"), z_a_np)
    np.save(os.path.join(align_dir, "embeddings_l2.npy"), z_b_np)
    print(f"[GMA] Saved embeddings: L1={z_a_np.shape}, L2={z_b_np.shape}")
    
    # Save alpha weights
    np.save(os.path.join(align_dir, "alpha_l1.npy"), alpha_a_np)
    np.save(os.path.join(align_dir, "alpha_l2.npy"), alpha_b_np)
    print(f"[GMA] Saved alpha: L1={alpha_a_np.shape}, L2={alpha_b_np.shape}")
    
    # S_align is saved by train_gma_model, but ensure it's there
    s_align_dir = os.path.join(align_dir, "similarity_matrix")
    os.makedirs(s_align_dir, exist_ok=True)
    s_align_path = os.path.join(s_align_dir, "S_align.npy")
    if not os.path.exists(s_align_path):
        np.save(s_align_path, s_align)
        print(f"[GMA] Saved S_align: {s_align.shape}")


def load_encoders_from_checkpoint(
    config: Dict,
    checkpoint_prefix: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load pre-trained KAA-GRIT encoders from checkpoint files.
    
    Auto-detects model architecture from checkpoint state_dict.
    
    Args:
        config: Configuration dictionary.
        checkpoint_prefix: Prefix for checkpoint files (e.g., 'kaa_grit_twostage_v2_best').
    
    Returns:
        Tuple of (encoder_a, encoder_b).
    """
    from code.gma_core.kaa_grit.kaa_grit_encoder import KaaGritEncoder
    
    gma_cfg = config.get("gma", {})
    ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
    device = config.get("project", {}).get("device", "cpu")
    device = torch.device(device)
    
    if checkpoint_prefix is None:
        checkpoint_prefix = gma_cfg.get("checkpoint_prefix", "kaa_grit")
    
    # Find checkpoint files
    ckpt_a = os.path.join(ckpt_dir, f"{checkpoint_prefix}_a.pt")
    ckpt_b = os.path.join(ckpt_dir, f"{checkpoint_prefix}_b.pt")
    
    if not os.path.exists(ckpt_a):
        ckpt_a = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_a.pt")
    if not os.path.exists(ckpt_b):
        ckpt_b = os.path.join(ckpt_dir, f"{checkpoint_prefix}_encoder_b.pt")
    
    if not os.path.exists(ckpt_a) or not os.path.exists(ckpt_b):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_a} or {ckpt_b}")
    
    # Load state_dict to infer architecture
    state_dict_a = torch.load(ckpt_a, map_location=device)
    
    # Infer architecture from state_dict
    # Count number of layers
    layer_indices = set()
    for key in state_dict_a.keys():
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    num_layers = len(layer_indices)
    
    # Infer dimensions from weight shapes
    # input_proj.linear.weight shape: [hidden_dim, in_dim]
    in_dim = gma_cfg.get("input_dim", 304)
    if "input_proj.linear.weight" in state_dict_a:
        hidden_dim = state_dict_a["input_proj.linear.weight"].shape[0]
        in_dim = state_dict_a["input_proj.linear.weight"].shape[1]
    else:
        hidden_dim = gma_cfg.get("hidden_dim", 512)
    
    # output_proj.linear.weight shape: [out_dim, hidden_dim]
    if "output_proj.linear.weight" in state_dict_a:
        out_dim = state_dict_a["output_proj.linear.weight"].shape[0]
    else:
        out_dim = gma_cfg.get("output_dim", 256)
    
    # Infer num_heads from Aw shape: [hidden_dim, num_heads]
    if "layers.0.Aw" in state_dict_a:
        aw_shape = state_dict_a["layers.0.Aw"].shape
        num_heads = aw_shape[1]
    else:
        num_heads = gma_cfg.get("heads", 8)
    
    # Infer k_steps from rrwp_proj shape if exists
    k_steps = gma_cfg.get("k_steps", 5)
    if "rrwp_proj.weight" in state_dict_a:
        k_steps = state_dict_a["rrwp_proj.weight"].shape[1]
    
    print(f"[GMA] Inferred architecture from checkpoint:")
    print(f"      in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}")
    print(f"      num_layers={num_layers}, num_heads={num_heads}, k_steps={k_steps}")
    
    # Other params from config (less critical for architecture)
    dropout = gma_cfg.get("dropout", 0.1)
    num_groups = gma_cfg.get("num_groups", 8)
    groupkan_backend = gma_cfg.get("groupkan_backend", "torch")
    ffn_mult = gma_cfg.get("ffn_mult", 4)
    attn_act = gma_cfg.get("attn_act", "relu")
    attn_clamp = gma_cfg.get("attn_clamp", 5.0)
    signed_sqrt = gma_cfg.get("signed_sqrt", True)
    edge_enhance = gma_cfg.get("edge_enhance", True)
    
    # Create encoders with inferred architecture
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
    
    # Load weights
    encoder_a.load_state_dict(state_dict_a)
    encoder_b.load_state_dict(torch.load(ckpt_b, map_location=device))
    print(f"[GMA] Loaded encoder_a from {ckpt_a}")
    print(f"[GMA] Loaded encoder_b from {ckpt_b}")
    
    encoder_a.eval()
    encoder_b.eval()
    
    return encoder_a, encoder_b


def generate_gma_outputs_from_checkpoint(
    config: Dict,
    checkpoint_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load pre-trained encoders and generate S_align, alpha, embeddings.
    
    This skips training and directly uses saved model weights.
    
    Args:
        config: Configuration dictionary.
        checkpoint_prefix: Prefix for checkpoint files.
    
    Returns:
        Dictionary with s_align, alpha_l1, alpha_l2, embeddings_l1, embeddings_l2.
    """
    from types import SimpleNamespace
    from code.gma_core.alignment import ManifoldAlignment
    from code.dataset.create_dataset import create_dataset
    from code.utils.data_loader import read_graph_dir, read_graph_file, build_pyg_data_list
    
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    device = config.get("project", {}).get("device", "cpu")
    device = torch.device(device)
    
    # Load encoders
    encoder_a, encoder_b = load_encoders_from_checkpoint(config, checkpoint_prefix)
    
    # Load graph data
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    train_path = os.path.join(dataset_root, "train")
    single_file = dataset_cfg.get("single_file", "")
    
    if single_file and os.path.isfile(single_file):
        graphs = read_graph_file(single_file)
    elif os.path.isdir(train_path):
        graphs = read_graph_dir(train_path)
    else:
        raise FileNotFoundError(f"No graph data found in {train_path}")
    
    # IMPORTANT: Only use the first graph for inference (no need to process all 55)
    if graphs:
        graphs = [graphs[0]]
        print(f"[GMA] Using first graph for inference: {graphs[0].get('name', 'unknown')}")
    
    # Create dataset with features
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
        rrwp_k_steps=dataset_cfg.get("rrwp_k_steps", gma_cfg.get("rrwp_k_steps", 5)),
        use_node2vec=dataset_cfg.get("use_node2vec", True),
        preprocess_workers=dataset_cfg.get("preprocess_workers", 0),
        label_path=dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/"),
        is_cover_label=dataset_cfg.get("is_cover_label", False),
        skip_label=True,
    )
    
    dataset = create_dataset(graphs, args, is_train=False)
    data_list = build_pyg_data_list(dataset[0], rrwp_k_steps=args.rrwp_k_steps)
    
    if len(data_list) == 1:
        data_a = data_list[0]
        data_b = data_list[0]
    else:
        data_a, data_b = data_list[0], data_list[1]
    
    data_a = data_a.to(device)
    data_b = data_b.to(device)
    
    print(f"[GMA] Data loaded: L1 x={tuple(data_a.x.shape)}, L2 x={tuple(data_b.x.shape)}")
    
    # Generate embeddings and alpha
    with torch.no_grad():
        z_a, alpha_a, _ = encoder_a(data_a.x, data_a.edge_index, data_a.rrwp)
        z_b, alpha_b, _ = encoder_b(data_b.x, data_b.edge_index, data_b.rrwp)
        s_align = ManifoldAlignment.compute_similarity_matrix(z_a, z_b)
    
    print(f"[GMA] Generated: z_a={tuple(z_a.shape)}, z_b={tuple(z_b.shape)}")
    print(f"[GMA] Generated: alpha_a={tuple(alpha_a.shape)}, alpha_b={tuple(alpha_b.shape)}")
    print(f"[GMA] Generated: S_align={s_align.shape}")
    
    # Save outputs
    data_root = config.get("paths", {}).get("data_root", "./data")
    align_dir = os.path.join(data_root, "alignment")
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(os.path.join(align_dir, "similarity_matrix"), exist_ok=True)
    
    z_a_np = z_a.cpu().numpy()
    z_b_np = z_b.cpu().numpy()
    alpha_a_np = alpha_a.cpu().numpy()
    alpha_b_np = alpha_b.cpu().numpy()
    
    np.save(os.path.join(align_dir, "embeddings_l1.npy"), z_a_np)
    np.save(os.path.join(align_dir, "embeddings_l2.npy"), z_b_np)
    np.save(os.path.join(align_dir, "alpha_l1.npy"), alpha_a_np)
    np.save(os.path.join(align_dir, "alpha_l2.npy"), alpha_b_np)
    np.save(os.path.join(align_dir, "similarity_matrix", "S_align.npy"), s_align)
    
    print(f"[GMA] Saved all outputs to {align_dir}")
    
    return {
        "s_align": s_align,
        "alpha_l1": alpha_a_np,
        "alpha_l2": alpha_b_np,
        "embeddings_l1": z_a_np,
        "embeddings_l2": z_b_np,
    }


def load_graph_data(config: Dict) -> Tuple[Any, Any, int]:
    """
    Load graph data for MFEA.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Tuple of (adj_l1, adj_l2, num_nodes).
    """
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    dataset_cfg = config.get("dataset", {})
    single_file = dataset_cfg.get("single_file", "")
    
    # Try single file first
    if single_file and os.path.isfile(single_file):
        graphs = read_graph_file(single_file)
        if len(graphs) >= 2:
            g1, g2 = graphs[0], graphs[1]
            adj_l1, n1 = get_adjacency_list(g1)
            adj_l2, n2 = get_adjacency_list(g2)
            num_nodes = max(n1, n2)
            print(f"[Data] Loaded from {single_file}: L1={n1} nodes, L2={n2} nodes")
            return adj_l1, adj_l2, num_nodes
    
    # Try train directory
    train_path = os.path.join(dataset_root, "train")
    if os.path.isdir(train_path):
        graphs = read_graph_dir(train_path)
        if graphs:
            # Take first multiplex network
            first_entry = graphs[0]
            
            # Handle dict format: {"name": ..., "networks": [g1, g2]}
            if isinstance(first_entry, dict) and "networks" in first_entry:
                networks = first_entry["networks"]
                if len(networks) >= 2:
                    g1, g2 = networks[0], networks[1]
                else:
                    g1 = g2 = networks[0]
            elif isinstance(first_entry, (list, tuple)) and len(first_entry) >= 2:
                g1, g2 = first_entry[0], first_entry[1]
            elif hasattr(first_entry, "layers"):
                g1, g2 = first_entry.layers[0], first_entry.layers[1]
            else:
                # Assume it's a single graph, use same for both layers
                g1 = g2 = first_entry
            
            adj_l1, n1 = get_adjacency_list(g1)
            adj_l2, n2 = get_adjacency_list(g2)
            num_nodes = max(n1, n2)
            print(f"[Data] Loaded from {train_path}: L1={n1} nodes, L2={n2} nodes")
            return adj_l1, adj_l2, num_nodes
    
    print("[Data] Warning: No graph data found, using mock data")
    return None, None, 0


def run_pipeline(
    config: Dict,
    mode: str = "all",
    s_align_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete GMA-MFEA pipeline.
    
    Args:
        config: Configuration dictionary.
        mode: Execution mode ('pretrain', 'evolution', 'all').
        s_align_path: Path to pre-computed S_align (for 'evolution' mode).
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with results from both phases.
    """
    results = {}
    
    # ========== Phase 1: GMA Pre-training ==========
    if mode in ["pretrain", "all"]:
        print("=" * 60)
        print("Phase 1: GMA Pre-training")
        print("=" * 60)
        
        gma_result = train_gma_model(config)
        
        if gma_result is not None:
            z_a, z_b, alpha_a, alpha_b, loss, s_align_path_out = gma_result
            
            results["gma"] = {
                "s_align_path": s_align_path_out,
                "final_loss": loss.item() if hasattr(loss, "item") else loss,
                "embedding_dim": z_a.shape[-1] if hasattr(z_a, "shape") else None,
                "num_nodes_l1": z_a.shape[0] if hasattr(z_a, "shape") else None,
                "num_nodes_l2": z_b.shape[0] if hasattr(z_b, "shape") else None,
            }
            
            print(f"\n[Phase 1] Completed. S_align saved to {s_align_path_out}")
        else:
            print("[Phase 1] Training returned None (possibly no data)")
    
    # ========== Phase 2: MFEA Evolution ==========
    if mode in ["evolution", "all"]:
        print("\n" + "=" * 60)
        print("Phase 2: MFEA Evolution")
        print("=" * 60)
        
        # Load GMA outputs
        gma_outputs = load_gma_outputs(config, s_align_path)
        
        # Load graph data
        adj_l1, adj_l2, num_nodes = load_graph_data(config)
        
        if adj_l1 is None or adj_l2 is None:
            print("[Phase 2] Error: No graph data available")
            return results
        
        # Update config with actual num_nodes
        if "mfea" not in config:
            config["mfea"] = {}
        config["mfea"]["num_nodes"] = num_nodes
        
        # Run MFEA solver
        mfea_result = run_mfea_solver(
            config=config,
            s_align=gma_outputs["s_align"],
            alpha_l1=gma_outputs["alpha_l1"],
            alpha_l2=gma_outputs["alpha_l2"],
            embeddings_l1=gma_outputs["embeddings_l1"],
            embeddings_l2=gma_outputs["embeddings_l2"],
            graph_l1=adj_l1,
            graph_l2=adj_l2,
            adj_l1=adj_l1,
            adj_l2=adj_l2,
            verbose=verbose,
        )
        
        results["mfea"] = mfea_result
        
        # Save final results
        save_final_results(config, mfea_result)
        
        print(f"\n[Phase 2] Completed.")
    
    return results


def save_final_results(config: Dict, mfea_result: Dict):
    """Save MFEA results to disk."""
    results_dir = config.get("paths", {}).get("results", "./results")
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    from datetime import datetime
    
    def to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best seeds
    if mfea_result.get("best_t1"):
        seeds_t1 = {
            "seeds_l1": to_native(mfea_result["best_t1"].seeds_l1),
            "seeds_l2": to_native(mfea_result["best_t1"].seeds_l2),
            "score": float(mfea_result["best_scores"]["T1"]),
        }
        with open(os.path.join(results_dir, f"best_t1_{timestamp}.json"), "w") as f:
            json.dump(seeds_t1, f, indent=2)
    
    if mfea_result.get("best_t3"):
        seeds_t3 = {
            "seeds_l1": to_native(mfea_result["best_t3"].seeds_l1),
            "seeds_l2": to_native(mfea_result["best_t3"].seeds_l2),
            "score": float(mfea_result["best_scores"]["T3"]),
        }
        with open(os.path.join(results_dir, f"best_t3_{timestamp}.json"), "w") as f:
            json.dump(seeds_t3, f, indent=2)
    
    # Save history
    if "history" in mfea_result:
        history_path = os.path.join(results_dir, f"history_{timestamp}.json")
        history_serializable = {}
        for k, v in mfea_result["history"].items():
            if isinstance(v, list):
                history_serializable[k] = [
                    x if not hasattr(x, "tolist") else x.tolist() for x in v
                ]
            else:
                history_serializable[k] = v
        with open(history_path, "w") as f:
            json.dump(history_serializable, f, indent=2)
        print(f"[Results] Saved history to {history_path}")
    
    print(f"[Results] Saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="GMA-MFEA: Multiplex Network Robust Influence Maximization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "evolution", "all", "load_checkpoint"],
        default="all",
        help="Execution mode: pretrain (GMA only), evolution (MFEA only), "
             "all (both), load_checkpoint (load .pt and run MFEA)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint prefix for loading pre-trained encoders "
             "(e.g., 'kaa_grit_twostage_v2_best'). Used with --mode load_checkpoint",
    )
    parser.add_argument(
        "--s_align_path",
        type=str,
        default=None,
        help="Path to pre-computed S_align.npy (for evolution mode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("GMA-MFEA: Multiplex Network Robust Influence Maximization")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {config.get('project', {}).get('device', 'cpu')}")
    print("=" * 60)
    
    results = {}
    
    # Special mode: load from checkpoint
    if args.mode == "load_checkpoint":
        print("\n" + "=" * 60)
        print("Loading pre-trained encoders from checkpoint...")
        print("=" * 60)
        
        checkpoint_prefix = args.checkpoint
        if checkpoint_prefix is None:
            checkpoint_prefix = config.get("gma", {}).get("checkpoint_prefix", "kaa_grit")
        
        # Generate GMA outputs from checkpoint
        gma_outputs = generate_gma_outputs_from_checkpoint(config, checkpoint_prefix)
        
        results["gma"] = {
            "checkpoint": checkpoint_prefix,
            "s_align_shape": gma_outputs["s_align"].shape if gma_outputs["s_align"] is not None else None,
        }
        
        # Then run MFEA
        print("\n" + "=" * 60)
        print("Phase 2: MFEA Evolution")
        print("=" * 60)
        
        adj_l1, adj_l2, num_nodes = load_graph_data(config)
        
        if adj_l1 is None or adj_l2 is None:
            print("[Phase 2] Error: No graph data available")
            return results
        
        if "mfea" not in config:
            config["mfea"] = {}
        config["mfea"]["num_nodes"] = num_nodes
        
        mfea_result = run_mfea_solver(
            config=config,
            s_align=gma_outputs["s_align"],
            alpha_l1=gma_outputs["alpha_l1"],
            alpha_l2=gma_outputs["alpha_l2"],
            embeddings_l1=gma_outputs["embeddings_l1"],
            embeddings_l2=gma_outputs["embeddings_l2"],
            graph_l1=adj_l1,
            graph_l2=adj_l2,
            adj_l1=adj_l1,
            adj_l2=adj_l2,
            verbose=not args.quiet,
        )
        
        results["mfea"] = mfea_result
        save_final_results(config, mfea_result)
    else:
        # Standard pipeline
        results = run_pipeline(
            config=config,
            mode=args.mode,
            s_align_path=args.s_align_path,
            verbose=not args.quiet,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    
    if "gma" in results:
        if "final_loss" in results["gma"]:
            print(f"[GMA] Final loss: {results['gma'].get('final_loss', 'N/A')}")
        if "checkpoint" in results["gma"]:
            print(f"[GMA] Loaded checkpoint: {results['gma'].get('checkpoint')}")
        if "s_align_path" in results["gma"]:
            print(f"[GMA] S_align: {results['gma'].get('s_align_path', 'N/A')}")
    
    if "mfea" in results:
        scores = results["mfea"].get("best_scores", {})
        print(f"[MFEA] Best T1: {scores.get('T1', 'N/A'):.4f}")
        print(f"[MFEA] Best T2: {scores.get('T2', 'N/A'):.4f}")
        print(f"[MFEA] Best T3: {scores.get('T3', 'N/A'):.4f}")
    
    print("=" * 60)
    print("Done!")
    
    return results


if __name__ == "__main__":
    main()
