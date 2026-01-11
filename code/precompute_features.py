import argparse
import os
import sys
from types import SimpleNamespace

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from code.dataset.create_dataset import create_dataset
from code.utils.data_loader import read_graph_dir, read_graph_file


def _build_args(config, split, workers, with_label):
    gma_cfg = config.get("gma", {})
    dataset_cfg = config.get("dataset", {})
    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    label_root = dataset_cfg.get("label_path", os.path.join(dataset_root, "label") + "/")

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
        preprocess_workers=workers,
        label_path=label_root,
        is_cover_label=dataset_cfg.get("is_cover_label", False),
        skip_label=not with_label,
        split=split,
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute graph features for GMA-MFEA.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--single-file", default="", help="Optional multilayer edgelist file")
    parser.add_argument("--workers", type=int, default=None, help="Override preprocess workers")
    parser.add_argument("--with-label", action="store_true", help="Also compute labels")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = config.get("paths", {}).get("dataset_root", "./dataset")
    split_dir = os.path.join(dataset_root, args.split)

    if args.single_file:
        graphs = read_graph_file(args.single_file)
    else:
        graphs = read_graph_dir(split_dir)

    if not graphs:
        print("No graphs found. Check dataset path or single file.")
        return

    dataset_cfg = config.get("dataset", {})
    workers = dataset_cfg.get("preprocess_workers", 0) if args.workers is None else args.workers
    proc_args = _build_args(config, args.split, workers, args.with_label)

    create_dataset(graphs, proc_args, is_train=(args.split == "train"))
    print("Precompute done.")


if __name__ == "__main__":
    main()
