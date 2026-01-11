import argparse
import os
import numpy as np

from code.utils.data_loader import read_graph_dir


def _is_finite(arr):
    return np.isfinite(arr).all()


def _check_graphs(graphs, max_graphs=0):
    total_layers = 0
    zero_edge_layers = 0
    issues = []
    for idx, entry in enumerate(graphs):
        if max_graphs and idx >= max_graphs:
            break
        for lid, g in enumerate(entry["networks"], start=1):
            total_layers += 1
            if g.number_of_edges() == 0:
                zero_edge_layers += 1
                issues.append(f"{entry['name']} layer{lid}: edges=0")
    return total_layers, zero_edge_layers, issues


def _check_features(
    graphs,
    feature_root,
    split,
    centrality_dim,
    lap_pe_dim,
    graph_embedding,
    p,
    q,
    check_node2vec,
    max_graphs=0,
):
    base = os.path.join(feature_root, split)
    centrality_missing = 0
    centrality_nan = 0
    centrality_max0 = 0
    lap_pe_bad = 0
    node2vec_bad = 0
    issues = []

    for idx, entry in enumerate(graphs):
        if max_graphs and idx >= max_graphs:
            break
        name = entry["name"]
        layer_num = len(entry["networks"])
        graph_dir = os.path.join(base, name)
        for lid in range(1, layer_num + 1):
            c_path = os.path.join(graph_dir, f"centrality{centrality_dim}", f"{lid}_c{centrality_dim}.npy")
            if not os.path.isfile(c_path):
                centrality_missing += 1
                issues.append(f"{name} layer{lid}: centrality missing")
            else:
                c = np.load(c_path)
                if not _is_finite(c):
                    centrality_nan += 1
                    issues.append(f"{name} layer{lid}: centrality NaN/Inf")
                if c.size > 0 and np.max(c) <= 0:
                    centrality_max0 += 1
                    issues.append(f"{name} layer{lid}: centrality max<=0")

            if lap_pe_dim > 0:
                l_path = os.path.join(graph_dir, "lap_pe", f"{lid}_lpe_{lap_pe_dim}.npy")
                if not os.path.isfile(l_path):
                    lap_pe_bad += 1
                    issues.append(f"{name} layer{lid}: lap_pe missing")
                else:
                    lp = np.load(l_path)
                    if not _is_finite(lp) or lp.ndim != 2 or lp.shape[1] != lap_pe_dim:
                        lap_pe_bad += 1
                        issues.append(f"{name} layer{lid}: lap_pe bad shape/NaN")

            if check_node2vec:
                nv_path = os.path.join(
                    graph_dir,
                    "embedding",
                    str(graph_embedding),
                    f"{lid}_{p}_{q}_nv.npy",
                )
                if not os.path.isfile(nv_path):
                    node2vec_bad += 1
                    issues.append(f"{name} layer{lid}: node2vec missing")
                else:
                    nv = np.load(nv_path)
                    if not _is_finite(nv) or nv.ndim != 2 or nv.shape[1] != graph_embedding:
                        node2vec_bad += 1
                        issues.append(f"{name} layer{lid}: node2vec bad shape/NaN")

    summary = {
        "centrality_missing": centrality_missing,
        "centrality_nan": centrality_nan,
        "centrality_max0": centrality_max0,
        "lap_pe_bad": lap_pe_bad,
        "node2vec_bad": node2vec_bad,
    }
    return summary, issues


def main():
    parser = argparse.ArgumentParser(description="Check dataset quality for NaN/zero issues.")
    parser.add_argument("--split", default="train", choices=["train", "test", "eval"])
    parser.add_argument("--dataset-root", default="./dataset")
    parser.add_argument("--feature-root", default="./dataset/feature")
    parser.add_argument("--centrality-dim", type=int, default=12)
    parser.add_argument("--lap-pe-dim", type=int, default=36)
    parser.add_argument("--graph-embedding", type=int, default=256)
    parser.add_argument("--p", type=float, default=1.5)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--check-features", action="store_true")
    parser.add_argument("--check-node2vec", action="store_true")
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-issues", type=int, default=20)
    args = parser.parse_args()

    split_dir = os.path.join(args.dataset_root, args.split)
    graphs = read_graph_dir(split_dir)
    if not graphs:
        print("No graphs found.")
        return

    total_layers, zero_edge_layers, graph_issues = _check_graphs(graphs, max_graphs=args.max_graphs)
    print(f"graphs: {len(graphs)}, layers: {total_layers}, zero_edge_layers: {zero_edge_layers}")

    if args.check_features:
        summary, feat_issues = _check_features(
            graphs,
            args.feature_root,
            args.split,
            args.centrality_dim,
            args.lap_pe_dim,
            args.graph_embedding,
            args.p,
            args.q,
            args.check_node2vec,
            max_graphs=args.max_graphs,
        )
        print("feature_summary:", summary)
        issues = graph_issues + feat_issues
    else:
        issues = graph_issues

    if issues:
        print("sample_issues:")
        for line in issues[: args.max_issues]:
            print(" -", line)
        if len(issues) > args.max_issues:
            print(f"... and {len(issues) - args.max_issues} more")
    else:
        print("no_issues_found")


if __name__ == "__main__":
    main()
