import argparse
import os

import config
from generate_network import generate_network


def iter_networks(graph_num=None, graph_types=None):
    """
    生成多层网络样本迭代器，避免一次性占用内存。
    """
    use_graph_num = config.graph_num if graph_num is None else graph_num
    use_graph_types = config.graph_type if graph_types is None else graph_types

    for gt in use_graph_types:
        for node in config.nodes_num:
            for layer in config.layer_num:
                for lnc in config.layer_nodes_cover:
                    for ad in config.avg_degree:
                        for _ in range(use_graph_num):
                            yield generate_network(node, layer, lnc, ad, gt)
                        print(f"已生成网络:{node},{layer},{lnc},{ad},{gt}")


def _format_name(idx, graph_type, nodes_num, layer_num, edges_num, width):
    return f"{idx:0{width}d}_{graph_type}_{nodes_num}_{layer_num}_{edges_num}.txt"


def save_dataset(output_dir, start_id=None, graph_num=None, graph_types=None):
    os.makedirs(output_dir, exist_ok=True)
    idx = config.id_start if start_id is None else start_id
    width = config.id_width

    for data in iter_networks(graph_num=graph_num, graph_types=graph_types):
        Gm, nodes_num, layer_num, edges_num, graph_type = data
        file_name = _format_name(idx, graph_type, nodes_num, layer_num, edges_num, width)
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            for j, G in enumerate(Gm):
                f.write(
                    f"layer:{j+1} nodes={G.number_of_nodes()} edges={G.number_of_edges()}\n"
                )
                for u, v in G.edges():
                    f.write(f"{u} {v}\n")
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multilayer graph datasets.")
    parser.add_argument(
        "--output",
        default=config.save_path,
        help="输出目录（默认: config.save_path）",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=config.id_start,
        help="文件编号起始值",
    )
    parser.add_argument(
        "--graph-num",
        type=int,
        default=config.graph_num,
        help="每类网络生成数量",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=",".join(config.graph_type),
        help="网络类型列表，用逗号分隔",
    )
    args = parser.parse_args()

    graph_types = [t.strip() for t in args.types.split(",") if t.strip()]
    save_dataset(
        output_dir=args.output,
        start_id=args.start_id,
        graph_num=args.graph_num,
        graph_types=graph_types,
    )
