import os
import sys
import random
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from code.utils.data_loader import read_graph_dir
from code.mfea_core.tasks import MFEATasks


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_first_test_graph(dataset_root):
    test_dir = os.path.join(dataset_root, "test")
    graphs = read_graph_dir(test_dir)
    if not graphs:
        raise RuntimeError(f"test 目录为空: {test_dir}")
    entry = graphs[0]
    networks = entry["networks"]
    if len(networks) < 2:
        raise RuntimeError("测试图层数不足 2 层，无法计算 T1/T2/T3。")
    return entry["name"], networks[0], networks[1]


def _sample_seeds(nodes, budget, rng):
    if budget <= 0:
        return []
    if budget > len(nodes):
        budget = len(nodes)
    return rng.sample(nodes, budget)


def main():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    config = _load_config(config_path)
    dataset_root = config["paths"]["dataset_root"]
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(ROOT_DIR, dataset_root)
    seed = config["project"].get("seed", 42)

    task_budget_a = config["tasks"]["T1"]["budget"]
    task_budget_b = config["tasks"]["T2"]["budget"]
    attack_ratio = config["tasks"]["T3"].get("attack_ratio")
    attack_mode = config["tasks"].get("attack_mode", "2hop")

    random.seed(seed)
    rng = random.Random(seed)

    name, graph_a, graph_b = _pick_first_test_graph(dataset_root)
    nodes = list(graph_a.nodes())
    if not nodes:
        raise RuntimeError("测试图为空，无法采样种子。")

    seeds_a = _sample_seeds(nodes, task_budget_a, rng)
    remaining = [n for n in nodes if n not in set(seeds_a)]
    seeds_b = _sample_seeds(remaining, task_budget_b, rng)

    if len(seeds_a) + len(seeds_b) == 0:
        raise RuntimeError("种子为空，请检查预算或图大小。")

    print("=== 测试图信息 ===")
    print(f"name: {name}")
    print(f"nodes: {len(nodes)}")
    print(f"budget A: {task_budget_a}, budget B: {task_budget_b}")
    print("=== 随机种子 ===")
    print(f"S_A ({len(seeds_a)}): {seeds_a}")
    print(f"S_B ({len(seeds_b)}): {seeds_b}")

    tasks = MFEATasks(
        graph_a,
        graph_b,
        attack_ratio=attack_ratio,
        attack_mode=attack_mode,
    )

    r_cs_1 = tasks.evaluate_t1(seeds_a, opponent_seeds=seeds_b, use_attention=False)
    r_cs_2 = tasks.evaluate_t2(seeds_b, opponent_seeds=seeds_a, use_attention=False)
    r_cr = tasks.evaluate_t3([seeds_a, seeds_b], use_attention=False)

    print("=== 指标结果 ===")
    print(f"R_CS^(1): {r_cs_1:.6f}")
    print(f"R_CS^(2): {r_cs_2:.6f}")
    print(f"R_CR: {r_cr:.6f}")


if __name__ == "__main__":
    main()
