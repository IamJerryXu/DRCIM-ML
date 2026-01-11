import numpy as np
from .approx_2hop import (
    calculate_2hop_influence,
    calculate_2hop_probabilities,
    calculate_robust_competitive_influence,
)

class MetricsCalculator:
    """
    Calculator for R_CS, R_CR, R_CAC.
    
    TODO: Implement metric calculations using 2-hop approximation or IC simulations.
    """

    @staticmethod
    def _as_adj_list(graph):
        if isinstance(graph, list):
            return graph, len(graph)
        if isinstance(graph, tuple):
            adj = [list(nei) for nei in graph]
            return adj, len(adj)
        if isinstance(graph, np.ndarray):
            if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
                raise ValueError("Adjacency matrix must be square.")
            n = graph.shape[0]
            adj = [np.nonzero(graph[i])[0].tolist() for i in range(n)]
            return adj, n
        if hasattr(graph, "adj"):
            nodes = list(graph.nodes())
            n = len(nodes)
            if nodes and all(isinstance(node, (int, np.integer)) for node in nodes):
                if set(nodes) == set(range(n)):
                    adj = [list(graph.neighbors(i)) for i in range(n)]
                    return adj, n
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            adj = [[] for _ in range(n)]
            for node in nodes:
                src = node_to_idx[node]
                adj[src] = [node_to_idx[nei] for nei in graph.neighbors(node)]
            return adj, n
        raise TypeError("Unsupported graph type for metric evaluation.")

    @staticmethod
    def _normalize_seeds(seeds, num_nodes):
        if seeds is None:
            return []
        clean = []
        for s in seeds:
            if isinstance(s, (int, np.integer)) and 0 <= int(s) < num_nodes:
                clean.append(int(s))
        return clean

    @staticmethod
    def _alpha_to_numpy(alpha, num_nodes):
        if alpha is None:
            return None
        if hasattr(alpha, "detach"):
            alpha = alpha.detach().cpu().numpy()
        alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
        if alpha.shape[0] < num_nodes:
            # Alpha dimension mismatch - return None to use uniform weights
            print(f"[Warning] alpha size ({alpha.shape[0]}) < num_nodes ({num_nodes}), using uniform weights")
            return None
        return alpha[:num_nodes]

    @staticmethod
    def _resolve_attack_steps(num_nodes, attack_steps=None, attack_ratio=None):
        if attack_steps is None:
            if attack_ratio is not None:
                attack_steps = int(max(1, round(num_nodes * attack_ratio)))
            else:
                attack_steps = num_nodes
        return max(0, int(attack_steps))

    @staticmethod
    def _node_2hop_score(node, adj, active, blocked):
        score = 0
        for u in adj[node]:
            if not active[u] or u in blocked:
                continue
            score += 1
            for v in adj[u]:
                if not active[v] or v in blocked:
                    continue
                score += 1
        return score

    @staticmethod
    def _select_attack_node(adj, active, seeds_self, seeds_opp, mode, rng):
        candidates = np.flatnonzero(active)
        if candidates.size == 0:
            return None
        if mode == "random":
            return int(rng.choice(candidates))
        if mode == "degree":
            best = None
            best_score = -1
            for node in candidates:
                deg = 0
                for u in adj[node]:
                    if active[u]:
                        deg += 1
                if deg > best_score:
                    best_score = deg
                    best = int(node)
            return best
        if mode == "2hop":
            best = None
            best_score = -1
            seed_set = set(seeds_self)
            opp_set = set(seeds_opp)
            both_set = seed_set | opp_set
            for node in candidates:
                if node in seed_set:
                    blocked = opp_set
                elif node in opp_set:
                    blocked = seed_set
                else:
                    blocked = both_set
                score = MetricsCalculator._node_2hop_score(node, adj, active, blocked)
                if score > best_score:
                    best_score = score
                    best = int(node)
            return best
        raise ValueError(f"Unsupported attack mode: {mode}")
    
    @staticmethod
    def calculate_r_cs(
        seeds,
        opponent_seeds,
        graph,
        alpha=None,
        p=0.01,
        attack_steps=None,
        attack_ratio=None,
        attack_mode="2hop",
        attn_on_overlap=False,
        rng=None,
    ):
        """
        Calculate Single Layer Competitive Robustness (R_CS).
        
        - Influence uses 2-hop approximation.
        - Overlap penalty uses expected overlap of activation probabilities.
        """
        adj, num_nodes = MetricsCalculator._as_adj_list(graph)
        seeds = MetricsCalculator._normalize_seeds(seeds, num_nodes)
        opponent_seeds = MetricsCalculator._normalize_seeds(opponent_seeds, num_nodes)
        alpha_vec = MetricsCalculator._alpha_to_numpy(alpha, num_nodes)

        result = calculate_robust_competitive_influence(
            seeds,
            opponent_seeds,
            adj,
            p=p,
            attack_steps=attack_steps,
            attack_ratio=attack_ratio,
            attack_mode=attack_mode,
            alpha_a=alpha_vec,
            alpha_b=None,
            rng=rng,
        )
        return result["a"]

    @staticmethod
    def calculate_r_cs_attn(
        seeds,
        opponent_seeds,
        graph,
        alpha,
        p=0.01,
        attack_steps=None,
        attack_ratio=None,
        attack_mode="2hop",
        attn_on_overlap=True,
        rng=None,
    ):
        return MetricsCalculator.calculate_r_cs(
            seeds,
            opponent_seeds,
            graph,
            alpha=alpha,
            p=p,
            attack_steps=attack_steps,
            attack_ratio=attack_ratio,
            attack_mode=attack_mode,
            attn_on_overlap=attn_on_overlap,
            rng=rng,
        )

    @staticmethod
    def calculate_r_cr(
        seeds_l1,
        seeds_l2,
        graph_l1,
        graph_l2,
        alpha_l1=None,
        alpha_l2=None,
        p=0.01,
        attack_steps=None,
        attack_ratio=None,
        attack_mode="2hop",
        use_attention=False,
        eps=1e-12,
    ):
        """
        Calculate Collaborative Robustness (R_CR).
        
        - Synergy Ratio: sigma_joint / (sigma_l1 + sigma_l2)
        - Survival Baseline: HMean(R_CS_1, R_CS_2)
        """
        r_cs_1 = MetricsCalculator.calculate_r_cs(
            seeds_l1,
            seeds_l2,
            graph_l1,
            alpha=alpha_l1 if use_attention else None,
            p=p,
            attack_steps=attack_steps,
            attack_ratio=attack_ratio,
            attack_mode=attack_mode,
        )
        r_cs_2 = MetricsCalculator.calculate_r_cs(
            seeds_l2,
            seeds_l1,
            graph_l2,
            alpha=alpha_l2 if use_attention else None,
            p=p,
            attack_steps=attack_steps,
            attack_ratio=attack_ratio,
            attack_mode=attack_mode,
        )

        hmean = (2.0 * r_cs_1 * r_cs_2) / (r_cs_1 + r_cs_2 + eps)

        sigma_l1 = calculate_2hop_influence(seeds_l1, graph_l1, p=p)
        sigma_l2 = calculate_2hop_influence(seeds_l2, graph_l2, p=p)
        _, prob_l1 = calculate_2hop_probabilities(seeds_l1, graph_l1, p=p, return_probs=True)
        _, prob_l2 = calculate_2hop_probabilities(seeds_l2, graph_l2, p=p, return_probs=True)
        n = min(len(prob_l1), len(prob_l2))
        prob_joint = 1.0 - (1.0 - prob_l1[:n]) * (1.0 - prob_l2[:n])
        sigma_joint = float(prob_joint.sum())

        denom = sigma_l1 + sigma_l2 + eps
        synergy = sigma_joint / denom
        return synergy * hmean

    @staticmethod
    def calculate_r_cr_attn(
        seeds_l1,
        seeds_l2,
        graph_l1,
        graph_l2,
        alpha_l1,
        alpha_l2,
        p=0.01,
        attack_steps=None,
        attack_ratio=None,
        attack_mode="2hop",
        eps=1e-12,
    ):
        return MetricsCalculator.calculate_r_cr(
            seeds_l1,
            seeds_l2,
            graph_l1,
            graph_l2,
            alpha_l1=alpha_l1,
            alpha_l2=alpha_l2,
            p=p,
            attack_steps=attack_steps,
            attack_ratio=attack_ratio,
            attack_mode=attack_mode,
            use_attention=True,
            eps=eps,
        )
