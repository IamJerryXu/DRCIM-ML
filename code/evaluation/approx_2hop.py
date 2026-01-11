import numpy as np


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
    raise TypeError("Unsupported graph type for 2-hop approximation.")


def _normalize_seeds(seeds, num_nodes):
    if seeds is None:
        return []
    clean = []
    for s in seeds:
        if isinstance(s, (int, np.integer)) and 0 <= int(s) < num_nodes:
            clean.append(int(s))
    return clean


def _neighbors(adj, node, active):
    if active is not None and not active[node]:
        return []
    if active is None:
        return adj[node]
    return [nbr for nbr in adj[node] if active[nbr]]


def _degree(adj, active):
    n = len(adj)
    deg = np.zeros(n, dtype=int)
    if active is None:
        for i in range(n):
            deg[i] = len(adj[i])
        return deg
    for i in range(n):
        if not active[i]:
            continue
        count = 0
        for nbr in adj[i]:
            if active[nbr]:
                count += 1
        deg[i] = count
    return deg


def _find_simi(cs, seeds):
    cs_set = set(cs)
    return [seed for seed in seeds if seed in cs_set]


def _find_third(cs, seeds, seed):
    seed_set = set(seeds)
    cs_dis_simi = [node for node in cs if node not in seed_set]
    cs_simi = _find_simi(cs, seeds)
    cs_d = [node for node in cs_simi if node != seed]
    return cs_dis_simi, cs_d


def _find_simi_cim(cs, seeds):
    cs_set = set(cs)
    return [seed for seed in seeds if seed in cs_set]


def _find_third_cim(cs, seeds, seed):
    seed_set = set(seeds)
    cs_dis_simi = [node for node in cs if node not in seed_set]
    cs_simi = _find_simi_cim(cs, seeds)
    cs_d = [node for node in cs_simi if node != seed]
    return cs_dis_simi, cs_d


def _sum_alpha_neighbors(adj, node, active, alpha, exclude=None):
    total = 0.0
    if active is not None and not active[node]:
        return total
    for nbr in adj[node]:
        if active is not None and not active[nbr]:
            continue
        if exclude is not None and nbr in exclude:
            continue
        total += alpha[nbr]
    return total


def calculate_2hop_influence(seeds, graph, p=0.01, active_mask=None, alpha=None):
    """
    Strict Multi_RCIM 2-hop influence (cal_Influ in C++).
    """
    adj, num_nodes = _as_adj_list(graph)
    seeds = _normalize_seeds(seeds, num_nodes)
    alpha_vec = None
    if alpha is not None:
        if hasattr(alpha, "detach"):
            alpha = alpha.detach().cpu().numpy()
        alpha_vec = np.asarray(alpha, dtype=np.float32).reshape(-1)
        if alpha_vec.shape[0] < num_nodes:
            raise ValueError("alpha length must be >= number of nodes.")
        alpha_vec = alpha_vec[:num_nodes]

    deg = _degree(adj, active_mask)
    influ = 0.0
    for seed in seeds:
        seed_weight = alpha_vec[seed] if alpha_vec is not None else 1.0
        influ += seed_weight
        sum_temp = 0.0
        if active_mask is None or active_mask[seed]:
            for j in adj[seed]:
                if active_mask is not None and not active_mask[j]:
                    continue
                sum_temp += alpha_vec[j] if alpha_vec is not None else 1.0
                if alpha_vec is not None:
                    deg_alpha = _sum_alpha_neighbors(adj, j, active_mask, alpha_vec)
                    sum_temp += deg_alpha * p
                else:
                    sum_temp += deg[j] * p
        influ += p * sum_temp

    sec_sum_temp = 0.0
    thr_sum_temp = 0.0
    for seed in seeds:
        cs = _neighbors(adj, seed, active_mask)
        cs_simi = _find_simi(cs, seeds)
        sum_temp = 0.0
        for node in cs_simi:
            base = 1.0 + p * deg[node]
            weight = alpha_vec[node] if alpha_vec is not None else 1.0
            sum_temp += (base * weight) * p
        sec_sum_temp += sum_temp
        cs_dis_simi, cs_d = _find_third(cs, seeds, seed)
        if cs_dis_simi and cs_d:
            cs_dis_set = set(cs_dis_simi)
            for node in cs_d:
                if node in cs_dis_set:
                    weight = alpha_vec[node] if alpha_vec is not None else 1.0
                    thr_sum_temp += p * p * weight

    return influ - sec_sum_temp - thr_sum_temp


def calculate_competitive_influence(
    seeds_a,
    seeds_b,
    graph,
    p=0.01,
    active_mask=None,
    alpha_a=None,
    alpha_b=None,
):
    """
    Strict Multi_RCIM competitive influence (cal_ComInflu in C++).
    Returns (sum, influ_a, influ_b).
    """
    adj, num_nodes = _as_adj_list(graph)
    seeds_a = _normalize_seeds(seeds_a, num_nodes)
    seeds_b = _normalize_seeds(seeds_b, num_nodes)

    alpha_vec_a = None
    alpha_vec_b = None
    if alpha_a is not None:
        if hasattr(alpha_a, "detach"):
            alpha_a = alpha_a.detach().cpu().numpy()
        alpha_vec_a = np.asarray(alpha_a, dtype=np.float32).reshape(-1)
        if alpha_vec_a.shape[0] < num_nodes:
            raise ValueError("alpha_a length must be >= number of nodes.")
        alpha_vec_a = alpha_vec_a[:num_nodes]
        # Rescale so mean=1 (preserve relative weights but fix magnitude)
        if alpha_vec_a.mean() > 0:
            alpha_vec_a = alpha_vec_a / alpha_vec_a.mean()
    if alpha_b is not None:
        if hasattr(alpha_b, "detach"):
            alpha_b = alpha_b.detach().cpu().numpy()
        alpha_vec_b = np.asarray(alpha_b, dtype=np.float32).reshape(-1)
        if alpha_vec_b.shape[0] < num_nodes:
            raise ValueError("alpha_b length must be >= number of nodes.")
        alpha_vec_b = alpha_vec_b[:num_nodes]
        # Rescale so mean=1
        if alpha_vec_b.mean() > 0:
            alpha_vec_b = alpha_vec_b / alpha_vec_b.mean()

    deg = _degree(adj, active_mask)
    seeds_a_set = set(seeds_a)
    seeds_b_set = set(seeds_b)

    def _calc_side(seeds_self, seeds_opp, alpha_vec):
        opp_set = set(seeds_opp)
        influ = 0.0
        for seed in seeds_self:
            weight_seed = alpha_vec[seed] if alpha_vec is not None else 1.0
            influ += weight_seed
            sum_temp = 0.0
            if active_mask is None or active_mask[seed]:
                for j in adj[seed]:
                    if active_mask is not None and not active_mask[j]:
                        continue
                    if j in opp_set:
                        continue
                    sum_temp += alpha_vec[j] if alpha_vec is not None else 1.0
                    if alpha_vec is not None:
                        deg_alpha = _sum_alpha_neighbors(adj, j, active_mask, alpha_vec, exclude=opp_set)
                        sum_temp += deg_alpha * p
                    else:
                        deg_temp = 0
                        for k in adj[j]:
                            if active_mask is not None and not active_mask[k]:
                                continue
                            if k in opp_set:
                                continue
                            deg_temp += 1
                        sum_temp += deg_temp * p
            influ += p * sum_temp

        sec_sum_temp = 0.0
        thr_sum_temp = 0.0
        for seed in seeds_self:
            if active_mask is not None and not active_mask[seed]:
                cs = []
            else:
                cs = [
                    j for j in adj[seed]
                    if (active_mask is None or active_mask[j]) and j not in opp_set
                ]
            cs_simi = _find_simi_cim(cs, seeds_self)
            sum_temp = 0.0
            for node in cs_simi:
                deg_temp = deg[node]
                for k in _neighbors(adj, node, active_mask):
                    if k in opp_set:
                        deg_temp //= 2
                base = 1.0 + p * deg_temp
                weight = alpha_vec[node] if alpha_vec is not None else 1.0
                sum_temp += (base * weight) * p
            sec_sum_temp += sum_temp
            cs_dis_simi, cs_d = _find_third_cim(cs, seeds_self, seed)
            if cs_dis_simi and cs_d:
                cs_dis_set = set(cs_dis_simi)
                for node in cs_d:
                    if node in cs_dis_set:
                        weight = alpha_vec[node] if alpha_vec is not None else 1.0
                        thr_sum_temp += p * p * weight

        return influ - sec_sum_temp - thr_sum_temp

    influ_a = _calc_side(seeds_a, seeds_b, alpha_vec_a)
    influ_b = _calc_side(seeds_b, seeds_a, alpha_vec_b)
    return influ_a + influ_b, influ_a, influ_b


def calculate_node_influence_scores(
    seeds_a,
    seeds_b,
    graph,
    attack_mode="2hop",
    active_mask=None,
    rng=None,
):
    adj, num_nodes = _as_adj_list(graph)
    seeds_a = _normalize_seeds(seeds_a, num_nodes)
    seeds_b = _normalize_seeds(seeds_b, num_nodes)
    seeds_a_set = set(seeds_a)
    seeds_b_set = set(seeds_b)

    if rng is None:
        rng = np.random.default_rng()

    scores = np.zeros(num_nodes, dtype=np.float32)
    if attack_mode == "random":
        scores = rng.integers(0, num_nodes, size=num_nodes, dtype=np.int32).astype(np.float32)
        return scores

    deg = _degree(adj, active_mask)
    if attack_mode == "degree":
        return deg.astype(np.float32)

    if attack_mode != "2hop":
        raise ValueError(f"Unsupported attack mode: {attack_mode}")

    for i in range(num_nodes):
        if active_mask is not None and not active_mask[i]:
            continue
        score = 0
        if i in seeds_a_set:
            for j in _neighbors(adj, i, active_mask):
                if j in seeds_b_set:
                    continue
                score += 1 + deg[j]
        elif i in seeds_b_set:
            for j in _neighbors(adj, i, active_mask):
                if j in seeds_a_set:
                    continue
                score += 1 + deg[j]
        else:
            for j in _neighbors(adj, i, active_mask):
                if j in seeds_a_set or j in seeds_b_set:
                    continue
                score += 1 + deg[j]
        scores[i] = score
    return scores


def calculate_robust_competitive_influence(
    seeds_a,
    seeds_b,
    graph,
    p=0.01,
    attack_steps=None,
    attack_ratio=None,
    attack_mode="2hop",
    alpha_a=None,
    alpha_b=None,
    rng=None,
):
    """
    Strict Multi_RCIM robust evaluation (cal_Influ_CIM_Net_Node in C++).
    """
    adj, num_nodes = _as_adj_list(graph)
    steps = num_nodes
    if attack_steps is not None:
        steps = max(0, int(attack_steps))
    elif attack_ratio is not None:
        steps = max(0, int(round(num_nodes * attack_ratio)))

    if steps == 0:
        return {"total": 0.0, "a": 0.0, "b": 0.0}

    active = np.ones(num_nodes, dtype=bool)
    total_sum = 0.0
    total_a = 0.0
    total_b = 0.0

    if rng is None:
        rng = np.random.default_rng()

    for _ in range(steps):
        scores = calculate_node_influence_scores(
            seeds_a,
            seeds_b,
            adj,
            attack_mode=attack_mode,
            active_mask=active,
            rng=rng,
        )
        atk_index = int(np.argmax(scores))
        active[atk_index] = False

        influ_sum, influ_a, influ_b = calculate_competitive_influence(
            seeds_a,
            seeds_b,
            adj,
            p=p,
            active_mask=active,
            alpha_a=alpha_a,
            alpha_b=alpha_b,
        )
        total_sum += influ_sum
        total_a += influ_a
        total_b += influ_b

    return {
        "total": total_sum / steps,
        "a": total_a / steps,
        "b": total_b / steps,
    }


def calculate_2hop_probabilities(
    seeds,
    graph,
    p=0.01,
    blocked=None,
    active_mask=None,
    return_probs=False,
):
    """
    Probability-style 2-hop approximation for doc-based P(v).
    """
    adj, num_nodes = _as_adj_list(graph)
    active = np.ones(num_nodes, dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape[0] != num_nodes:
        raise ValueError("active_mask size must match number of nodes.")

    seeds = _normalize_seeds(seeds, num_nodes)
    blocked_set = set(_normalize_seeds(blocked, num_nodes))

    probs = np.zeros(num_nodes, dtype=np.float32)
    for s in seeds:
        if active[s] and s not in blocked_set:
            probs[s] = 1.0

    if len(seeds) == 0:
        return (0.0, probs) if return_probs else 0.0

    p2 = p * p
    for s in seeds:
        if not active[s] or s in blocked_set:
            continue
        for u in adj[s]:
            if not active[u] or u in blocked_set:
                continue
            if probs[u] < 1.0:
                probs[u] += p
            for v in adj[u]:
                if not active[v] or v in blocked_set or v == s:
                    continue
                if probs[v] < 1.0:
                    probs[v] += p2

    np.clip(probs, 0.0, 1.0, out=probs)
    influence = float(probs.sum())
    return (influence, probs) if return_probs else influence
