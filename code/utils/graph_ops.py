"""
Graph Operations Utility Module.

Provides common graph operations for MFEA population initialization
and fitness evaluation.
"""

import numpy as np
import networkx as nx
from typing import List, Set, Optional, Tuple, Union


def get_adjacency_list(graph) -> Tuple[List[List[int]], int]:
    """
    Convert graph to adjacency list format.
    
    Args:
        graph: NetworkX graph, adjacency matrix, or adjacency list.
    
    Returns:
        Tuple of (adjacency_list, num_nodes).
    """
    if isinstance(graph, list):
        return graph, len(graph)
    
    if isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        n = graph.shape[0]
        adj = [np.nonzero(graph[i])[0].tolist() for i in range(n)]
        return adj, n
    
    if hasattr(graph, "adj"):  # NetworkX graph
        nodes = list(graph.nodes())
        n = len(nodes)
        if nodes and all(isinstance(node, (int, np.integer)) for node in nodes):
            if set(nodes) == set(range(n)):
                adj = [list(graph.neighbors(i)) for i in range(n)]
                return adj, n
        # Handle non-contiguous node IDs
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        adj = [[] for _ in range(n)]
        for node in nodes:
            src = node_to_idx[node]
            adj[src] = [node_to_idx[nei] for nei in graph.neighbors(node)]
        return adj, n
    
    raise TypeError(f"Unsupported graph type: {type(graph)}")


def get_2hop_neighbors(
    graph,
    node: int,
    adj: Optional[List[List[int]]] = None,
) -> Set[int]:
    """
    Get all nodes reachable within 2 hops from a given node.
    
    Args:
        graph: Graph object (NetworkX or adjacency list).
        node: Source node index.
        adj: Precomputed adjacency list (optional, for efficiency).
    
    Returns:
        Set of node indices reachable within 2 hops (excluding source).
    """
    if adj is None:
        adj, _ = get_adjacency_list(graph)
    
    if node >= len(adj):
        return set()
    
    # 1-hop neighbors
    one_hop = set(adj[node])
    
    # 2-hop neighbors
    two_hop = set()
    for neighbor in one_hop:
        if neighbor < len(adj):
            two_hop.update(adj[neighbor])
    
    # Remove source node
    two_hop.discard(node)
    
    return two_hop


def get_2hop_coverage(
    graph,
    node: int,
    adj: Optional[List[List[int]]] = None,
) -> int:
    """
    Count number of nodes reachable within 2 hops.
    
    Args:
        graph: Graph object.
        node: Source node index.
        adj: Precomputed adjacency list.
    
    Returns:
        Count of 2-hop reachable nodes.
    """
    return len(get_2hop_neighbors(graph, node, adj))


def get_local_clustering_coefficient(
    graph,
    node: int,
    adj: Optional[List[List[int]]] = None,
) -> float:
    """
    Compute local clustering coefficient for a node.
    
    C(v) = 2 * triangles / (degree * (degree - 1))
    
    Args:
        graph: Graph object.
        node: Node index.
        adj: Precomputed adjacency list.
    
    Returns:
        Clustering coefficient in [0, 1].
    """
    if adj is None:
        adj, _ = get_adjacency_list(graph)
    
    if node >= len(adj):
        return 0.0
    
    neighbors = adj[node]
    degree = len(neighbors)
    
    if degree < 2:
        return 0.0
    
    # Count triangles
    neighbor_set = set(neighbors)
    triangles = 0
    for i, u in enumerate(neighbors):
        if u >= len(adj):
            continue
        for v in neighbors[i + 1:]:
            if v in adj[u]:
                triangles += 1
    
    return 2.0 * triangles / (degree * (degree - 1))


def get_degree_centrality(
    graph,
    adj: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """
    Compute degree centrality for all nodes.
    
    Args:
        graph: Graph object.
        adj: Precomputed adjacency list.
    
    Returns:
        Array of degree centrality values.
    """
    if adj is None:
        adj, n = get_adjacency_list(graph)
    else:
        n = len(adj)
    
    degrees = np.array([len(adj[i]) if i < len(adj) else 0 for i in range(n)])
    
    # Normalize by max possible degree (n-1)
    if n > 1:
        degrees = degrees / (n - 1)
    
    return degrees


def extract_subgraph(
    graph,
    nodes: List[int],
) -> Optional[nx.Graph]:
    """
    Extract subgraph induced by given nodes.
    
    Args:
        graph: NetworkX graph.
        nodes: List of node indices to include.
    
    Returns:
        Induced subgraph or None if graph is not NetworkX.
    """
    if hasattr(graph, "subgraph"):
        return graph.subgraph(nodes).copy()
    return None


def compute_robustness_scores(
    graph,
    alpha: Optional[np.ndarray] = None,
    weight_2hop: float = 0.5,
    weight_clustering: float = 0.3,
    weight_degree: float = 0.2,
) -> np.ndarray:
    """
    Compute robustness scores for all nodes.
    
    Score combines:
    1. 2-hop coverage (spread potential)
    2. Local clustering (connection redundancy)
    3. Degree centrality (basic connectivity)
    
    Args:
        graph: Graph object.
        alpha: Optional attention weights to incorporate.
        weight_2hop: Weight for 2-hop coverage component.
        weight_clustering: Weight for clustering component.
        weight_degree: Weight for degree component.
    
    Returns:
        Array of robustness scores (normalized to sum to 1).
    """
    adj, n = get_adjacency_list(graph)
    
    scores = np.zeros(n, dtype=np.float64)
    
    for node in range(n):
        # 2-hop coverage
        two_hop_count = get_2hop_coverage(None, node, adj)
        two_hop_score = two_hop_count / max(n, 1)
        
        # Clustering coefficient
        clustering = get_local_clustering_coefficient(None, node, adj)
        
        # Degree
        degree = len(adj[node]) if node < len(adj) else 0
        degree_score = degree / max(n - 1, 1)
        
        # Combined score
        scores[node] = (
            weight_2hop * two_hop_score +
            weight_clustering * clustering +
            weight_degree * degree_score
        )
    
    # Optionally weight by alpha
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=np.float64).flatten()[:n]
        scores = scores * alpha
    
    # Normalize
    total = scores.sum()
    if total > 0:
        scores = scores / total
    else:
        scores = np.ones(n) / n
    
    return scores


def compute_2hop_influence_potential(
    graph,
    node: int,
    p: float = 0.01,
    adj: Optional[List[List[int]]] = None,
) -> float:
    """
    Estimate influence potential using 2-hop approximation.
    
    Args:
        graph: Graph object.
        node: Node index.
        p: Propagation probability.
        adj: Precomputed adjacency list.
    
    Returns:
        Estimated influence potential.
    """
    if adj is None:
        adj, _ = get_adjacency_list(graph)
    
    if node >= len(adj):
        return 0.0
    
    influence = 1.0  # Self
    
    # 1-hop contribution
    neighbors = adj[node]
    influence += p * len(neighbors)
    
    # 2-hop contribution
    for neighbor in neighbors:
        if neighbor < len(adj):
            second_hop = len(adj[neighbor])
            influence += p * p * second_hop
    
    return influence
