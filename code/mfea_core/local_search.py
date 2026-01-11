"""
Local Search Module for MFEA.

Implements advanced local search strategies to refine seed sets:
1. Competition-Aware Search: Marginal gain with opponent penalty
2. Gap-Filling Search: Cross-layer optimization using S_align
3. 1-Swap/2-Swap: Classic seed replacement optimization

Reference: TECHNICAL_DOC.md §3.5
"""

import numpy as np
from typing import List, Set, Optional, Tuple, Callable, Dict
from dataclasses import dataclass

from .population import Individual, TaskID
from ..utils.graph_ops import (
    get_adjacency_list,
    get_2hop_neighbors,
    compute_2hop_influence_potential,
)


@dataclass
class LocalSearchResult:
    """Result of a local search operation."""
    improved: bool
    old_fitness: float
    new_fitness: float
    iterations: int
    swaps_made: int


class LocalSearch:
    """
    Advanced Local Search Strategies for MFEA.
    
    Provides competition-aware and cross-layer optimization
    techniques to refine seed sets after genetic operations.
    """
    
    def __init__(
        self,
        graph_l1,
        graph_l2,
        s_align: Optional[np.ndarray] = None,
        alpha_l1: Optional[np.ndarray] = None,
        alpha_l2: Optional[np.ndarray] = None,
        p: float = 0.01,
        max_iterations: int = 50,
        improvement_threshold: float = 0.001,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize LocalSearch.
        
        Args:
            graph_l1: Layer 1 graph (adjacency list or NetworkX).
            graph_l2: Layer 2 graph.
            s_align: Cross-layer similarity matrix S_align[i][j].
            alpha_l1: Node importance for Layer 1.
            alpha_l2: Node importance for Layer 2.
            p: Propagation probability for influence estimation.
            max_iterations: Maximum iterations for local search.
            improvement_threshold: Minimum improvement to continue.
            rng: Random number generator.
        """
        self.adj_l1, self.num_nodes = get_adjacency_list(graph_l1)
        self.adj_l2, _ = get_adjacency_list(graph_l2)
        
        self.s_align = s_align
        self.alpha_l1 = self._normalize_weights(alpha_l1)
        self.alpha_l2 = self._normalize_weights(alpha_l2)
        self.p = p
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Precompute influence potentials
        self._precompute_influence()
    
    def _normalize_weights(self, alpha: Optional[np.ndarray]) -> np.ndarray:
        """Normalize weights to [0, 1] range."""
        if alpha is None:
            return np.ones(self.num_nodes) / self.num_nodes
        alpha = np.asarray(alpha, dtype=np.float64).flatten()
        if len(alpha) < self.num_nodes:
            alpha = np.pad(alpha, (0, self.num_nodes - len(alpha)),
                          constant_values=1.0 / self.num_nodes)
        alpha = alpha[:self.num_nodes]
        alpha = np.maximum(alpha, 1e-10)
        return alpha / alpha.sum()
    
    def _precompute_influence(self):
        """Precompute 2-hop influence potential for all nodes."""
        self.influence_l1 = np.array([
            compute_2hop_influence_potential(None, i, self.p, self.adj_l1)
            for i in range(self.num_nodes)
        ])
        self.influence_l2 = np.array([
            compute_2hop_influence_potential(None, i, self.p, self.adj_l2)
            for i in range(self.num_nodes)
        ])
    
    # =========================================================================
    # Competition-Aware Search (for T1/T2 tasks)
    # =========================================================================
    
    def competition_aware_marginal_gain(
        self,
        candidate: int,
        current_seeds: Set[int],
        opponent_seeds: Set[int],
        adj: List[List[int]],
        influence: np.ndarray,
        alpha: np.ndarray,
        overlap_penalty: float = 0.5,
    ) -> float:
        """
        Calculate competition-aware marginal gain for adding a candidate.
        
        Marginal gain = influence_potential - overlap_penalty * overlap_with_opponent
        
        Args:
            candidate: Candidate node to add.
            current_seeds: Current seed set.
            opponent_seeds: Opponent's seed set.
            adj: Adjacency list.
            influence: Precomputed influence potentials.
            alpha: Node importance weights.
            overlap_penalty: Penalty factor for overlap with opponent.
        
        Returns:
            Competition-aware marginal gain.
        """
        if candidate in current_seeds:
            return -np.inf
        
        # Base influence potential weighted by alpha
        base_gain = influence[candidate] * alpha[candidate]
        
        # Check overlap: candidate's influence zone vs opponent's
        candidate_zone = get_2hop_neighbors(None, candidate, adj)
        candidate_zone.add(candidate)
        
        opponent_zone = set()
        for opp in opponent_seeds:
            opp_neighbors = get_2hop_neighbors(None, opp, adj)
            opponent_zone.update(opp_neighbors)
            opponent_zone.add(opp)
        
        # Calculate overlap
        overlap = len(candidate_zone & opponent_zone)
        overlap_ratio = overlap / max(len(candidate_zone), 1)
        
        # Also check existing seed redundancy
        current_zone = set()
        for s in current_seeds:
            current_zone.update(get_2hop_neighbors(None, s, adj))
            current_zone.add(s)
        
        redundancy = len(candidate_zone & current_zone)
        redundancy_ratio = redundancy / max(len(candidate_zone), 1)
        
        # Competition-aware marginal gain
        gain = base_gain * (1.0 - overlap_penalty * overlap_ratio - 0.3 * redundancy_ratio)
        
        return gain
    
    def competition_aware_search(
        self,
        individual: Individual,
        opponent_seeds_l1: List[int],
        opponent_seeds_l2: List[int],
        task_id: TaskID = TaskID.T1,
        overlap_penalty: float = 0.5,
    ) -> LocalSearchResult:
        """
        Competition-Aware Local Search.
        
        Performs 1-swap optimization considering opponent overlap.
        For T1: optimizes seeds_l1, for T2: optimizes seeds_l2.
        
        Args:
            individual: Individual to optimize.
            opponent_seeds_l1: Opponent seeds for Layer 1.
            opponent_seeds_l2: Opponent seeds for Layer 2.
            task_id: Which task to optimize (T1 or T2).
            overlap_penalty: Penalty factor for overlap.
        
        Returns:
            LocalSearchResult with improvement details.
        """
        if task_id == TaskID.T1:
            seeds = individual.seeds_l1.copy()
            opponent = set(opponent_seeds_l1)
            adj = self.adj_l1
            influence = self.influence_l1
            alpha = self.alpha_l1
        else:  # T2
            seeds = individual.seeds_l2.copy()
            opponent = set(opponent_seeds_l2)
            adj = self.adj_l2
            influence = self.influence_l2
            alpha = self.alpha_l2
        
        current_seeds = set(seeds)
        budget = len(seeds)
        
        # Calculate initial fitness (sum of marginal gains)
        def calculate_fitness(seed_set: Set[int]) -> float:
            total = 0.0
            for s in seed_set:
                zone = get_2hop_neighbors(None, s, adj)
                zone.add(s)
                opp_overlap = len(zone & opponent)
                total += influence[s] * alpha[s] * (1.0 - overlap_penalty * opp_overlap / max(len(zone), 1))
            return total
        
        initial_fitness = calculate_fitness(current_seeds)
        best_fitness = initial_fitness
        improved = False
        iterations = 0
        swaps_made = 0
        
        # Candidate pool: nodes not in current seeds
        for iteration in range(self.max_iterations):
            iterations += 1
            best_swap = None
            best_swap_gain = 0.0
            
            # Try replacing each seed
            for remove_seed in list(current_seeds):
                temp_seeds = current_seeds - {remove_seed}
                
                # Find best replacement
                candidates = [n for n in range(self.num_nodes) 
                             if n not in current_seeds]
                
                for add_candidate in candidates:
                    gain = self.competition_aware_marginal_gain(
                        add_candidate, temp_seeds, opponent,
                        adj, influence, alpha, overlap_penalty
                    )
                    
                    # Compare with removed seed's gain
                    removed_gain = self.competition_aware_marginal_gain(
                        remove_seed, temp_seeds, opponent,
                        adj, influence, alpha, overlap_penalty
                    )
                    
                    swap_gain = gain - removed_gain
                    
                    if swap_gain > best_swap_gain + self.improvement_threshold:
                        best_swap_gain = swap_gain
                        best_swap = (remove_seed, add_candidate)
            
            if best_swap is not None:
                remove_seed, add_candidate = best_swap
                current_seeds.remove(remove_seed)
                current_seeds.add(add_candidate)
                best_fitness += best_swap_gain
                swaps_made += 1
                improved = True
            else:
                break  # No improvement found
        
        # Update individual
        new_seeds = list(current_seeds)
        if task_id == TaskID.T1:
            individual.seeds_l1 = new_seeds
        else:
            individual.seeds_l2 = new_seeds
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    # =========================================================================
    # Gap-Filling Search (for T3 task)
    # =========================================================================
    
    def _find_aligned_nodes(
        self,
        node: int,
        source_layer: int,
        top_k: int = 5,
    ) -> List[int]:
        """
        Find top-k aligned nodes in the other layer using S_align.
        
        Args:
            node: Source node index.
            source_layer: 1 for L1->L2, 2 for L2->L1.
            top_k: Number of aligned nodes to return.
        
        Returns:
            List of aligned node indices in target layer.
        """
        if self.s_align is None:
            return []
        
        if source_layer == 1:
            # L1 -> L2: use S_align[node, :]
            scores = self.s_align[node, :]
        else:
            # L2 -> L1: use S_align[:, node]
            scores = self.s_align[:, node]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()
    
    def _estimate_layer_performance(
        self,
        seeds: List[int],
        adj: List[List[int]],
        influence: np.ndarray,
        alpha: np.ndarray,
    ) -> float:
        """Estimate layer performance based on influence coverage."""
        total = 0.0
        covered = set()
        
        for s in seeds:
            zone = get_2hop_neighbors(None, s, adj)
            zone.add(s)
            new_coverage = zone - covered
            
            # Weight by alpha
            for node in new_coverage:
                total += influence[node] * alpha[node]
            
            covered.update(zone)
        
        return total
    
    def gap_filling_search(
        self,
        individual: Individual,
        balance_weight: float = 0.5,
    ) -> LocalSearchResult:
        """
        Gap-Filling Local Search for T3 (Collaborative Robustness).
        
        Identifies the weaker layer and uses S_align to find replacement
        nodes that improve weak layer performance without hurting strong layer.
        
        Args:
            individual: Individual to optimize.
            balance_weight: Weight for cross-layer balance (0=ignore, 1=prioritize).
        
        Returns:
            LocalSearchResult with improvement details.
        """
        seeds_l1 = individual.seeds_l1.copy()
        seeds_l2 = individual.seeds_l2.copy()
        
        # Estimate performance of each layer
        perf_l1 = self._estimate_layer_performance(
            seeds_l1, self.adj_l1, self.influence_l1, self.alpha_l1
        )
        perf_l2 = self._estimate_layer_performance(
            seeds_l2, self.adj_l2, self.influence_l2, self.alpha_l2
        )
        
        # Combined initial fitness
        initial_fitness = perf_l1 + perf_l2 - balance_weight * abs(perf_l1 - perf_l2)
        best_fitness = initial_fitness
        
        improved = False
        iterations = 0
        swaps_made = 0
        
        for iteration in range(self.max_iterations):
            iterations += 1
            best_swap = None
            best_swap_gain = 0.0
            
            # Determine weak layer
            weak_layer = 1 if perf_l1 < perf_l2 else 2
            
            if weak_layer == 1:
                weak_seeds = seeds_l1
                strong_seeds = seeds_l2
                weak_adj = self.adj_l1
                weak_influence = self.influence_l1
                weak_alpha = self.alpha_l1
            else:
                weak_seeds = seeds_l2
                strong_seeds = seeds_l1
                weak_adj = self.adj_l2
                weak_influence = self.influence_l2
                weak_alpha = self.alpha_l2
            
            # Try gap-filling for weak layer using S_align guidance
            current_weak_set = set(weak_seeds)
            
            for idx, seed in enumerate(weak_seeds):
                # Find aligned nodes from strong layer
                aligned_candidates = []
                
                for strong_seed in strong_seeds:
                    # Find what nodes in weak layer align with strong seeds
                    aligned = self._find_aligned_nodes(
                        strong_seed, 
                        source_layer=2 if weak_layer == 1 else 1,
                        top_k=3
                    )
                    aligned_candidates.extend(aligned)
                
                # Remove duplicates and existing seeds
                aligned_candidates = list(set(aligned_candidates) - current_weak_set)
                
                if not aligned_candidates:
                    continue
                
                # Evaluate each candidate
                for candidate in aligned_candidates:
                    # Calculate new performance with swap
                    temp_seeds = weak_seeds.copy()
                    temp_seeds[idx] = candidate
                    
                    new_perf = self._estimate_layer_performance(
                        temp_seeds, weak_adj, weak_influence, weak_alpha
                    )
                    
                    # Calculate gain
                    old_perf = perf_l1 if weak_layer == 1 else perf_l2
                    strong_perf = perf_l2 if weak_layer == 1 else perf_l1
                    
                    new_combined = new_perf + strong_perf - balance_weight * abs(new_perf - strong_perf)
                    swap_gain = new_combined - best_fitness
                    
                    if swap_gain > best_swap_gain + self.improvement_threshold:
                        best_swap_gain = swap_gain
                        best_swap = (weak_layer, idx, candidate, new_perf)
            
            # Also try regular 1-swap for both layers (not just gap-filling)
            for layer in [1, 2]:
                target_seeds = seeds_l1 if layer == 1 else seeds_l2
                target_adj = self.adj_l1 if layer == 1 else self.adj_l2
                target_influence = self.influence_l1 if layer == 1 else self.influence_l2
                target_alpha = self.alpha_l1 if layer == 1 else self.alpha_l2
                target_set = set(target_seeds)
                
                for idx, seed in enumerate(target_seeds):
                    # Regular swap candidates (high influence nodes)
                    candidates = np.argsort(target_influence * target_alpha)[::-1][:20]
                    candidates = [c for c in candidates if c not in target_set]
                    
                    for candidate in candidates:
                        temp_seeds = target_seeds.copy()
                        temp_seeds[idx] = candidate
                        
                        new_perf = self._estimate_layer_performance(
                            temp_seeds, target_adj, target_influence, target_alpha
                        )
                        
                        if layer == 1:
                            new_combined = new_perf + perf_l2 - balance_weight * abs(new_perf - perf_l2)
                        else:
                            new_combined = perf_l1 + new_perf - balance_weight * abs(perf_l1 - new_perf)
                        
                        swap_gain = new_combined - best_fitness
                        
                        if swap_gain > best_swap_gain + self.improvement_threshold:
                            best_swap_gain = swap_gain
                            best_swap = (layer, idx, candidate, new_perf)
            
            if best_swap is not None:
                layer, idx, candidate, new_perf = best_swap
                
                if layer == 1:
                    seeds_l1[idx] = candidate
                    perf_l1 = new_perf
                else:
                    seeds_l2[idx] = candidate
                    perf_l2 = new_perf
                
                best_fitness += best_swap_gain
                swaps_made += 1
                improved = True
            else:
                break
        
        # Update individual
        individual.seeds_l1 = seeds_l1
        individual.seeds_l2 = seeds_l2
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    # =========================================================================
    # Classic 1-Swap and 2-Swap Local Search
    # =========================================================================
    
    def one_swap_search(
        self,
        individual: Individual,
        task_id: TaskID,
        evaluate_fn: Callable[[Individual], float],
    ) -> LocalSearchResult:
        """
        Classic 1-Swap Local Search with fitness function.
        
        For each seed, try replacing it with every candidate and
        keep the best improvement.
        
        Args:
            individual: Individual to optimize.
            task_id: Task being optimized.
            evaluate_fn: Function to evaluate individual fitness.
        
        Returns:
            LocalSearchResult with improvement details.
        """
        if task_id == TaskID.T1:
            seeds = individual.seeds_l1
            layer = 1
        elif task_id == TaskID.T2:
            seeds = individual.seeds_l2
            layer = 2
        else:  # T3: optimize both layers
            return self._one_swap_search_t3(individual, evaluate_fn)
        
        initial_fitness = evaluate_fn(individual)
        best_fitness = initial_fitness
        improved = False
        iterations = 0
        swaps_made = 0
        
        for iteration in range(self.max_iterations):
            iterations += 1
            best_swap = None
            best_swap_fitness = best_fitness
            
            current_set = set(seeds)
            
            for idx in range(len(seeds)):
                old_seed = seeds[idx]
                
                # Try all candidates
                candidates = [n for n in range(self.num_nodes) if n not in current_set]
                
                for candidate in candidates:
                    # Swap
                    seeds[idx] = candidate
                    
                    # Evaluate
                    new_fitness = evaluate_fn(individual)
                    
                    if new_fitness > best_swap_fitness + self.improvement_threshold:
                        best_swap_fitness = new_fitness
                        best_swap = (idx, candidate)
                    
                    # Restore
                    seeds[idx] = old_seed
            
            if best_swap is not None:
                idx, candidate = best_swap
                seeds[idx] = candidate
                best_fitness = best_swap_fitness
                swaps_made += 1
                improved = True
            else:
                break
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    def _one_swap_search_t3(
        self,
        individual: Individual,
        evaluate_fn: Callable[[Individual], float],
    ) -> LocalSearchResult:
        """1-Swap for T3: alternates between layers."""
        initial_fitness = evaluate_fn(individual)
        best_fitness = initial_fitness
        improved = False
        iterations = 0
        swaps_made = 0
        
        for iteration in range(self.max_iterations):
            iterations += 1
            best_swap = None
            best_swap_fitness = best_fitness
            
            # Alternate layers
            for layer in [1, 2]:
                seeds = individual.seeds_l1 if layer == 1 else individual.seeds_l2
                current_set = set(seeds)
                
                for idx in range(len(seeds)):
                    old_seed = seeds[idx]
                    candidates = [n for n in range(self.num_nodes) if n not in current_set]
                    
                    # Sample candidates for efficiency
                    if len(candidates) > 30:
                        candidates = self.rng.choice(candidates, 30, replace=False).tolist()
                    
                    for candidate in candidates:
                        seeds[idx] = candidate
                        new_fitness = evaluate_fn(individual)
                        
                        if new_fitness > best_swap_fitness + self.improvement_threshold:
                            best_swap_fitness = new_fitness
                            best_swap = (layer, idx, candidate)
                        
                        seeds[idx] = old_seed
            
            if best_swap is not None:
                layer, idx, candidate = best_swap
                if layer == 1:
                    individual.seeds_l1[idx] = candidate
                else:
                    individual.seeds_l2[idx] = candidate
                best_fitness = best_swap_fitness
                swaps_made += 1
                improved = True
            else:
                break
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    def two_swap_search(
        self,
        individual: Individual,
        task_id: TaskID,
        evaluate_fn: Callable[[Individual], float],
        sample_size: int = 20,
    ) -> LocalSearchResult:
        """
        2-Swap Local Search.
        
        Tries replacing two seeds simultaneously for larger neighborhood.
        
        Args:
            individual: Individual to optimize.
            task_id: Task being optimized.
            evaluate_fn: Fitness function.
            sample_size: Number of candidate pairs to sample.
        
        Returns:
            LocalSearchResult with improvement details.
        """
        if task_id == TaskID.T3:
            # For T3, use cross-layer 2-swap
            return self._two_swap_cross_layer(individual, evaluate_fn, sample_size)
        
        seeds = individual.seeds_l1 if task_id == TaskID.T1 else individual.seeds_l2
        
        initial_fitness = evaluate_fn(individual)
        best_fitness = initial_fitness
        improved = False
        iterations = 0
        swaps_made = 0
        
        for iteration in range(self.max_iterations // 2):  # Fewer iterations for 2-swap
            iterations += 1
            best_swap = None
            best_swap_fitness = best_fitness
            
            current_set = set(seeds)
            candidates = [n for n in range(self.num_nodes) if n not in current_set]
            
            # Sample index pairs
            if len(seeds) < 2:
                break
            
            for i in range(min(len(seeds) - 1, sample_size)):
                for j in range(i + 1, min(len(seeds), sample_size)):
                    old_i, old_j = seeds[i], seeds[j]
                    
                    # Sample candidate pairs
                    if len(candidates) >= 2:
                        sampled = self.rng.choice(candidates, min(sample_size, len(candidates)), replace=False)
                        
                        for ci_idx in range(len(sampled) - 1):
                            for cj_idx in range(ci_idx + 1, len(sampled)):
                                c1, c2 = sampled[ci_idx], sampled[cj_idx]
                                
                                seeds[i], seeds[j] = c1, c2
                                new_fitness = evaluate_fn(individual)
                                
                                if new_fitness > best_swap_fitness + self.improvement_threshold:
                                    best_swap_fitness = new_fitness
                                    best_swap = (i, j, c1, c2)
                                
                                seeds[i], seeds[j] = old_i, old_j
            
            if best_swap is not None:
                i, j, c1, c2 = best_swap
                seeds[i], seeds[j] = c1, c2
                best_fitness = best_swap_fitness
                swaps_made += 2
                improved = True
            else:
                break
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    def _two_swap_cross_layer(
        self,
        individual: Individual,
        evaluate_fn: Callable[[Individual], float],
        sample_size: int = 20,
    ) -> LocalSearchResult:
        """2-Swap for T3: swap one seed from each layer."""
        initial_fitness = evaluate_fn(individual)
        best_fitness = initial_fitness
        improved = False
        iterations = 0
        swaps_made = 0
        
        for iteration in range(self.max_iterations // 2):
            iterations += 1
            best_swap = None
            best_swap_fitness = best_fitness
            
            seeds_l1 = individual.seeds_l1
            seeds_l2 = individual.seeds_l2
            set_l1 = set(seeds_l1)
            set_l2 = set(seeds_l2)
            
            candidates_l1 = [n for n in range(self.num_nodes) if n not in set_l1]
            candidates_l2 = [n for n in range(self.num_nodes) if n not in set_l2]
            
            # Sample pairs: one from each layer
            for i in range(min(len(seeds_l1), sample_size)):
                for j in range(min(len(seeds_l2), sample_size)):
                    old_l1, old_l2 = seeds_l1[i], seeds_l2[j]
                    
                    # Use S_align to find good cross-layer pairs
                    if self.s_align is not None:
                        # Find candidates aligned with each other
                        aligned_l1 = self._find_aligned_nodes(old_l2, source_layer=2, top_k=5)
                        aligned_l2 = self._find_aligned_nodes(old_l1, source_layer=1, top_k=5)
                        
                        aligned_l1 = [n for n in aligned_l1 if n not in set_l1]
                        aligned_l2 = [n for n in aligned_l2 if n not in set_l2]
                    else:
                        aligned_l1 = self.rng.choice(candidates_l1, min(5, len(candidates_l1)), replace=False).tolist() if candidates_l1 else []
                        aligned_l2 = self.rng.choice(candidates_l2, min(5, len(candidates_l2)), replace=False).tolist() if candidates_l2 else []
                    
                    for c1 in aligned_l1:
                        for c2 in aligned_l2:
                            seeds_l1[i], seeds_l2[j] = c1, c2
                            new_fitness = evaluate_fn(individual)
                            
                            if new_fitness > best_swap_fitness + self.improvement_threshold:
                                best_swap_fitness = new_fitness
                                best_swap = (i, j, c1, c2)
                            
                            seeds_l1[i], seeds_l2[j] = old_l1, old_l2
            
            if best_swap is not None:
                i, j, c1, c2 = best_swap
                seeds_l1[i], seeds_l2[j] = c1, c2
                best_fitness = best_swap_fitness
                swaps_made += 2
                improved = True
            else:
                break
        
        return LocalSearchResult(
            improved=improved,
            old_fitness=initial_fitness,
            new_fitness=best_fitness,
            iterations=iterations,
            swaps_made=swaps_made,
        )
    
    # =========================================================================
    # Greedy Construction (for re-initialization)
    # =========================================================================
    
    def greedy_construction(
        self,
        budget: int,
        layer: int,
        opponent_seeds: Optional[List[int]] = None,
        use_competition_aware: bool = True,
    ) -> List[int]:
        """
        Greedy seed set construction with competition awareness.
        
        Args:
            budget: Number of seeds to select.
            layer: Which layer (1 or 2).
            opponent_seeds: Opponent seeds for competition awareness.
            use_competition_aware: Whether to use competition-aware selection.
        
        Returns:
            List of selected seed indices.
        """
        adj = self.adj_l1 if layer == 1 else self.adj_l2
        influence = self.influence_l1 if layer == 1 else self.influence_l2
        alpha = self.alpha_l1 if layer == 1 else self.alpha_l2
        
        opponent = set(opponent_seeds) if opponent_seeds else set()
        selected = set()
        seeds = []
        
        for _ in range(budget):
            best_node = None
            best_gain = -np.inf
            
            for node in range(self.num_nodes):
                if node in selected:
                    continue
                
                if use_competition_aware and opponent:
                    gain = self.competition_aware_marginal_gain(
                        node, selected, opponent, adj, influence, alpha
                    )
                else:
                    # Simple marginal gain
                    zone = get_2hop_neighbors(None, node, adj)
                    zone.add(node)
                    
                    # Avoid redundancy with existing seeds
                    covered = set()
                    for s in selected:
                        covered.update(get_2hop_neighbors(None, s, adj))
                        covered.add(s)
                    
                    new_coverage = zone - covered
                    gain = sum(influence[n] * alpha[n] for n in new_coverage)
                
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
            
            if best_node is not None:
                selected.add(best_node)
                seeds.append(best_node)
        
        return seeds
    
    # =========================================================================
    # Unified Interface
    # =========================================================================
    
    def apply(
        self,
        individual: Individual,
        task_id: TaskID,
        evaluate_fn: Optional[Callable[[Individual], float]] = None,
        method: str = "auto",
    ) -> LocalSearchResult:
        """
        Apply local search to an individual.
        
        Args:
            individual: Individual to optimize.
            task_id: Task being optimized.
            evaluate_fn: Fitness function (required for 1-swap/2-swap).
            method: Search method ('auto', 'competition', 'gap_filling', 
                    '1swap', '2swap').
        
        Returns:
            LocalSearchResult with improvement details.
        """
        if method == "auto":
            if task_id == TaskID.T3:
                method = "gap_filling"
            else:
                method = "competition"
        
        if method == "competition":
            return self.competition_aware_search(
                individual,
                opponent_seeds_l1=individual.seeds_l2,  # In T1, B is opponent
                opponent_seeds_l2=individual.seeds_l1,  # In T2, A is opponent
                task_id=task_id,
            )
        elif method == "gap_filling":
            return self.gap_filling_search(individual)
        elif method == "1swap":
            if evaluate_fn is None:
                raise ValueError("evaluate_fn required for 1swap method")
            return self.one_swap_search(individual, task_id, evaluate_fn)
        elif method == "2swap":
            if evaluate_fn is None:
                raise ValueError("evaluate_fn required for 2swap method")
            return self.two_swap_search(individual, task_id, evaluate_fn)
        else:
            raise ValueError(f"Unknown method: {method}")
