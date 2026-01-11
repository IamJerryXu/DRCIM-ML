"""
MFEA Population Management Module.

Implements four-group initialization strategy from TECHNICAL_DOC.md §3.1.3:
1. Elite Group (40%): α-weighted sampling for high diffusion potential
2. Robust Group (20%): 2-hop coverage + local connectivity for survivability
3. Aligned Group (20%): S_align-based cross-layer matching pairs
4. Random Group (20%): Uniform sampling for exploration diversity
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum


class TaskID(IntEnum):
    """Task identifiers for MFEA."""
    T1 = 1  # Layer 1 局部最优 (R_CS^(1))
    T2 = 2  # Layer 2 局部最优 (R_CS^(2))
    T3 = 3  # 全局协同最优 (R_CR)


@dataclass
class Individual:
    """
    Represents an individual in the MFEA population.
    
    Attributes:
        seeds_l1: Seed set for Layer 1 (S_A).
        seeds_l2: Seed set for Layer 2 (S_B).
        skill_factor: The task ID (1, 2, or 3) this individual is best at.
        scalar_fitness: The fitness value used for selection (φ = 1/rank).
        factorial_ranks: Dict mapping task_id -> rank on that task.
        factorial_costs: Dict mapping task_id -> raw fitness score.
        init_type: Initialization group type ('elite', 'robust', 'aligned', 'random').
    """
    seeds_l1: List[int] = field(default_factory=list)
    seeds_l2: List[int] = field(default_factory=list)
    skill_factor: Optional[int] = None
    scalar_fitness: float = 0.0
    factorial_ranks: Dict[int, int] = field(default_factory=dict)
    factorial_costs: Dict[int, float] = field(default_factory=dict)
    init_type: str = "random"
    
    @property
    def genes(self) -> Tuple[List[int], List[int]]:
        """Combined gene representation (S_A, S_B)."""
        return (self.seeds_l1.copy(), self.seeds_l2.copy())
    
    @genes.setter
    def genes(self, value: Tuple[List[int], List[int]]):
        if isinstance(value, tuple) and len(value) == 2:
            self.seeds_l1 = list(value[0])
            self.seeds_l2 = list(value[1])
        else:
            raise ValueError("genes must be a tuple of (seeds_l1, seeds_l2)")
    
    def get_flat_genes(self) -> List[int]:
        """Return flattened gene array [S_A | S_B]."""
        return self.seeds_l1 + self.seeds_l2
    
    def copy(self) -> "Individual":
        """Create a deep copy of this individual."""
        return Individual(
            seeds_l1=self.seeds_l1.copy(),
            seeds_l2=self.seeds_l2.copy(),
            skill_factor=self.skill_factor,
            scalar_fitness=self.scalar_fitness,
            factorial_ranks=self.factorial_ranks.copy(),
            factorial_costs=self.factorial_costs.copy(),
            init_type=self.init_type,
        )
    
    def __repr__(self) -> str:
        return (
            f"Individual(τ={self.skill_factor}, φ={self.scalar_fitness:.4f}, "
            f"|S_A|={len(self.seeds_l1)}, |S_B|={len(self.seeds_l2)}, type={self.init_type})"
        )


class PopulationInitializer:
    """
    Four-group initialization strategy for MFEA population.
    
    Based on TECHNICAL_DOC.md §3.1.3:
    - Elite Group (40%): High diffusion potential nodes via α-weighted sampling
    - Robust Group (20%): High 2-hop coverage + clustering coefficient
    - Aligned Group (20%): Cross-layer aligned pairs from S_align matrix
    - Random Group (20%): Uniform random sampling for diversity
    """
    
    def __init__(
        self,
        num_nodes: int,
        budget_l1: int,
        budget_l2: int,
        alpha_l1: Optional[np.ndarray] = None,
        alpha_l2: Optional[np.ndarray] = None,
        s_align: Optional[np.ndarray] = None,
        adj_l1: Optional[List[List[int]]] = None,
        adj_l2: Optional[List[List[int]]] = None,
        elite_ratio: float = 0.4,
        robust_ratio: float = 0.2,
        aligned_ratio: float = 0.2,
        random_ratio: float = 0.2,
        temperature: float = 1.0,
        top_k_align: int = 50,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the population initializer.
        
        Args:
            num_nodes: Total number of nodes in each layer.
            budget_l1: Seed budget for Layer 1 (|S_A|).
            budget_l2: Seed budget for Layer 2 (|S_B|).
            alpha_l1: Diffusion-aware weights for Layer 1 from KAA-GRIT.
            alpha_l2: Diffusion-aware weights for Layer 2 from KAA-GRIT.
            s_align: Cross-layer similarity matrix S_align[i][j] = sim(z_i^(1), z_j^(2)).
            adj_l1: Adjacency list for Layer 1.
            adj_l2: Adjacency list for Layer 2.
            elite_ratio: Fraction of population for elite group (default 0.4).
            robust_ratio: Fraction of population for robust group (default 0.2).
            aligned_ratio: Fraction of population for aligned group (default 0.2).
            random_ratio: Fraction of population for random group (default 0.2).
            temperature: Temperature for softmax sampling in elite group.
            top_k_align: Number of top aligned pairs to consider.
            rng: Random number generator for reproducibility.
        """
        self.num_nodes = num_nodes
        self.budget_l1 = budget_l1
        self.budget_l2 = budget_l2
        
        # Normalize alpha weights
        self.alpha_l1 = self._normalize_alpha(alpha_l1, num_nodes)
        self.alpha_l2 = self._normalize_alpha(alpha_l2, num_nodes)
        
        self.s_align = s_align
        self.adj_l1 = adj_l1
        self.adj_l2 = adj_l2
        
        # Validate and normalize ratios
        total_ratio = elite_ratio + robust_ratio + aligned_ratio + random_ratio
        self.elite_ratio = elite_ratio / total_ratio
        self.robust_ratio = robust_ratio / total_ratio
        self.aligned_ratio = aligned_ratio / total_ratio
        self.random_ratio = random_ratio / total_ratio
        
        self.temperature = temperature
        self.top_k_align = top_k_align
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Precompute node metrics for robust group
        self._precompute_robustness_scores()
    
    def _normalize_alpha(
        self, alpha: Optional[np.ndarray], num_nodes: int
    ) -> np.ndarray:
        """Normalize alpha weights to probability distribution."""
        if alpha is None:
            return np.ones(num_nodes) / num_nodes
        
        alpha = np.asarray(alpha, dtype=np.float64).flatten()
        if len(alpha) < num_nodes:
            # Pad with mean value
            pad = np.full(num_nodes - len(alpha), alpha.mean())
            alpha = np.concatenate([alpha, pad])
        alpha = alpha[:num_nodes]
        
        # Ensure non-negative
        alpha = np.maximum(alpha, 0.0)
        
        # Normalize to sum to 1
        total = alpha.sum()
        if total > 0:
            alpha = alpha / total
        else:
            alpha = np.ones(num_nodes) / num_nodes
        
        return alpha
    
    def _precompute_robustness_scores(self):
        """
        Precompute robustness scores for each node.
        
        Score combines:
        1. 2-hop coverage: Number of nodes reachable within 2 hops
        2. Local clustering coefficient: Measures redundancy of connections
        3. Degree centrality: Basic connectivity measure
        """
        self.robust_scores_l1 = self._compute_layer_robustness(self.adj_l1)
        self.robust_scores_l2 = self._compute_layer_robustness(self.adj_l2)
    
    def _compute_layer_robustness(
        self, adj: Optional[List[List[int]]]
    ) -> np.ndarray:
        """Compute robustness score for each node in a layer."""
        if adj is None:
            return np.ones(self.num_nodes) / self.num_nodes
        
        scores = np.zeros(self.num_nodes, dtype=np.float64)
        
        for node in range(self.num_nodes):
            if node >= len(adj):
                continue
            
            neighbors = set(adj[node])
            degree = len(neighbors)
            
            if degree == 0:
                continue
            
            # 1. 2-hop coverage count
            two_hop_nodes = set()
            for neighbor in neighbors:
                if neighbor < len(adj):
                    two_hop_nodes.update(adj[neighbor])
            two_hop_nodes.discard(node)
            two_hop_coverage = len(two_hop_nodes)
            
            # 2. Local clustering coefficient
            # C = 2 * triangles / (degree * (degree - 1))
            if degree >= 2:
                triangles = 0
                neighbor_list = list(neighbors)
                for i, u in enumerate(neighbor_list):
                    for v in neighbor_list[i + 1:]:
                        if u < len(adj) and v in adj[u]:
                            triangles += 1
                clustering = 2 * triangles / (degree * (degree - 1))
            else:
                clustering = 0.0
            
            # 3. Combined score (weighted sum)
            # Higher weight on 2-hop coverage for influence spread
            # Clustering adds redundancy for robustness under attack
            scores[node] = (
                0.5 * two_hop_coverage / max(self.num_nodes, 1) +
                0.3 * clustering +
                0.2 * degree / max(self.num_nodes, 1)
            )
        
        # Normalize to probability
        total = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            scores = np.ones(self.num_nodes) / self.num_nodes
        
        return scores
    
    def _sample_with_temperature(
        self,
        probs: np.ndarray,
        k: int,
        excluded: Optional[set] = None,
    ) -> List[int]:
        """
        Sample k nodes using temperature-scaled softmax.
        
        Args:
            probs: Probability distribution over nodes.
            k: Number of nodes to sample.
            excluded: Set of nodes to exclude from sampling.
        
        Returns:
            List of sampled node indices (without replacement).
        """
        probs = probs.copy()
        
        # Zero out excluded nodes
        if excluded:
            for node in excluded:
                if 0 <= node < len(probs):
                    probs[node] = 0.0
        
        # Temperature scaling
        if self.temperature != 1.0:
            # Apply temperature to log-probs to avoid overflow
            probs = np.maximum(probs, 1e-10)
            log_probs = np.log(probs) / self.temperature
            log_probs = log_probs - log_probs.max()  # Numerical stability
            probs = np.exp(log_probs)
        
        # Normalize
        total = probs.sum()
        if total <= 0:
            # Fallback to uniform if all probs are zero
            valid_nodes = [i for i in range(len(probs)) 
                          if excluded is None or i not in excluded]
            if len(valid_nodes) <= k:
                return valid_nodes
            return self.rng.choice(valid_nodes, size=k, replace=False).tolist()
        
        probs = probs / total
        
        # Sample without replacement
        k = min(k, np.count_nonzero(probs))
        if k <= 0:
            return []
        
        try:
            sampled = self.rng.choice(
                len(probs), size=k, replace=False, p=probs
            )
            return sampled.tolist()
        except ValueError:
            # Fallback if sampling fails
            valid_nodes = np.where(probs > 0)[0]
            k = min(k, len(valid_nodes))
            return self.rng.choice(valid_nodes, size=k, replace=False).tolist()
    
    def _init_elite_individual(self) -> Individual:
        """
        Initialize an individual using α-weighted sampling (Elite Group).
        
        High diffusion potential nodes are sampled with probability
        proportional to their attention weights from KAA-GRIT.
        """
        # Sample Layer 1 seeds using α_l1
        seeds_l1 = self._sample_with_temperature(
            self.alpha_l1, self.budget_l1
        )
        
        # Sample Layer 2 seeds using α_l2
        seeds_l2 = self._sample_with_temperature(
            self.alpha_l2, self.budget_l2
        )
        
        return Individual(
            seeds_l1=seeds_l1,
            seeds_l2=seeds_l2,
            init_type="elite",
        )
    
    def _init_robust_individual(self) -> Individual:
        """
        Initialize an individual using robustness-aware sampling (Robust Group).
        
        Selects nodes with high 2-hop coverage and local clustering
        to maintain spread capability under node removal attacks.
        """
        # Sample Layer 1 seeds using robustness scores
        seeds_l1 = self._sample_with_temperature(
            self.robust_scores_l1, self.budget_l1
        )
        
        # Sample Layer 2 seeds using robustness scores
        seeds_l2 = self._sample_with_temperature(
            self.robust_scores_l2, self.budget_l2
        )
        
        return Individual(
            seeds_l1=seeds_l1,
            seeds_l2=seeds_l2,
            init_type="robust",
        )
    
    def _init_aligned_individual(self) -> Individual:
        """
        Initialize an individual using cross-layer alignment (Aligned Group).
        
        Uses S_align matrix to find matching pairs (i, j) where node i
        in Layer 1 has high similarity to node j in Layer 2, creating
        "natural collaborators" for Task T3.
        """
        if self.s_align is None:
            # Fallback to random if no alignment matrix
            return self._init_random_individual()
        
        seeds_l1 = []
        seeds_l2 = []
        used_l1 = set()
        used_l2 = set()
        
        # Find top aligned pairs from S_align matrix
        # S_align[i][j] = similarity between node i in L1 and node j in L2
        n1, n2 = self.s_align.shape
        
        # Flatten and get top-k pairs
        flat_indices = np.argsort(self.s_align.flatten())[::-1]
        
        # Sample from top aligned pairs with some randomness
        top_pairs = []
        for flat_idx in flat_indices[:self.top_k_align * 3]:
            i = flat_idx // n2
            j = flat_idx % n2
            if i < self.num_nodes and j < self.num_nodes:
                top_pairs.append((i, j, self.s_align[i, j]))
        
        # Randomly select from top pairs to avoid always picking same nodes
        self.rng.shuffle(top_pairs)
        
        # Greedily select non-overlapping pairs
        budget = min(self.budget_l1, self.budget_l2)
        for i, j, sim in top_pairs:
            if len(seeds_l1) >= budget:
                break
            if i not in used_l1 and j not in used_l2:
                seeds_l1.append(i)
                seeds_l2.append(j)
                used_l1.add(i)
                used_l2.add(j)
        
        # Fill remaining budget with α-weighted sampling
        if len(seeds_l1) < self.budget_l1:
            extra_l1 = self._sample_with_temperature(
                self.alpha_l1,
                self.budget_l1 - len(seeds_l1),
                excluded=used_l1,
            )
            seeds_l1.extend(extra_l1)
        
        if len(seeds_l2) < self.budget_l2:
            extra_l2 = self._sample_with_temperature(
                self.alpha_l2,
                self.budget_l2 - len(seeds_l2),
                excluded=used_l2,
            )
            seeds_l2.extend(extra_l2)
        
        return Individual(
            seeds_l1=seeds_l1,
            seeds_l2=seeds_l2,
            init_type="aligned",
        )
    
    def _init_random_individual(self) -> Individual:
        """
        Initialize an individual using uniform random sampling (Random Group).
        
        Maintains exploration diversity to prevent premature convergence.
        """
        all_nodes = np.arange(self.num_nodes)
        
        seeds_l1 = self.rng.choice(
            all_nodes, size=min(self.budget_l1, self.num_nodes), replace=False
        ).tolist()
        
        seeds_l2 = self.rng.choice(
            all_nodes, size=min(self.budget_l2, self.num_nodes), replace=False
        ).tolist()
        
        return Individual(
            seeds_l1=seeds_l1,
            seeds_l2=seeds_l2,
            init_type="random",
        )
    
    def initialize_population(self, pop_size: int) -> List[Individual]:
        """
        Initialize the full population using four-group strategy.
        
        Args:
            pop_size: Total population size.
        
        Returns:
            List of initialized individuals.
        """
        # Calculate group sizes
        n_elite = int(pop_size * self.elite_ratio)
        n_robust = int(pop_size * self.robust_ratio)
        n_aligned = int(pop_size * self.aligned_ratio)
        n_random = pop_size - n_elite - n_robust - n_aligned
        
        population = []
        
        # 1. Elite Group (40%) - α-weighted sampling
        for _ in range(n_elite):
            population.append(self._init_elite_individual())
        
        # 2. Robust Group (20%) - 2-hop + clustering
        for _ in range(n_robust):
            population.append(self._init_robust_individual())
        
        # 3. Aligned Group (20%) - S_align pairs
        for _ in range(n_aligned):
            population.append(self._init_aligned_individual())
        
        # 4. Random Group (20%) - uniform sampling
        for _ in range(n_random):
            population.append(self._init_random_individual())
        
        # Shuffle to mix groups
        self.rng.shuffle(population)
        
        return population


class Population:
    """
    Manages the MFEA population with four-group initialization.
    
    Supports:
    - Four-group initialization (Elite/Robust/Aligned/Random)
    - Skill factor assignment based on factorial ranks
    - Selection via scalar fitness
    - Population statistics tracking
    """
    
    def __init__(
        self,
        size: int,
        num_nodes: int,
        budget_l1: int,
        budget_l2: int,
        alpha_l1: Optional[np.ndarray] = None,
        alpha_l2: Optional[np.ndarray] = None,
        s_align: Optional[np.ndarray] = None,
        adj_l1: Optional[List[List[int]]] = None,
        adj_l2: Optional[List[List[int]]] = None,
        elite_ratio: float = 0.4,
        robust_ratio: float = 0.2,
        aligned_ratio: float = 0.2,
        random_ratio: float = 0.2,
        temperature: float = 1.0,
        top_k_align: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Initialize the population manager.
        
        Args:
            size: Population size.
            num_nodes: Number of nodes in each layer.
            budget_l1: Seed budget for Layer 1.
            budget_l2: Seed budget for Layer 2.
            alpha_l1: α weights for Layer 1 from KAA-GRIT.
            alpha_l2: α weights for Layer 2 from KAA-GRIT.
            s_align: Cross-layer similarity matrix.
            adj_l1: Adjacency list for Layer 1.
            adj_l2: Adjacency list for Layer 2.
            elite_ratio: Fraction for elite group (default 0.4).
            robust_ratio: Fraction for robust group (default 0.2).
            aligned_ratio: Fraction for aligned group (default 0.2).
            random_ratio: Fraction for random group (default 0.2).
            temperature: Temperature for elite sampling.
            top_k_align: Top-k pairs for aligned group.
            seed: Random seed for reproducibility.
        """
        self.size = size
        self.num_nodes = num_nodes
        self.budget_l1 = budget_l1
        self.budget_l2 = budget_l2
        self.rng = np.random.default_rng(seed)
        
        self.individuals: List[Individual] = []
        self.generation = 0
        
        # Store initializer config
        self._init_config = {
            "alpha_l1": alpha_l1,
            "alpha_l2": alpha_l2,
            "s_align": s_align,
            "adj_l1": adj_l1,
            "adj_l2": adj_l2,
            "elite_ratio": elite_ratio,
            "robust_ratio": robust_ratio,
            "aligned_ratio": aligned_ratio,
            "random_ratio": random_ratio,
            "temperature": temperature,
            "top_k_align": top_k_align,
        }
        
        # Statistics tracking
        self.stats_history: List[Dict] = []
    
    def initialize(self):
        """Initialize population using four-group strategy."""
        initializer = PopulationInitializer(
            num_nodes=self.num_nodes,
            budget_l1=self.budget_l1,
            budget_l2=self.budget_l2,
            rng=self.rng,
            **self._init_config,
        )
        
        self.individuals = initializer.initialize_population(self.size)
        self.generation = 0
        
        # Log initialization stats
        self._log_init_stats()
    
    def _log_init_stats(self):
        """Log statistics about initialized population."""
        type_counts = {"elite": 0, "robust": 0, "aligned": 0, "random": 0}
        for ind in self.individuals:
            if ind.init_type in type_counts:
                type_counts[ind.init_type] += 1
        
        print(f"[Population] Initialized {self.size} individuals:")
        print(f"  - Elite Group:   {type_counts['elite']} ({100*type_counts['elite']/self.size:.1f}%)")
        print(f"  - Robust Group:  {type_counts['robust']} ({100*type_counts['robust']/self.size:.1f}%)")
        print(f"  - Aligned Group: {type_counts['aligned']} ({100*type_counts['aligned']/self.size:.1f}%)")
        print(f"  - Random Group:  {type_counts['random']} ({100*type_counts['random']/self.size:.1f}%)")
    
    def evaluate_and_assign_skill_factors(
        self,
        tasks: "MFEATasks",  # Forward reference
        use_attention: bool = True,
    ):
        """
        Evaluate all individuals on all tasks and assign skill factors.
        
        Based on TECHNICAL_DOC.md §3.1.2:
        1. Compute raw fitness on T1, T2, T3 for each individual
        2. Compute factorial ranks on each task
        3. Assign skill factor τ = argmin(ranks)
        4. Compute scalar fitness φ = 1 / best_rank
        
        Args:
            tasks: MFEATasks object with evaluate_t1, evaluate_t2, evaluate_t3.
            use_attention: Whether to use attention-weighted metrics.
        """
        n = len(self.individuals)
        
        # 1. Evaluate all individuals on all tasks
        costs_t1 = np.zeros(n)
        costs_t2 = np.zeros(n)
        costs_t3 = np.zeros(n)
        
        for i, ind in enumerate(self.individuals):
            # T1: Layer 1 competitive robustness (S_A vs S_B on G1)
            costs_t1[i] = tasks.evaluate_t1(
                ind.seeds_l1,
                opponent_seeds=ind.seeds_l2,
                use_attention=use_attention,
            )
            
            # T2: Layer 2 competitive robustness (S_B vs S_A on G2)
            costs_t2[i] = tasks.evaluate_t2(
                ind.seeds_l2,
                opponent_seeds=ind.seeds_l1,
                use_attention=use_attention,
            )
            
            # T3: Collaborative robustness (joint S_A ∪ S_B)
            costs_t3[i] = tasks.evaluate_t3(
                ind,
                use_attention=use_attention,
            )
            
            # Store raw costs
            ind.factorial_costs = {
                TaskID.T1: costs_t1[i],
                TaskID.T2: costs_t2[i],
                TaskID.T3: costs_t3[i],
            }
        
        # 2. Compute factorial ranks (higher fitness = lower rank = better)
        # For maximization: rank 1 = highest value
        ranks_t1 = self._compute_ranks(costs_t1, minimize=False)
        ranks_t2 = self._compute_ranks(costs_t2, minimize=False)
        ranks_t3 = self._compute_ranks(costs_t3, minimize=False)
        
        # 3. Assign skill factors and scalar fitness
        for i, ind in enumerate(self.individuals):
            ind.factorial_ranks = {
                TaskID.T1: ranks_t1[i],
                TaskID.T2: ranks_t2[i],
                TaskID.T3: ranks_t3[i],
            }
            
            # Skill factor = task with best (lowest) rank
            best_rank = min(ranks_t1[i], ranks_t2[i], ranks_t3[i])
            if ranks_t1[i] == best_rank:
                ind.skill_factor = TaskID.T1
            elif ranks_t2[i] == best_rank:
                ind.skill_factor = TaskID.T2
            else:
                ind.skill_factor = TaskID.T3
            
            # Scalar fitness = 1 / best_rank
            ind.scalar_fitness = 1.0 / best_rank if best_rank > 0 else 0.0
    
    def _compute_ranks(self, values: np.ndarray, minimize: bool = True) -> np.ndarray:
        """
        Compute ranks from values (1 = best).
        
        Args:
            values: Array of fitness values.
            minimize: If True, lower value = better rank. If False, higher = better.
        
        Returns:
            Array of ranks (1-indexed).
        """
        n = len(values)
        if minimize:
            order = np.argsort(values)
        else:
            order = np.argsort(-values)  # Descending for maximization
        
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        return ranks
    
    def get_best_individuals(
        self, task_id: Optional[int] = None, top_k: int = 1
    ) -> List[Individual]:
        """
        Get top-k individuals, optionally filtered by task.
        
        Args:
            task_id: If specified, filter by skill factor.
            top_k: Number of individuals to return.
        
        Returns:
            List of top individuals sorted by scalar fitness.
        """
        candidates = self.individuals
        if task_id is not None:
            candidates = [ind for ind in candidates if ind.skill_factor == task_id]
        
        sorted_inds = sorted(candidates, key=lambda x: x.scalar_fitness, reverse=True)
        return sorted_inds[:top_k]
    
    def get_statistics(self) -> Dict:
        """Get current population statistics."""
        if not self.individuals:
            return {}
        
        fitnesses = [ind.scalar_fitness for ind in self.individuals]
        
        # Task distribution
        task_counts = {TaskID.T1: 0, TaskID.T2: 0, TaskID.T3: 0}
        for ind in self.individuals:
            if ind.skill_factor in task_counts:
                task_counts[ind.skill_factor] += 1
        
        # Best fitness per task
        best_per_task = {}
        for tid in [TaskID.T1, TaskID.T2, TaskID.T3]:
            task_inds = [ind for ind in self.individuals if ind.skill_factor == tid]
            if task_inds:
                best_per_task[tid] = max(ind.scalar_fitness for ind in task_inds)
            else:
                best_per_task[tid] = 0.0
        
        stats = {
            "generation": self.generation,
            "size": len(self.individuals),
            "fitness_mean": np.mean(fitnesses),
            "fitness_std": np.std(fitnesses),
            "fitness_max": np.max(fitnesses),
            "fitness_min": np.min(fitnesses),
            "task_distribution": task_counts,
            "best_per_task": best_per_task,
        }
        
        return stats
    
    def advance_generation(self):
        """Advance to next generation."""
        self.generation += 1
        stats = self.get_statistics()
        self.stats_history.append(stats)
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __iter__(self):
        return iter(self.individuals)
    
    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]
