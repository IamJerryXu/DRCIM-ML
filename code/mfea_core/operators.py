"""
MFEA Genetic Operators Module.

Implements GMA-guided crossover, mutation, and selection strategies
based on TECHNICAL_DOC.md §3.2, §3.3, §3.4.

Key Features:
1. GMA-Guided Crossover: Uses S_align for cross-task gene translation
2. Hybrid Mutation: Random + GMA-neighborhood mutation
3. Elitist Selection: Preserves top individuals per task
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from .population import Individual, TaskID


class MFEAOperators:
    """
    Genetic Operators for MFEA with GMA guidance.
    
    Provides crossover, mutation, and selection operators that leverage
    the S_align similarity matrix from GMA pre-training.
    """
    
    def __init__(
        self,
        num_nodes: int,
        budget_l1: int,
        budget_l2: int,
        s_align: Optional[np.ndarray] = None,
        alpha_l1: Optional[np.ndarray] = None,
        alpha_l2: Optional[np.ndarray] = None,
        embeddings_l1: Optional[np.ndarray] = None,
        embeddings_l2: Optional[np.ndarray] = None,
        crossover_prob: float = 0.9,
        rmp: float = 0.3,
        mutation_prob: float = 0.1,
        gma_mutation_prob: float = 0.2,
        transfer_ratio: float = 0.3,
        neighbor_k: int = 10,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize MFEA operators.
        
        Args:
            num_nodes: Number of nodes in each layer.
            budget_l1: Seed budget for Layer 1 (|S_A|).
            budget_l2: Seed budget for Layer 2 (|S_B|).
            s_align: Cross-layer similarity matrix S_align[i][j].
            alpha_l1: Node importance weights for Layer 1.
            alpha_l2: Node importance weights for Layer 2.
            embeddings_l1: Node embeddings for Layer 1 (for GMA mutation).
            embeddings_l2: Node embeddings for Layer 2.
            crossover_prob: Probability of crossover (pc).
            rmp: Random mating probability for cross-task crossover.
            mutation_prob: Random mutation probability (pm).
            gma_mutation_prob: GMA-neighborhood mutation probability.
            transfer_ratio: Fraction of genes to transfer in cross-task crossover.
            neighbor_k: Number of neighbors for GMA-neighborhood mutation.
            rng: Random number generator.
        """
        self.num_nodes = num_nodes
        self.budget_l1 = budget_l1
        self.budget_l2 = budget_l2
        
        self.s_align = s_align
        self.alpha_l1 = self._normalize_alpha(alpha_l1)
        self.alpha_l2 = self._normalize_alpha(alpha_l2)
        self.embeddings_l1 = embeddings_l1
        self.embeddings_l2 = embeddings_l2
        
        self.crossover_prob = crossover_prob
        self.rmp = rmp
        self.mutation_prob = mutation_prob
        self.gma_mutation_prob = gma_mutation_prob
        self.transfer_ratio = transfer_ratio
        self.neighbor_k = neighbor_k
        
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Precompute embedding neighbors for GMA mutation
        self._precompute_neighbors()
    
    def _normalize_alpha(self, alpha: Optional[np.ndarray]) -> np.ndarray:
        """Normalize alpha to probability distribution."""
        if alpha is None:
            return np.ones(self.num_nodes) / self.num_nodes
        alpha = np.asarray(alpha, dtype=np.float64).flatten()[:self.num_nodes]
        if len(alpha) < self.num_nodes:
            alpha = np.pad(alpha, (0, self.num_nodes - len(alpha)), 
                          constant_values=alpha.mean())
        alpha = np.maximum(alpha, 1e-10)
        return alpha / alpha.sum()
    
    def _precompute_neighbors(self):
        """Precompute k-nearest neighbors in embedding space."""
        self.neighbors_l1 = None
        self.neighbors_l2 = None
        
        if self.embeddings_l1 is not None:
            self.neighbors_l1 = self._compute_knn(self.embeddings_l1)
        if self.embeddings_l2 is not None:
            self.neighbors_l2 = self._compute_knn(self.embeddings_l2)
    
    def _compute_knn(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute k-nearest neighbors for each node."""
        n = min(len(embeddings), self.num_nodes)
        k = min(self.neighbor_k, n - 1)
        
        # Compute pairwise distances
        # Using cosine distance for efficiency
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        similarity = normalized @ normalized.T
        
        # Get k nearest (highest similarity, excluding self)
        neighbors = np.zeros((n, k), dtype=np.int32)
        for i in range(n):
            sim_i = similarity[i].copy()
            sim_i[i] = -np.inf  # Exclude self
            neighbors[i] = np.argsort(sim_i)[-k:][::-1]
        
        return neighbors
    
    # ==================== CROSSOVER ====================
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        
        Decides between same-task and cross-task crossover based on
        skill factors and random mating probability (rmp).
        
        Args:
            parent1: First parent individual.
            parent2: Second parent individual.
        
        Returns:
            Tuple of two offspring individuals.
        """
        # Check if crossover happens
        if self.rng.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Check if same task or different task
        same_task = (parent1.skill_factor == parent2.skill_factor)
        
        if same_task:
            # Same-task crossover: standard crossover
            return self._standard_crossover(parent1, parent2)
        else:
            # Cross-task crossover: GMA-guided with probability rmp
            if self.rng.random() < self.rmp:
                return self._gma_guided_crossover(parent1, parent2)
            else:
                # No crossover, just copy
                return parent1.copy(), parent2.copy()
    
    def _standard_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        Standard single-point crossover for same-task parents.
        
        Performs crossover on S_A and S_B independently.
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover S_A (seeds_l1)
        offspring1.seeds_l1, offspring2.seeds_l1 = self._single_point_crossover(
            parent1.seeds_l1, parent2.seeds_l1, self.budget_l1
        )
        
        # Crossover S_B (seeds_l2)
        offspring1.seeds_l2, offspring2.seeds_l2 = self._single_point_crossover(
            parent1.seeds_l2, parent2.seeds_l2, self.budget_l2
        )
        
        # Reset fitness (needs re-evaluation)
        self._reset_fitness(offspring1)
        self._reset_fitness(offspring2)
        offspring1.init_type = "crossover"
        offspring2.init_type = "crossover"
        
        return offspring1, offspring2
    
    def _single_point_crossover(
        self,
        genes1: List[int],
        genes2: List[int],
        budget: int,
    ) -> Tuple[List[int], List[int]]:
        """Single-point crossover with deduplication."""
        if len(genes1) == 0 or len(genes2) == 0:
            return genes1.copy(), genes2.copy()
        
        # Random crossover point
        point = self.rng.integers(1, max(2, len(genes1)))
        
        # Create offspring
        child1 = genes1[:point] + genes2[point:]
        child2 = genes2[:point] + genes1[point:]
        
        # Deduplicate and repair
        child1 = self._repair_genes(child1, budget)
        child2 = self._repair_genes(child2, budget)
        
        return child1, child2
    
    def _gma_guided_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        GMA-guided crossover for cross-task parents.
        
        Uses S_align matrix to translate genes between layers:
        - When transferring node u from T1 to T2, find v = argmax S_align[u][:]
        - This preserves "functional role" across layers
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Determine which parent is better at which task
        # and transfer knowledge accordingly
        
        # Number of genes to transfer
        n_transfer_l1 = max(1, int(self.budget_l1 * self.transfer_ratio))
        n_transfer_l2 = max(1, int(self.budget_l2 * self.transfer_ratio))
        
        # Select valuable genes to transfer based on alpha weights
        genes_from_p1_l1 = self._select_valuable_genes(
            parent1.seeds_l1, self.alpha_l1, n_transfer_l1
        )
        genes_from_p1_l2 = self._select_valuable_genes(
            parent1.seeds_l2, self.alpha_l2, n_transfer_l2
        )
        genes_from_p2_l1 = self._select_valuable_genes(
            parent2.seeds_l1, self.alpha_l1, n_transfer_l1
        )
        genes_from_p2_l2 = self._select_valuable_genes(
            parent2.seeds_l2, self.alpha_l2, n_transfer_l2
        )
        
        # Cross-layer translation using S_align
        if self.s_align is not None:
            # Translate L1 genes to L2 equivalents and vice versa
            translated_p1_l1_to_l2 = self._translate_genes_l1_to_l2(genes_from_p1_l1)
            translated_p1_l2_to_l1 = self._translate_genes_l2_to_l1(genes_from_p1_l2)
            translated_p2_l1_to_l2 = self._translate_genes_l1_to_l2(genes_from_p2_l1)
            translated_p2_l2_to_l1 = self._translate_genes_l2_to_l1(genes_from_p2_l2)
            
            # Inject translated genes into offspring
            # offspring1 gets genes from parent2, offspring2 gets genes from parent1
            offspring1.seeds_l1 = self._inject_genes(
                offspring1.seeds_l1, genes_from_p2_l1 + translated_p2_l2_to_l1,
                self.budget_l1, self.alpha_l1
            )
            offspring1.seeds_l2 = self._inject_genes(
                offspring1.seeds_l2, genes_from_p2_l2 + translated_p2_l1_to_l2,
                self.budget_l2, self.alpha_l2
            )
            offspring2.seeds_l1 = self._inject_genes(
                offspring2.seeds_l1, genes_from_p1_l1 + translated_p1_l2_to_l1,
                self.budget_l1, self.alpha_l1
            )
            offspring2.seeds_l2 = self._inject_genes(
                offspring2.seeds_l2, genes_from_p1_l2 + translated_p1_l1_to_l2,
                self.budget_l2, self.alpha_l2
            )
        else:
            # No S_align: direct gene transfer (may cause negative transfer)
            offspring1.seeds_l1 = self._inject_genes(
                offspring1.seeds_l1, genes_from_p2_l1, self.budget_l1, self.alpha_l1
            )
            offspring1.seeds_l2 = self._inject_genes(
                offspring1.seeds_l2, genes_from_p2_l2, self.budget_l2, self.alpha_l2
            )
            offspring2.seeds_l1 = self._inject_genes(
                offspring2.seeds_l1, genes_from_p1_l1, self.budget_l1, self.alpha_l1
            )
            offspring2.seeds_l2 = self._inject_genes(
                offspring2.seeds_l2, genes_from_p1_l2, self.budget_l2, self.alpha_l2
            )
        
        self._reset_fitness(offspring1)
        self._reset_fitness(offspring2)
        offspring1.init_type = "gma_crossover"
        offspring2.init_type = "gma_crossover"
        
        return offspring1, offspring2
    
    def _select_valuable_genes(
        self,
        genes: List[int],
        alpha: np.ndarray,
        k: int,
    ) -> List[int]:
        """Select k most valuable genes based on alpha weights."""
        if len(genes) == 0 or k <= 0:
            return []
        
        k = min(k, len(genes))
        
        # Get alpha values for genes
        gene_values = [(g, alpha[g]) for g in genes if 0 <= g < len(alpha)]
        
        # Sort by alpha descending
        gene_values.sort(key=lambda x: x[1], reverse=True)
        
        return [g for g, _ in gene_values[:k]]
    
    def _translate_genes_l1_to_l2(self, genes: List[int]) -> List[int]:
        """Translate Layer 1 genes to Layer 2 using S_align."""
        if self.s_align is None or len(genes) == 0:
            return []
        
        translated = []
        for u in genes:
            if 0 <= u < self.s_align.shape[0]:
                # Find most similar node in L2
                v = np.argmax(self.s_align[u, :])
                translated.append(int(v))
        
        return translated
    
    def _translate_genes_l2_to_l1(self, genes: List[int]) -> List[int]:
        """Translate Layer 2 genes to Layer 1 using S_align."""
        if self.s_align is None or len(genes) == 0:
            return []
        
        translated = []
        for v in genes:
            if 0 <= v < self.s_align.shape[1]:
                # Find most similar node in L1
                u = np.argmax(self.s_align[:, v])
                translated.append(int(u))
        
        return translated
    
    def _inject_genes(
        self,
        current_genes: List[int],
        new_genes: List[int],
        budget: int,
        alpha: np.ndarray,
    ) -> List[int]:
        """
        Inject new genes into current gene set.
        
        Replaces lowest-alpha genes with new genes.
        """
        if len(new_genes) == 0:
            return current_genes
        
        result = list(current_genes)
        new_genes_set = set(new_genes)
        
        # Remove duplicates from new genes
        new_genes_filtered = [g for g in new_genes 
                             if 0 <= g < self.num_nodes and g not in result]
        
        if len(new_genes_filtered) == 0:
            return result
        
        # Find genes to replace (lowest alpha in current)
        current_with_alpha = [(g, alpha[g]) for g in result if 0 <= g < len(alpha)]
        current_with_alpha.sort(key=lambda x: x[1])  # Ascending
        
        n_replace = min(len(new_genes_filtered), len(current_with_alpha))
        genes_to_remove = set(g for g, _ in current_with_alpha[:n_replace])
        
        # Remove low-value genes
        result = [g for g in result if g not in genes_to_remove]
        
        # Add new genes
        result.extend(new_genes_filtered[:n_replace])
        
        # Ensure correct size
        return self._repair_genes(result, budget)
    
    def _repair_genes(self, genes: List[int], budget: int) -> List[int]:
        """Repair gene list: deduplicate and ensure correct size."""
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for g in genes:
            if g not in seen and 0 <= g < self.num_nodes:
                seen.add(g)
                unique.append(g)
        
        # Adjust size
        if len(unique) > budget:
            unique = unique[:budget]
        elif len(unique) < budget:
            # Fill with random nodes
            candidates = [i for i in range(self.num_nodes) if i not in seen]
            if candidates:
                n_fill = budget - len(unique)
                fill = self.rng.choice(candidates, size=min(n_fill, len(candidates)), 
                                       replace=False)
                unique.extend(fill.tolist())
        
        return unique
    
    def _reset_fitness(self, individual: Individual):
        """Reset fitness values after genetic operations."""
        individual.skill_factor = None
        individual.scalar_fitness = 0.0
        individual.factorial_ranks = {}
        individual.factorial_costs = {}
    
    # ==================== MUTATION ====================
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Apply hybrid mutation to an individual.
        
        Two mutation types:
        1. Random mutation: Replace with random node
        2. GMA-neighborhood mutation: Replace with similar node in embedding space
        
        Args:
            individual: Individual to mutate.
        
        Returns:
            Mutated individual (modified in-place and returned).
        """
        # Mutate S_A (seeds_l1)
        individual.seeds_l1 = self._mutate_genes(
            individual.seeds_l1,
            self.alpha_l1,
            self.neighbors_l1,
            self.budget_l1,
        )
        
        # Mutate S_B (seeds_l2)
        individual.seeds_l2 = self._mutate_genes(
            individual.seeds_l2,
            self.alpha_l2,
            self.neighbors_l2,
            self.budget_l2,
        )
        
        self._reset_fitness(individual)
        
        return individual
    
    def _mutate_genes(
        self,
        genes: List[int],
        alpha: np.ndarray,
        neighbors: Optional[np.ndarray],
        budget: int,
    ) -> List[int]:
        """Apply mutation to a gene list."""
        result = list(genes)
        used = set(result)
        
        for i in range(len(result)):
            r = self.rng.random()
            
            if r < self.mutation_prob:
                # Random mutation
                candidates = [n for n in range(self.num_nodes) if n not in used]
                if candidates:
                    # Weighted by alpha
                    weights = np.array([alpha[c] for c in candidates])
                    weights = weights / weights.sum()
                    new_gene = self.rng.choice(candidates, p=weights)
                    used.discard(result[i])
                    used.add(new_gene)
                    result[i] = new_gene
                    
            elif r < self.mutation_prob + self.gma_mutation_prob:
                # GMA-neighborhood mutation
                if neighbors is not None and result[i] < len(neighbors):
                    # Get neighbors of current gene
                    nbrs = neighbors[result[i]]
                    # Filter out already used
                    valid_nbrs = [n for n in nbrs if n not in used]
                    if valid_nbrs:
                        # Select neighbor with highest alpha
                        nbr_alpha = [(n, alpha[n]) for n in valid_nbrs]
                        nbr_alpha.sort(key=lambda x: x[1], reverse=True)
                        new_gene = nbr_alpha[0][0]
                        used.discard(result[i])
                        used.add(new_gene)
                        result[i] = new_gene
        
        return self._repair_genes(result, budget)
    
    # ==================== SELECTION ====================
    
    @staticmethod
    def tournament_select(
        population: List[Individual],
        tournament_size: int = 2,
        skill_factor: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Individual:
        """
        Binary tournament selection.
        
        Args:
            population: List of individuals.
            tournament_size: Number of candidates in tournament.
            skill_factor: If specified, only consider individuals with this skill factor.
            rng: Random number generator.
        
        Returns:
            Selected individual.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Filter by skill factor if specified
        candidates = population
        if skill_factor is not None:
            candidates = [ind for ind in population if ind.skill_factor == skill_factor]
            if not candidates:
                candidates = population
        
        # Select tournament participants
        tournament_size = min(tournament_size, len(candidates))
        participants_idx = rng.choice(len(candidates), size=tournament_size, replace=False)
        participants = [candidates[i] for i in participants_idx]
        
        # Return the one with highest scalar fitness
        return max(participants, key=lambda x: x.scalar_fitness)
    
    @staticmethod
    def survival_selection(
        combined_population: List[Individual],
        target_size: int,
        elite_per_task: int = 5,
        ensure_task_balance: bool = True,
    ) -> List[Individual]:
        """
        Elitist survival selection with task balancing.
        
        Strategy:
        1. Preserve top-K elites for each task
        2. Fill remaining slots by scalar fitness
        3. Optionally ensure minimum individuals per task
        
        Args:
            combined_population: R = P ∪ Q (parent + offspring).
            target_size: Target population size N.
            elite_per_task: Number of elites to preserve per task.
            ensure_task_balance: Ensure minimum representation per task.
        
        Returns:
            Selected population of size target_size.
        """
        survivors = []
        survivor_set = set()
        
        # Step 1: Preserve elites for each task
        for task in [TaskID.T1, TaskID.T2, TaskID.T3]:
            task_inds = [ind for ind in combined_population 
                        if ind.skill_factor == task]
            
            if not task_inds:
                continue
            
            # Sort by task-specific fitness (factorial_costs)
            task_inds_sorted = sorted(
                task_inds,
                key=lambda x: x.factorial_costs.get(task, 0),
                reverse=True
            )
            
            # Add top elites
            for ind in task_inds_sorted[:elite_per_task]:
                if id(ind) not in survivor_set:
                    survivors.append(ind)
                    survivor_set.add(id(ind))
        
        # Step 2: If task balance required, ensure minimum per task
        if ensure_task_balance:
            min_per_task = max(1, target_size // 6)  # At least ~17% per task
            
            for task in [TaskID.T1, TaskID.T2, TaskID.T3]:
                current_count = sum(1 for ind in survivors if ind.skill_factor == task)
                
                if current_count < min_per_task:
                    # Add more from this task
                    task_inds = [ind for ind in combined_population 
                                if ind.skill_factor == task and id(ind) not in survivor_set]
                    task_inds_sorted = sorted(
                        task_inds,
                        key=lambda x: x.scalar_fitness,
                        reverse=True
                    )
                    
                    need = min_per_task - current_count
                    for ind in task_inds_sorted[:need]:
                        survivors.append(ind)
                        survivor_set.add(id(ind))
        
        # Step 3: Fill remaining slots by scalar fitness
        remaining = target_size - len(survivors)
        
        if remaining > 0:
            # Get candidates not yet selected
            candidates = [ind for ind in combined_population 
                         if id(ind) not in survivor_set]
            
            # Sort by scalar fitness
            candidates_sorted = sorted(
                candidates,
                key=lambda x: x.scalar_fitness,
                reverse=True
            )
            
            survivors.extend(candidates_sorted[:remaining])
        
        # Truncate if too many (shouldn't happen)
        return survivors[:target_size]
    
    @staticmethod
    def select_parents(
        population: List[Individual],
        n_pairs: int,
        cross_task_prob: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Tuple[Individual, Individual]]:
        """
        Select parent pairs for crossover.
        
        With probability cross_task_prob, select parents from different tasks
        to encourage knowledge transfer.
        
        Args:
            population: Current population.
            n_pairs: Number of parent pairs to select.
            cross_task_prob: Probability of cross-task pairing.
            rng: Random number generator.
        
        Returns:
            List of (parent1, parent2) tuples.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        pairs = []
        
        # Group by skill factor
        task_groups = {
            TaskID.T1: [ind for ind in population if ind.skill_factor == TaskID.T1],
            TaskID.T2: [ind for ind in population if ind.skill_factor == TaskID.T2],
            TaskID.T3: [ind for ind in population if ind.skill_factor == TaskID.T3],
        }
        
        for _ in range(n_pairs):
            if rng.random() < cross_task_prob:
                # Cross-task pairing
                available_tasks = [t for t, inds in task_groups.items() if len(inds) >= 1]
                if len(available_tasks) >= 2:
                    task1, task2 = rng.choice(available_tasks, size=2, replace=False)
                    parent1 = MFEAOperators.tournament_select(
                        task_groups[task1], rng=rng
                    )
                    parent2 = MFEAOperators.tournament_select(
                        task_groups[task2], rng=rng
                    )
                else:
                    # Fallback to any
                    parent1 = MFEAOperators.tournament_select(population, rng=rng)
                    parent2 = MFEAOperators.tournament_select(population, rng=rng)
            else:
                # Same-task pairing (or random)
                parent1 = MFEAOperators.tournament_select(population, rng=rng)
                # Try to find same-task partner
                if parent1.skill_factor is not None and task_groups[parent1.skill_factor]:
                    parent2 = MFEAOperators.tournament_select(
                        task_groups[parent1.skill_factor], rng=rng
                    )
                else:
                    parent2 = MFEAOperators.tournament_select(population, rng=rng)
            
            pairs.append((parent1, parent2))
        
        return pairs


# ==================== CONVENIENCE FUNCTIONS ====================

def create_offspring_population(
    population: List[Individual],
    operators: MFEAOperators,
    n_offspring: Optional[int] = None,
) -> List[Individual]:
    """
    Create offspring population through crossover and mutation.
    
    Args:
        population: Current parent population.
        operators: MFEA operators instance.
        n_offspring: Number of offspring to generate (default: same as population).
    
    Returns:
        List of offspring individuals.
    """
    if n_offspring is None:
        n_offspring = len(population)
    
    n_pairs = (n_offspring + 1) // 2
    
    # Select parent pairs
    pairs = operators.select_parents(
        population, n_pairs,
        cross_task_prob=operators.rmp,
        rng=operators.rng,
    )
    
    # Generate offspring through crossover
    offspring = []
    for parent1, parent2 in pairs:
        child1, child2 = operators.crossover(parent1, parent2)
        offspring.append(child1)
        if len(offspring) < n_offspring:
            offspring.append(child2)
    
    # Apply mutation
    for ind in offspring:
        operators.mutate(ind)
    
    return offspring[:n_offspring]
