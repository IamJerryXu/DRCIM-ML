"""
MFEA Main Evolution Loop.

Implements the complete MFEA evolutionary optimization pipeline:
1. Population initialization (four-group strategy)
2. Evaluation and skill factor assignment
3. Parent selection and offspring generation
4. Survival selection
5. Optional local search

Based on TECHNICAL_DOC.md §3.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .population import Population, Individual, TaskID
from .tasks import MFEATasks
from .operators import MFEAOperators, create_offspring_population
from .local_search import LocalSearch


def run_mfea_solver(
    config: Dict,
    s_align: Optional[np.ndarray] = None,
    alpha_l1: Optional[np.ndarray] = None,
    alpha_l2: Optional[np.ndarray] = None,
    embeddings_l1: Optional[np.ndarray] = None,
    embeddings_l2: Optional[np.ndarray] = None,
    graph_l1: Any = None,
    graph_l2: Any = None,
    adj_l1: Optional[List[List[int]]] = None,
    adj_l2: Optional[List[List[int]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Main MFEA Evolution Loop.
    
    Args:
        config: Configuration dictionary with MFEA parameters.
        s_align: Cross-layer similarity matrix from GMA pre-training.
        alpha_l1: Node importance weights for Layer 1.
        alpha_l2: Node importance weights for Layer 2.
        embeddings_l1: Node embeddings for Layer 1.
        embeddings_l2: Node embeddings for Layer 2.
        graph_l1: Layer 1 graph (NetworkX or adjacency list).
        graph_l2: Layer 2 graph.
        adj_l1: Adjacency list for Layer 1.
        adj_l2: Adjacency list for Layer 2.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary containing:
        - best_t1: Best individual for Task 1
        - best_t2: Best individual for Task 2
        - best_t3: Best individual for Task 3
        - history: Evolution history statistics
    """
    print("=" * 60)
    print("Starting MFEA Evolution Solver")
    print("=" * 60)
    
    # ==================== Extract Config ====================
    mfea_cfg = config.get("mfea", {})
    
    num_nodes = mfea_cfg.get("num_nodes", 100)
    budget_l1 = mfea_cfg.get("budget_l1", mfea_cfg.get("seed_budget", 10))
    budget_l2 = mfea_cfg.get("budget_l2", mfea_cfg.get("seed_budget", 10))
    population_size = mfea_cfg.get("population_size", 100)
    num_generations = mfea_cfg.get("num_generations", 100)
    
    # Operator parameters
    crossover_prob = mfea_cfg.get("crossover_prob", 0.9)
    rmp = mfea_cfg.get("rmp", 0.3)
    mutation_prob = mfea_cfg.get("mutation_prob", 0.1)
    gma_mutation_prob = mfea_cfg.get("gma_mutation_prob", 0.2)
    transfer_ratio = mfea_cfg.get("transfer_ratio", 0.3)
    
    # Selection parameters
    elite_per_task = mfea_cfg.get("elite_per_task", 5)
    
    # Local search parameters
    local_search_enabled = mfea_cfg.get("local_search_enabled", False)
    local_search_interval = mfea_cfg.get("local_search_interval", 10)
    
    # Initialization ratios
    elite_ratio = mfea_cfg.get("elite_ratio", 0.4)
    robust_ratio = mfea_cfg.get("robust_ratio", 0.2)
    aligned_ratio = mfea_cfg.get("aligned_ratio", 0.2)
    random_ratio = mfea_cfg.get("random_ratio", 0.2)
    
    # Attack parameters for evaluation
    attack_cfg = config.get("attack", {})
    attack_steps = attack_cfg.get("attack_steps", None)
    attack_ratio = attack_cfg.get("attack_ratio", 0.1)
    attack_mode = attack_cfg.get("attack_mode", "2hop")
    p = mfea_cfg.get("propagation_prob", 0.01)
    
    # Random seed
    seed = mfea_cfg.get("seed", None)
    rng = np.random.default_rng(seed)
    
    if verbose:
        print(f"[Config] num_nodes={num_nodes}, budget=({budget_l1}, {budget_l2})")
        print(f"[Config] population={population_size}, generations={num_generations}")
        print(f"[Config] pc={crossover_prob}, rmp={rmp}, pm={mutation_prob}")
    
    # ==================== Initialize Tasks ====================
    tasks = MFEATasks(
        graph_l1=graph_l1,
        graph_l2=graph_l2,
        attack_steps=attack_steps,
        attack_ratio=attack_ratio,
        attack_mode=attack_mode,
        p=p,
        alpha_l1=alpha_l1,
        alpha_l2=alpha_l2,
        budget_l1=budget_l1,
        budget_l2=budget_l2,
    )
    
    if verbose:
        print("[Tasks] MFEATasks initialized")
    
    # ==================== Initialize Operators ====================
    operators = MFEAOperators(
        num_nodes=num_nodes,
        budget_l1=budget_l1,
        budget_l2=budget_l2,
        s_align=s_align,
        alpha_l1=alpha_l1,
        alpha_l2=alpha_l2,
        embeddings_l1=embeddings_l1,
        embeddings_l2=embeddings_l2,
        crossover_prob=crossover_prob,
        rmp=rmp,
        mutation_prob=mutation_prob,
        gma_mutation_prob=gma_mutation_prob,
        transfer_ratio=transfer_ratio,
        rng=rng,
    )
    
    if verbose:
        print("[Operators] MFEAOperators initialized")
    
    # ==================== Initialize Local Search ====================
    local_searcher = None
    if local_search_enabled:
        local_searcher = LocalSearch(
            graph_l1=graph_l1 if graph_l1 is not None else adj_l1,
            graph_l2=graph_l2 if graph_l2 is not None else adj_l2,
            s_align=s_align,
            alpha_l1=alpha_l1,
            alpha_l2=alpha_l2,
            p=p,
            max_iterations=20,
            improvement_threshold=0.001,
            rng=rng,
        )
        if verbose:
            print("[LocalSearch] LocalSearch initialized")
    
    # ==================== Initialize Population ====================
    population = Population(
        size=population_size,
        num_nodes=num_nodes,
        budget_l1=budget_l1,
        budget_l2=budget_l2,
        alpha_l1=alpha_l1,
        alpha_l2=alpha_l2,
        s_align=s_align,
        adj_l1=adj_l1,
        adj_l2=adj_l2,
        elite_ratio=elite_ratio,
        robust_ratio=robust_ratio,
        aligned_ratio=aligned_ratio,
        random_ratio=random_ratio,
        seed=seed,
    )
    population.initialize()
    
    if verbose:
        print(f"[Population] Initialized {population_size} individuals")
    
    # ==================== Evolution History ====================
    history = {
        "best_fitness_t1": [],
        "best_fitness_t2": [],
        "best_fitness_t3": [],
        "mean_fitness": [],
        "task_distribution": [],
    }
    
    # Track best individuals
    best_t1 = None
    best_t2 = None
    best_t3 = None
    best_score_t1 = -np.inf
    best_score_t2 = -np.inf
    best_score_t3 = -np.inf
    
    # ==================== Evolution Loop ====================
    start_time = time.time()
    
    for gen in range(num_generations):
        gen_start = time.time()
        
        # ----- Step 1: Evaluate and Assign Skill Factors -----
        population.evaluate_and_assign_skill_factors(tasks, use_attention=True)
        
        # ----- Step 2: Track Best Individuals -----
        for ind in population.individuals:
            score_t1 = ind.factorial_costs.get(TaskID.T1, 0)
            score_t2 = ind.factorial_costs.get(TaskID.T2, 0)
            score_t3 = ind.factorial_costs.get(TaskID.T3, 0)
            
            if score_t1 > best_score_t1:
                best_score_t1 = score_t1
                best_t1 = ind.copy()
            if score_t2 > best_score_t2:
                best_score_t2 = score_t2
                best_t2 = ind.copy()
            if score_t3 > best_score_t3:
                best_score_t3 = score_t3
                best_t3 = ind.copy()
        
        # ----- Step 3: Record History -----
        stats = population.get_statistics()
        history["best_fitness_t1"].append(best_score_t1)
        history["best_fitness_t2"].append(best_score_t2)
        history["best_fitness_t3"].append(best_score_t3)
        history["mean_fitness"].append(stats.get("fitness_mean", 0))
        history["task_distribution"].append(stats.get("task_distribution", {}))
        
        # ----- Step 4: Generate Offspring -----
        offspring = create_offspring_population(
            population.individuals,
            operators,
            n_offspring=population_size,
        )
        
        # ----- Step 5: Evaluate Offspring -----
        # Create temporary population for evaluation
        for ind in offspring:
            # Evaluate on all tasks
            ind.factorial_costs = {
                TaskID.T1: tasks.evaluate_t1(
                    ind.seeds_l1, opponent_seeds=ind.seeds_l2, use_attention=True
                ),
                TaskID.T2: tasks.evaluate_t2(
                    ind.seeds_l2, opponent_seeds=ind.seeds_l1, use_attention=True
                ),
                TaskID.T3: tasks.evaluate_t3(ind, use_attention=True),
            }
            
            # Update best if offspring is better
            score_t1 = ind.factorial_costs.get(TaskID.T1, 0)
            score_t2 = ind.factorial_costs.get(TaskID.T2, 0)
            score_t3 = ind.factorial_costs.get(TaskID.T3, 0)
            
            if score_t1 > best_score_t1:
                best_score_t1 = score_t1
                best_t1 = ind.copy()
            if score_t2 > best_score_t2:
                best_score_t2 = score_t2
                best_t2 = ind.copy()
            if score_t3 > best_score_t3:
                best_score_t3 = score_t3
                best_t3 = ind.copy()
        
        # Compute ranks for offspring
        _assign_ranks_and_skill_factors(offspring)
        
        # ----- Step 6: Survival Selection -----
        combined = population.individuals + offspring
        survivors = MFEAOperators.survival_selection(
            combined,
            target_size=population_size,
            elite_per_task=elite_per_task,
            ensure_task_balance=True,
        )
        population.individuals = survivors
        
        # ----- Step 7: Optional Local Search -----
        if local_search_enabled and local_searcher is not None and (gen + 1) % local_search_interval == 0:
            # Apply local search to top individuals per task
            ls_improved = 0
            
            # Get top individuals per task
            top_per_task = {}
            for task_id in [TaskID.T1, TaskID.T2, TaskID.T3]:
                task_inds = [ind for ind in population.individuals if ind.skill_factor == task_id]
                if task_inds:
                    task_inds.sort(key=lambda x: x.factorial_costs.get(task_id, 0), reverse=True)
                    top_per_task[task_id] = task_inds[:2]  # Top 2 per task
            
            # Apply appropriate local search
            for task_id, inds in top_per_task.items():
                for ind in inds:
                    if task_id == TaskID.T3:
                        result = local_searcher.gap_filling_search(ind)
                    else:
                        result = local_searcher.competition_aware_search(
                            ind,
                            opponent_seeds_l1=ind.seeds_l2,
                            opponent_seeds_l2=ind.seeds_l1,
                            task_id=task_id,
                        )
                    
                    if result.improved:
                        ls_improved += 1
                        # Re-evaluate after local search
                        ind.factorial_costs = {
                            TaskID.T1: tasks.evaluate_t1(
                                ind.seeds_l1, opponent_seeds=ind.seeds_l2, use_attention=True
                            ),
                            TaskID.T2: tasks.evaluate_t2(
                                ind.seeds_l2, opponent_seeds=ind.seeds_l1, use_attention=True
                            ),
                            TaskID.T3: tasks.evaluate_t3(ind, use_attention=True),
                        }
            
            if verbose and ls_improved > 0:
                print(f"  [LS Gen {gen+1}] Improved {ls_improved} individuals")
        
        # ----- Step 8: Advance Generation -----
        population.advance_generation()
        gen_time = time.time() - gen_start
        
        # ----- Logging -----
        if verbose and (gen % 5 == 0 or gen == num_generations - 1):
            task_dist = stats.get("task_distribution", {})
            # 直接显示 Rcs 值 (平均每步影响节点数)
            print(
                f"[Gen {gen+1:3d}/{num_generations}] "
                f"Rcs: T1={best_score_t1:.2f}, T2={best_score_t2:.2f}, T3={best_score_t3:.2f} | "
                f"τ dist: T1={task_dist.get(TaskID.T1, 0)}, "
                f"T2={task_dist.get(TaskID.T2, 0)}, "
                f"T3={task_dist.get(TaskID.T3, 0)} | "
                f"{gen_time:.2f}s"
            )
    
    # ==================== Final Results ====================
    total_time = time.time() - start_time
    
    print("=" * 60)
    print("MFEA Evolution Completed!")
    print(f"Total time: {total_time:.2f}s ({total_time/num_generations:.2f}s/gen)")
    print(f"Best Rcs T1: {best_score_t1:.4f}")
    print(f"Best Rcs T2: {best_score_t2:.4f}")
    print(f"Best Rcs T3: {best_score_t3:.4f}")
    print("=" * 60)
    
    if best_t1:
        print(f"Best T1 Seeds: S_A={best_t1.seeds_l1[:5]}..., S_B={best_t1.seeds_l2[:5]}...")
    if best_t3:
        print(f"Best T3 Seeds: S_A={best_t3.seeds_l1[:5]}..., S_B={best_t3.seeds_l2[:5]}...")
    
    return {
        "best_t1": best_t1,
        "best_t2": best_t2,
        "best_t3": best_t3,
        "best_scores": {
            "T1": best_score_t1,
            "T2": best_score_t2,
            "T3": best_score_t3,
        },
        "history": history,
        "final_population": population.individuals,
    }


def _assign_ranks_and_skill_factors(individuals: List[Individual]):
    """
    Assign factorial ranks and skill factors to a list of individuals.
    
    Helper function for offspring evaluation.
    """
    n = len(individuals)
    if n == 0:
        return
    
    # Extract costs
    costs_t1 = np.array([ind.factorial_costs.get(TaskID.T1, 0) for ind in individuals])
    costs_t2 = np.array([ind.factorial_costs.get(TaskID.T2, 0) for ind in individuals])
    costs_t3 = np.array([ind.factorial_costs.get(TaskID.T3, 0) for ind in individuals])
    
    # Compute ranks (higher is better, so descending order)
    ranks_t1 = _compute_ranks(costs_t1)
    ranks_t2 = _compute_ranks(costs_t2)
    ranks_t3 = _compute_ranks(costs_t3)
    
    # Assign to individuals
    for i, ind in enumerate(individuals):
        ind.factorial_ranks = {
            TaskID.T1: ranks_t1[i],
            TaskID.T2: ranks_t2[i],
            TaskID.T3: ranks_t3[i],
        }
        
        # Skill factor = task with best rank
        best_rank = min(ranks_t1[i], ranks_t2[i], ranks_t3[i])
        if ranks_t1[i] == best_rank:
            ind.skill_factor = TaskID.T1
        elif ranks_t2[i] == best_rank:
            ind.skill_factor = TaskID.T2
        else:
            ind.skill_factor = TaskID.T3
        
        # Scalar fitness
        ind.scalar_fitness = 1.0 / best_rank if best_rank > 0 else 0.0


def _compute_ranks(values: np.ndarray) -> np.ndarray:
    """Compute ranks (1 = best, higher value = lower rank)."""
    n = len(values)
    order = np.argsort(-values)  # Descending
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    return ranks
