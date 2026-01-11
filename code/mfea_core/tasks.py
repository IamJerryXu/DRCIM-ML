import numpy as np
from ..evaluation.calc_metrics import MetricsCalculator


class MFEATasks:
    """
    Defines the optimization tasks for MFEA.

    Task 1 (T1): Layer 1 Influence Maximization (R_CS_1)
    Task 2 (T2): Layer 2 Influence Maximization (R_CS_2)
    Task 3 (T3): Collaborative Robustness (R_CR)
    """

    def __init__(
        self,
        graph_l1,
        graph_l2,
        attack_steps=None,
        attack_ratio=None,
        attack_mode="2hop",
        p=0.01,
        alpha_l1=None,
        alpha_l2=None,
        budget_l1=None,
        budget_l2=None,
    ):
        self.graph_l1 = graph_l1
        self.graph_l2 = graph_l2
        self.attack_steps = attack_steps
        self.attack_ratio = attack_ratio
        self.attack_mode = attack_mode
        self.p = p
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.budget_l1 = budget_l1
        self.budget_l2 = budget_l2

    def _parse_seed_list(self, seeds):
        if seeds is None:
            return []
        if isinstance(seeds, dict):
            if "seeds" in seeds:
                seeds = seeds["seeds"]
            elif "genes" in seeds:
                seeds = seeds["genes"]
        if isinstance(seeds, np.ndarray):
            seeds = seeds.tolist()
        if isinstance(seeds, (list, tuple)):
            return [int(s) for s in seeds]
        raise TypeError("Unsupported seed container type.")

    def _parse_seed_pair(self, individual):
        # Handle Individual dataclass
        if hasattr(individual, "seeds_l1") and hasattr(individual, "seeds_l2"):
            return (
                self._parse_seed_list(individual.seeds_l1),
                self._parse_seed_list(individual.seeds_l2),
            )
        if isinstance(individual, dict):
            keys = [
                ("seeds_l1", "seeds_l2"),
                ("layer1", "layer2"),
                ("l1", "l2"),
                ("A", "B"),
            ]
            for k1, k2 in keys:
                if k1 in individual and k2 in individual:
                    return (
                        self._parse_seed_list(individual[k1]),
                        self._parse_seed_list(individual[k2]),
                    )
        if isinstance(individual, (list, tuple)) and len(individual) == 2:
            if isinstance(individual[0], (list, tuple, np.ndarray)) and isinstance(
                individual[1], (list, tuple, np.ndarray)
            ):
                return (
                    self._parse_seed_list(individual[0]),
                    self._parse_seed_list(individual[1]),
                )
        if isinstance(individual, (list, tuple, np.ndarray)):
            flat = self._parse_seed_list(individual)
            if self.budget_l1 is not None and self.budget_l2 is not None:
                if len(flat) >= self.budget_l1 + self.budget_l2:
                    return (
                        flat[: self.budget_l1],
                        flat[self.budget_l1 : self.budget_l1 + self.budget_l2],
                    )
        raise ValueError("Cannot parse a (seeds_l1, seeds_l2) pair from individual.")

    def evaluate_t1(self, individual, opponent_seeds=None, use_attention=False):
        seeds = self._parse_seed_list(individual)
        opponent = self._parse_seed_list(opponent_seeds)
        return MetricsCalculator.calculate_r_cs(
            seeds,
            opponent,
            self.graph_l1,
            alpha=self.alpha_l1 if use_attention else None,
            p=self.p,
            attack_steps=self.attack_steps,
            attack_ratio=self.attack_ratio,
            attack_mode=self.attack_mode,
        )

    def evaluate_t2(self, individual, opponent_seeds=None, use_attention=False):
        seeds = self._parse_seed_list(individual)
        opponent = self._parse_seed_list(opponent_seeds)
        return MetricsCalculator.calculate_r_cs(
            seeds,
            opponent,
            self.graph_l2,
            alpha=self.alpha_l2 if use_attention else None,
            p=self.p,
            attack_steps=self.attack_steps,
            attack_ratio=self.attack_ratio,
            attack_mode=self.attack_mode,
        )

    def evaluate_t3(self, individual, use_attention=False):
        seeds_l1, seeds_l2 = self._parse_seed_pair(individual)
        return MetricsCalculator.calculate_r_cr(
            seeds_l1,
            seeds_l2,
            self.graph_l1,
            self.graph_l2,
            alpha_l1=self.alpha_l1,
            alpha_l2=self.alpha_l2,
            p=self.p,
            attack_steps=self.attack_steps,
            attack_ratio=self.attack_ratio,
            attack_mode=self.attack_mode,
            use_attention=use_attention,
        )
