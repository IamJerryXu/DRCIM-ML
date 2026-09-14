<a id="top"></a>

<div align="center">

<picture>
<source media="(max-width: 600px)" srcset="assets/readme/hero-mobile.svg">
<img src="assets/readme/hero.svg" width="100%" alt="DRCIM-ML — Solving the Robust Influence Maximization Problem in Competitive Multilayer Networks via a Diffusion-Aware Role-Guided Evolutionary Approach">
</picture>

<p>
<a href="#overview"><img src="assets/readme/nav-overview.svg" height="24" alt="Overview"></a>
<a href="#method"><img src="assets/readme/nav-method.svg" height="24" alt="Method"></a>
<a href="#implementation"><img src="assets/readme/nav-implementation.svg" height="24" alt="Implementation"></a>
<a href="#citation"><img src="assets/readme/nav-citation.svg" height="24" alt="Citation"></a>
</p>

</div>

<a id="overview"></a>

<h2><img src="assets/readme/heading-overview.svg" height="36" alt="Research overview"></h2>

**DRCIM-ML** is a diffusion-aware role-guided evolutionary approach to robust competitive influence maximization in multilayer networks. It selects seed sets that sustain influence under competing cascades and progressive structural damage.

A node can be central in one layer and peripheral in another. Transferring its identity therefore need not preserve its diffusion role. DRCIM-ML combines diffusion-aware representation learning with evolutionary multitasking to propose candidates by role and evaluate them under competition and node removal.

<p align="center"><a href="assets/figures/motivation.pdf"><img src="assets/figures/motivation.png" width="100%" alt="Cross-layer transfer proposals: source node 3 maps to alternative target candidates 3 or 7; removing target node 6 isolates node 3."></a></p>

<a id="method"></a>

<h2><img src="assets/readme/heading-method.svg" height="36" alt="Method overview"></h2>

<p align="center"><a href="assets/figures/framework.pdf"><img src="assets/figures/framework.png" width="100%" alt="DRCIM-ML framework: diffusion-aware role learning, role-guided multifactorial search, and robust evaluation."></a></p>

**Diffusion-aware role learning.** The Diffusion-Aware Transformer (DAT) encodes graph structure using relative random-walk probabilities (RRWP). Reconstruction, attention-aware distribution alignment, and role constraints support layer-wise importance priors and cross-layer role similarity.

**Role-guided evolutionary search.** Related seed-selection tasks exchange candidates through a shared population. Importance priors guide sampling, while role correspondence guides cross-layer transfer. Population feedback updates the guidance, and a reinforcement-learning controller selects local-refinement actions.

**Robust evaluation.** Layer-wise competitive robustness, <i>R</i><sub>CS</sub>, and collaborative robustness, <i>R</i><sub>CR</sub>, are evaluated over successive node removals. Candidate quality depends on the complete seed pair and the damage trajectory.

<a id="dat-architecture"></a>

<h2><img src="assets/readme/heading-dat.svg" height="36" alt="DAT architecture"></h2>

DAT provides the node representations used for role-guided search. The figure expands one attention block from the stacked encoder, showing how node features and pairwise RRWP representations enter the attention computation.

<p align="center"><a href="assets/figures/dat.pdf"><img src="assets/figures/dat.png" width="820" alt="DAT architecture, with the stacked encoder on the left and expanded RRWP-conditioned attention on the right."></a></p>

<a id="implementation"></a>

<h2><img src="assets/readme/heading-implementation.svg" height="36" alt="Implementation"></h2>

The implementation combines representation pretraining with role-guided evolutionary search. Follow the [usage guide](docs/usage.md) to install the dependencies and prepare multilayer network data, then set the seed budgets, population size, and attack settings in [config.yaml](config.yaml).

Run both stages with

```bash
python -m code.run_gma_mfea --config config.yaml --mode all
```

Separate pretraining and search modes, checkpoint loading, and saved outputs are described in the guide.

<a id="citation"></a>

<h2><img src="assets/readme/heading-citation.svg" height="36" alt="Citation"></h2>

If this work supports your research, please consider citing the manuscript.

```bibtex
@unpublished{xu2026drcimml,
  title  = {Solving the Robust Influence Maximization Problem in Competitive
            Multilayer Networks via a Diffusion-Aware Role-Guided
            Evolutionary Approach},
  author = {Xu, Yongxue and Liu, Ziqian and Huang, Yongqing and Wang, Shuai},
  year   = {2026},
  note   = {Unpublished manuscript},
  url    = {https://github.com/IamJerryXu/DRCIM-ML}
}
```

---

<p align="center"><em>We welcome discussion and collaboration on network diffusion and evolutionary optimization.</em></p>

<p align="center"><a href="https://jerrysnow.me">Contact</a> · <a href="https://github.com/IamJerryXu/DRCIM-ML/issues">Questions and discussion</a></p>
