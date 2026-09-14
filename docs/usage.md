# Usage guide

[Return to the research overview](../README.md)

## Environment

```bash
git clone https://github.com/IamJerryXu/DRCIM-ML.git
cd DRCIM-ML

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install node2vec
```

Two additional dependencies must be supplied before running the pipeline:

- **`torch-scatter`**, installed for the chosen PyTorch and CUDA versions. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
- **`ikan.kat_1dgroup_torch.KAT_Group_Torch`**, imported by the encoder. This external module must be available in the Python environment; it is not bundled in the repository.

Set `project.device` in [config.yaml](../config.yaml) to an available device. The checked-in value is `cuda:6`; change it to `cuda:0` or `cpu` as appropriate. The command-line interface has no `--device` option.

## Network input

Place network files in `dataset/train/`. Each text file contains layers marked by `layer:`, followed by undirected, unweighted edges. Use consistent node identifiers across layers and declare the node count in each layer header.

```text
layer:1 nodes=4 edges=3
0 1
1 2
2 3
layer:2 nodes=4 edges=3
0 2
0 3
1 3
```

This example illustrates the file format. Prepare actual network data before running the pipeline. The evolution entry point uses the first loaded network and its first two layers. An empty training directory can trigger a zero-epoch synthetic fallback and should not be treated as a successful training run.

## Training and search

After preparing the data and dependencies, configure the seed budgets, population size, generation count, and attack settings in [config.yaml](../config.yaml).

```bash
# Pretraining followed by evolutionary search
python -m code.run_gma_mfea --config config.yaml --mode all

# Pretraining only
python -m code.run_gma_mfea --config config.yaml --mode pretrain

# Evolution using previously generated alignment outputs
python -m code.run_gma_mfea --config config.yaml --mode evolution
```

To load locally trained encoder weights:

```bash
python -m code.run_gma_mfea --config config.yaml \
  --mode load_checkpoint --checkpoint kaa_grit_v2
```

`kaa_grit_v2` is the configured filename prefix, not a downloadable checkpoint. The corresponding files must already exist in `checkpoints/`.

## Saved outputs

- `checkpoints/<prefix>_encoder_a.pt` and `_encoder_b.pt` store the encoder weights.
- `data/alignment/similarity_matrix/S_align.npy` stores cross-layer similarity.
- `data/alignment/` also contains layer embeddings and `alpha_l1.npy` / `alpha_l2.npy`.
- `results/best_t1_<timestamp>.json`, `best_t3_<timestamp>.json`, and `history_<timestamp>.json` store the exported search results and history. There is no separate T2 best-seed export in this entry point.

## Code guide

| Component | Source |
| :--- | :--- |
| Pipeline and command-line modes | [code/run_gma_mfea.py](../code/run_gma_mfea.py) |
| Representation learning and alignment | [code/gma_core/](../code/gma_core/) |
| Evolutionary search and local refinement | [code/mfea_core/](../code/mfea_core/) |
| Competitive influence evaluation | [code/evaluation/](../code/evaluation/) |
| Network loading and preprocessing | [code/utils/data_loader.py](../code/utils/data_loader.py), [code/dataset/](../code/dataset/) |
| Synthetic network generation | [network_generation/](../network_generation/) |

## Use and permissions

The project is shared for academic research. No repository-wide license has been specified; public access alone does not grant unrestricted reuse or redistribution. Contact the maintainers for permissions. Dependencies remain subject to their respective licenses.
