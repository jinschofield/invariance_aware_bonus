# Temporal Invariance (ICML figures)

This repo generates all figure assets for the ICML draft using the Periodicity, Slippery, and Teacup mazes (rotation maze removed).

## Quick start (local)
```
pip install -r requirements.txt
python scripts/run_figures.py --config configs/paper.yaml
```

## Colab / multi‑GPU
Use the split notebooks to run on 3 GPUs in parallel (each mounts Google Drive and writes outputs there):
- `notebooks/01_rep_figures.ipynb` (GPU 0)
- `notebooks/02_bonus_figures.ipynb` (GPU 1)
- `notebooks/03_online_figures.ipynb` (GPU 2)

Each notebook calls `scripts/run_figures.py` with a Drive‑based config:
- `configs/colab_drive_rep.yaml`
- `configs/colab_drive_bonus.yaml`
- `configs/colab_drive_online.yaml`

## Outputs
- Figures: `outputs/figures/<run_id>/`
- Tables: `outputs/tables/<run_id>/`
- Runs: `outputs/runs/<run_id>/` (checkpoints + logs + config snapshot)
- Cache: `outputs/cache/` (datasets and episode indices)

## Configs
- `configs/paper.yaml` selects envs, methods, and figure lineup
- `configs/envs.yaml` defines the three environments (Periodicity, Slippery, Teacup)
- `configs/methods.yaml` defines training/probe/elliptical settings
- `configs/figures.yaml` controls figure generation order
- `configs/figures_rep.yaml`, `configs/figures_bonus.yaml`, `configs/figures_online.yaml` split the figure lineup

## Caching and checkpoints
- Datasets are cached under `outputs/cache/` when `runtime.cache_datasets=true`.
- Training losses are saved under `outputs/runs/<run_id>/logs/`.
- Checkpoints are saved under `outputs/runs/<run_id>/checkpoints/`.

## Throughput profiles
- `runtime.throughput_mode` controls when max‑throughput settings apply:
  - `rl_only`: only for figures tagged with `profile: rl`
  - `always`: always apply RL profile
  - `never`: never apply RL profile
- Use `runtime.rl_profile` to define the max‑throughput settings.

## Optuna (hyperparameter search)
```
python scripts/optuna_search.py --config configs/paper.yaml --optuna-config configs/optuna.yaml
```

## Online RL
```
python scripts/run_online_rl.py --config configs/paper.yaml --env periodicity --method CRTR --alpha 1.0 --seed 0
```
