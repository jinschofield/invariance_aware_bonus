# Periodicity Study (Standalone Subfolder)

This subfolder contains a standalone study that runs across multiple environments
(periodicity, delay action queue, teacup) and large-maze variants. It implements
three stages of measurement across three representation types:

1) Representation invariance (same position, different nuisance).
2) Elliptical bonus behavior (fixed position, across nuisance).
3) PPO action distributions (bonus-only reward).

Representations evaluated:
- Coord-only (normalized x,y).
- Coord + nuisance (x,y + phase index).
- CRTR learned representation (offline-trained).
- IDM learned representation (offline-trained).
- CRTR learned representation (online joint training with PPO + bonus).
- IDM learned representation (online joint training with PPO + bonus).

## Quick start

```bash
python -m periodicity_study.run_study
```

### Selecting environments
You can filter which environments run:

```bash
# Only large mazes
python -m periodicity_study.run_study --only-large

# Only base (small) mazes
python -m periodicity_study.run_study --only-small

# Specific env ids (comma-separated)
python -m periodicity_study.run_study --envs periodicity_large
python -m periodicity_study.run_study --envs periodicity,slippery
```

### Selecting reps and reward mode (PPO only)
You can run a subset of representation PPO runs and choose reward mode:

```bash
# Only goal (extrinsic) PPO for learned reps
python -m periodicity_study.run_study --goal-only --reps crtr_learned,idm_learned,crtr_online_joint,idm_online_joint

# Only intrinsic PPO runs
python -m periodicity_study.run_study --intrinsic-only
```

### PPO policy input
By default PPO consumes representation embeddings. To use raw observations instead:

```bash
python -m periodicity_study.run_study --policy-input raw
```

This study is GPU-first and will error if CUDA is not available. Use `--device cuda:0`
to select a specific GPU.

## SLURM

```bash
sbatch periodicity_study/slurm/periodicity_study.sbatch
sbatch periodicity_study/slurm/periodicity_study_fast.sbatch
```

## Colab notebook

Notebook path: `periodicity_study/notebooks/periodicity_study_colab.ipynb`

Outputs (figures + CSVs) are written under:
`periodicity_study/outputs/`, organized by environment (subfolders per env).

## Online training logs

During PPO training, metrics for levels (1), (2), and (3) are logged over time:

- `periodicity_study/outputs/logs/metrics_timeseries_<rep>.csv`
- `periodicity_study/outputs/logs/metrics_timeseries_crtr_online_joint.csv`
