# Periodicity Study (Standalone Subfolder)

This subfolder contains a standalone study focused exclusively on the Periodicity
environment. It implements three stages of measurement across three
representation types:

1) Representation invariance (same position, different nuisance).
2) Elliptical bonus behavior (fixed position, across nuisance).
3) PPO action distributions (bonus-only reward).

Representations evaluated:
- Coord-only (normalized x,y).
- Coord + nuisance (x,y + phase index).
- CRTR learned representation (offline-trained).
- CRTR learned representation (online joint training with PPO + bonus).

## Quick start

```bash
python -m periodicity_study.run_study
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
`periodicity_study/outputs/`

## Online training logs

During PPO training, metrics for levels (1), (2), and (3) are logged over time:

- `periodicity_study/outputs/logs/metrics_timeseries_<rep>.csv`
- `periodicity_study/outputs/logs/metrics_timeseries_crtr_online_joint.csv`
