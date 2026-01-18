import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ti.utils import ensure_dir
from ti.online.train import run_online_training


def _load_runs(table_dir, env_id, method, alpha):
    rows = []
    for fname in os.listdir(table_dir):
        if fname.startswith(f"{env_id}_{method}_") and f"alpha{alpha}" in fname and fname.endswith(".csv"):
            rows.append(pd.read_csv(os.path.join(table_dir, fname)))
    return rows


def _success_curve(df, step_grid):
    df = df.sort_values("env_step")
    steps = df["env_step"].values
    success = df["success"].values
    cum_success = np.cumsum(success) / np.arange(1, len(success) + 1)
    out = np.interp(step_grid, steps, cum_success, left=cum_success[0], right=cum_success[-1])
    return out


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    online_cfg = methods_cfg["online"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], "online")
    ensure_dir(fig_dir)
    ensure_dir(table_dir)

    env_id = fig_spec.get("env", "periodicity")
    methods = fig_spec.get("methods", ["CRTR", "ICM", "RND"])
    alpha = float(fig_spec.get("alpha", 1.0))
    max_steps = int(online_cfg["total_steps"])
    step_grid = np.linspace(0, max_steps, 200)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for method in methods:
        runs = _load_runs(table_dir, env_id, method, alpha)
        if not runs:
            run_online_training(cfg, env_id, method, runtime["seed"], alpha, table_dir)
            runs = _load_runs(table_dir, env_id, method, alpha)
        if not runs:
            continue
        curves = np.stack([_success_curve(df, step_grid) for df in runs], axis=0)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(step_grid, mean, label=method)
        ax.fill_between(step_grid, mean - std, mean + std, alpha=0.2)

    ax.set_title(f"Convergence ({env_id})")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Success rate")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
