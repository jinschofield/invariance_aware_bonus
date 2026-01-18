import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ti.utils import ensure_dir
from ti.online.train import run_online_training


def _auc_success(df):
    df = df.sort_values("env_step")
    steps = df["env_step"].values
    success = df["success"].values
    cum_success = np.cumsum(success) / np.arange(1, len(success) + 1)
    return np.trapz(cum_success, steps)


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    table_dir = os.path.join(runtime["table_dir"], "online")
    bonus_path = os.path.join(runtime["table_dir"], "elliptical_heatmaps", "elliptical_scalars.csv")
    if not os.path.exists(bonus_path):
        raise FileNotFoundError(f"Missing elliptical_scalars.csv at {bonus_path}")

    if not os.path.exists(bonus_path):
        from ti.figures import fig_elliptical_heatmaps

        fig_spec = {
            "envs": ["periodicity", "slippery", "teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
        }
        fig_elliptical_heatmaps.run(cfg, "elliptical_heatmaps", fig_spec)

    bonus_df = pd.read_csv(bonus_path)
    metric = fig_spec.get("metric", "orbit_ratio_W_over_B")

    rows = []
    for fname in os.listdir(table_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(table_dir, fname))
        if df.empty:
            continue
        env = df["env"].iloc[0]
        method = df["method"].iloc[0]
        seed = df["seed"].iloc[0]
        auc = _auc_success(df)
        rows.append({"env": env, "method": method, "seed": seed, "auc": auc})

    if not rows:
        # Attempt a minimal run with default methods if nothing exists
        for method in ["CRTR", "ICM", "RND"]:
            run_online_training(cfg, "periodicity", method, runtime["seed"], 1.0, table_dir)
        for fname in os.listdir(table_dir):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(table_dir, fname))
            if df.empty:
                continue
            env = df["env"].iloc[0]
            method = df["method"].iloc[0]
            seed = df["seed"].iloc[0]
            auc = _auc_success(df)
            rows.append({"env": env, "method": method, "seed": seed, "auc": auc})
        if not rows:
            raise RuntimeError("No online results found for Spearman figure.")

    online_df = pd.DataFrame(rows)
    merged = online_df.merge(bonus_df, on=["env", "method"])
    rho, _ = spearmanr(merged[metric], merged["auc"])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.scatter(merged[metric], merged["auc"], s=25, alpha=0.8)
    ax.set_title(f"Spearman rho={rho:.2f}")
    ax.set_xlabel(metric)
    ax.set_ylabel("AUC (success)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
