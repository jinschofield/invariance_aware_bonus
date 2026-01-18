import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from ti.utils import ensure_dir


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    rep_path = os.path.join(runtime["table_dir"], "rep_sweep", "rep_sweep_results.csv")
    bonus_path = os.path.join(runtime["table_dir"], "elliptical_heatmaps", "elliptical_scalars.csv")
    if not os.path.exists(rep_path):
        from ti.figures import fig_rep_sweep

        fig_spec = {
            "envs": ["periodicity", "slippery", "teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
            "inv_samples": 2048,
        }
        fig_rep_sweep.run(cfg, "rep_sweep", fig_spec)

    if not os.path.exists(bonus_path):
        from ti.figures import fig_elliptical_heatmaps

        fig_spec = {
            "envs": ["periodicity", "slippery", "teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
        }
        fig_elliptical_heatmaps.run(cfg, "elliptical_heatmaps", fig_spec)

    rep_df = pd.read_csv(rep_path)
    bonus_df = pd.read_csv(bonus_path)

    metric_x = fig_spec.get("metric_x", "inv")
    metric_y = fig_spec.get("metric_y", "orbit_ratio_W_over_B")

    envs = sorted(rep_df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 3))
    if len(envs) == 1:
        axes = [axes]

    for i, env_name in enumerate(envs):
        rep_env = rep_df[rep_df["env"] == env_name]
        bonus_env = bonus_df[bonus_df["env"] == env_name]
        merged = rep_env.merge(bonus_env, on=["env", "method"])
        ax = axes[i]
        ax.scatter(merged[metric_x], merged[metric_y], s=20, alpha=0.8)
        rho, _ = spearmanr(merged[metric_x], merged[metric_y])
        ax.set_title(f"{env_name} (rho={rho:.2f})")
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
