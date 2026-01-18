import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ti.utils import ensure_dir


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    scalars_path = os.path.join(runtime["table_dir"], "elliptical_heatmaps", "elliptical_scalars.csv")
    if not os.path.exists(scalars_path):
        from ti.figures import fig_elliptical_heatmaps

        fig_spec = {
            "envs": ["periodicity", "slippery", "teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
        }
        fig_elliptical_heatmaps.run(cfg, "elliptical_heatmaps", fig_spec)

    df = pd.read_csv(scalars_path)
    envs = fig_spec.get("envs", ["Periodicity", "Slippery", "Teacup Maze"])

    fig, axes = plt.subplots(2, len(envs), figsize=(4 * len(envs), 6))
    if len(envs) == 1:
        axes = [axes]

    for i, env_name in enumerate(envs):
        rows = df[df["env"] == env_name]
        ax = axes[0][i] if len(envs) > 1 else axes[0]
        sns.barplot(ax=ax, x="method", y="orbit_ratio_W_over_B", data=rows, color="#4c72b0")
        ax.set_title(f"{env_name}: Orbit ratio (low)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")

    teacup_rows = df[df["env"] == "Teacup Maze"]
    if not teacup_rows.empty:
        ax1 = axes[1][0] if len(envs) > 1 else axes[1]
        sns.barplot(ax=ax1, x="method", y="cup_contrast_in_minus_out", data=teacup_rows, color="#55a868")
        ax1.set_title("Teacup: cup contrast (high)")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_xlabel("")

        ax2 = axes[1][1] if len(envs) > 1 else axes[1]
        sns.barplot(ax=ax2, x="method", y="inside_std_mean", data=teacup_rows, color="#c44e52")
        ax2.set_title("Teacup: inside std (low)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.set_xlabel("")
        if len(envs) > 2:
            axes[1][2].axis("off")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
