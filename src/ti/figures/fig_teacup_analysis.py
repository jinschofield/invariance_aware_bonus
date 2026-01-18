import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ti.utils import ensure_dir


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    bonus_path = os.path.join(runtime["table_dir"], "elliptical_heatmaps", "elliptical_scalars.csv")
    if not os.path.exists(bonus_path):
        from ti.figures import fig_elliptical_heatmaps

        fig_spec = {
            "envs": ["teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
        }
        fig_elliptical_heatmaps.run(cfg, "elliptical_heatmaps", fig_spec)

    df = pd.read_csv(bonus_path)
    teacup = df[df["env"] == "Teacup Maze"]
    if teacup.empty:
        raise RuntimeError("No Teacup Maze rows found in elliptical scalars.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    sns.barplot(ax=axes[0], x="method", y="cup_contrast_in_minus_out", data=teacup, color="#55a868")
    axes[0].set_title("Cup contrast (high)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_xlabel("")

    sns.barplot(ax=axes[1], x="method", y="inside_std_mean", data=teacup, color="#c44e52")
    axes[1].set_title("Inside std (low)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel("")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
