import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ti.figures.helpers import get_env_spec
from ti.utils import ensure_dir


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    results_path = os.path.join(runtime["table_dir"], "rep_sweep", "rep_sweep_results.csv")
    if not os.path.exists(results_path):
        from ti.figures import fig_rep_sweep

        fig_spec = {
            "envs": ["periodicity", "slippery", "teacup"],
            "crtr_rep_list": cfg["methods"]["crtr_rep_list"],
            "inv_samples": 2048,
        }
        fig_rep_sweep.run(cfg, "rep_sweep", fig_spec)

    df = pd.read_csv(results_path)
    env_ids = fig_spec.get("envs", list(cfg["envs"].keys()))
    env_names = [get_env_spec(cfg, e)["name"] for e in env_ids]
    order = [f"CRTR_R{r}" for r in methods_cfg["crtr_rep_list"]] + ["IDM", "ICM", "RND", "BISCUIT", "CBM"]

    fig, axes = plt.subplots(1, len(env_ids), figsize=(4 * len(env_ids), 3))
    if len(env_ids) == 1:
        axes = [axes]

    for i, env_name in enumerate(env_names):
        env_rows = df[df["env"] == env_name]
        env_rows["method"] = pd.Categorical(env_rows["method"], categories=order, ordered=True)
        env_rows = env_rows.sort_values("method")
        ax = axes[i]
        sns.barplot(ax=ax, x="method", y="xy_mse", data=env_rows, color="#4c72b0")
        ax.set_title(f"{env_name}: XY MSE (low)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("xy_mse")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
