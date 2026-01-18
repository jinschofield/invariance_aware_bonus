import os

import matplotlib.pyplot as plt
import pandas as pd

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
    metric = fig_spec.get("metric", "inv")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for env_id in env_ids:
        env_name = get_env_spec(cfg, env_id)["name"]
        env_rows = df[df["env"] == env_name].copy()
        env_rows = env_rows[env_rows["method"].str.startswith("CRTR_R")]
        env_rows["R"] = env_rows["method"].str.replace("CRTR_R", "").astype(int)
        env_rows = env_rows.sort_values("R")
        ax.plot(env_rows["R"], env_rows[metric], marker="o", label=env_name)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Repetition factor R")
    ax.set_ylabel(metric)
    ax.set_title(f"CRTR repetition sweep ({metric})")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
