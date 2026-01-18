import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ti.online.train import run_online_training
from ti.utils import ensure_dir


def _auc_success(df):
    df = df.sort_values("env_step")
    steps = df["env_step"].values
    success = df["success"].values
    cum_success = np.cumsum(success) / np.arange(1, len(success) + 1)
    return np.trapz(cum_success, steps)


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    online_cfg = methods_cfg["online"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], "online")
    ensure_dir(fig_dir)
    ensure_dir(table_dir)

    env_id = fig_spec.get("env", "periodicity")
    method = fig_spec.get("method", "CRTR")
    alpha_list = fig_spec.get("alpha_list", online_cfg["alpha_list"])
    seeds = fig_spec.get("seeds", [int(runtime["seed"])])

    rows = []
    for alpha in alpha_list:
        aucs = []
        for seed in seeds:
            out_csv = os.path.join(table_dir, f"{env_id}_{method}_seed{seed}_alpha{alpha}.csv")
            if not os.path.exists(out_csv):
                run_online_training(cfg, env_id, method, seed, alpha, table_dir)
            df = pd.read_csv(out_csv)
            aucs.append(_auc_success(df))
        rows.append({"alpha": float(alpha), "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(df["alpha"], df["auc_mean"], marker="o")
    ax.fill_between(df["alpha"], df["auc_mean"] - df["auc_std"], df["auc_mean"] + df["auc_std"], alpha=0.2)
    ax.set_xlabel("alpha")
    ax.set_ylabel("AUC (success)")
    ax.set_title(f"Hyperparameter sweep ({env_id})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
