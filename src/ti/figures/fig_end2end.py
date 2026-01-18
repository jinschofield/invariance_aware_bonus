import os

import matplotlib.pyplot as plt
import torch

from ti.figures.helpers import get_env_spec
from ti.online.train import run_online_training
from ti.utils import ensure_dir


def _load_heatmap(path):
    payload = torch.load(path, map_location="cpu")
    return payload["heatmap"]


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], "online")
    ensure_dir(fig_dir)
    ensure_dir(table_dir)

    env_id = fig_spec.get("env", "periodicity")
    env_name = get_env_spec(cfg, env_id)["name"]
    method_a = fig_spec.get("method_a", "CRTR")
    method_b = fig_spec.get("method_b", "ICM")
    alpha = float(fig_spec.get("alpha", 1.0))
    seed = int(runtime["seed"])

    def heatmap_path(method):
        return os.path.join(table_dir, f"{env_id}_{method}_seed{seed}_alpha{alpha}_heatmap.pt")

    for method in (method_a, method_b):
        path = heatmap_path(method)
        if not os.path.exists(path):
            run_online_training(cfg, env_id, method, seed, alpha, table_dir)

    heat_a = _load_heatmap(heatmap_path(method_a))
    heat_b = _load_heatmap(heatmap_path(method_b))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(heat_a, cmap="viridis")
    axes[0].set_title(f"{method_a} visitation")
    axes[1].imshow(heat_b, cmap="viridis")
    axes[1].set_title(f"{method_b} visitation")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"End-to-end visitation ({env_name})")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
