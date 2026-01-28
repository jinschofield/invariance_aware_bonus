from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _mean_sem(values: torch.Tensor) -> Tuple[float, float]:
    arr = values.detach().cpu().numpy()
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / max(1, np.sqrt(arr.size)))
    return mean, sem


def plot_bar(
    values_by_rep: Dict[str, torch.Tensor],
    title: str,
    ylabel: str,
    out_path: str,
    p_values: Optional[Dict[str, float]] = None,
):
    labels = list(values_by_rep.keys())
    means, sems = [], []
    for key in labels:
        mean, sem = _mean_sem(values_by_rep[key])
        means.append(mean)
        sems.append(sem)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, yerr=sems, capsize=6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Representation")
    ax.grid(axis="y", alpha=0.3)

    if p_values:
        text = "p-values (t-test):\n" + "\n".join(
            [f"{k}: {v:.3e}" for k, v in p_values.items()]
        )
        ax.text(
            1.02,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=9,
            va="center",
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(
    heatmap: torch.Tensor,
    title: str,
    out_path: str,
    cmap: str = "viridis",
):
    data = heatmap.detach().cpu().numpy()
    mask = np.isnan(data)
    data = np.ma.masked_where(mask, data)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap_diff(
    heat_a: torch.Tensor,
    heat_b: torch.Tensor,
    title: str,
    out_path: str,
):
    diff = heat_a - heat_b
    data = diff.detach().cpu().numpy()
    mask = np.isnan(data)
    data = np.ma.masked_where(mask, data)
    vmax = np.nanmax(np.abs(data))
    vmax = float(vmax) if np.isfinite(vmax) else 1.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
