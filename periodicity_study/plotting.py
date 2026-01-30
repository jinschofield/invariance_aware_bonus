from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _mean_sem(values: torch.Tensor) -> Tuple[float, float]:
    arr = values.detach().cpu().numpy()
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / max(1, np.sqrt(arr.size)))
    return mean, sem


def _format_x_ticks(ax: plt.Axes, rotation: int = 30) -> None:
    ax.tick_params(axis="x", labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_ha("right")


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
    _format_x_ticks(ax)

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


def plot_bar_values(
    values_by_label: Dict[str, float],
    title: str,
    ylabel: str,
    out_path: str,
):
    labels = list(values_by_label.keys())
    values = [float(values_by_label[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Comparison")
    ax.grid(axis="y", alpha=0.3)
    _format_x_ticks(ax)
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _format_x_ticks(ax)
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _format_x_ticks(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timeseries(
    rows: Iterable[Dict[str, float]],
    title: str,
    out_path: str,
    metrics: Dict[str, str],
    x_key: str = "env_steps",
):
    rows = list(rows)
    if not rows:
        return
    x = [float(row.get(x_key, row.get("update", 0))) for row in rows]
    n = len(metrics)
    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (key, label) in zip(axes, metrics.items()):
        y = [float(row.get(key, float("nan"))) for row in rows]
        ax.plot(x, y, marker="o", markersize=3)
        ax.set_title(label)
        ax.set_xlabel(x_key)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        _format_x_ticks(ax)

    # Hide any unused axes
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes:
        ax.label_outer()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_multi_timeseries(
    series_by_rep: Dict[str, Iterable[Dict[str, float]]],
    title: str,
    out_path: str,
    y_key: str,
    y_label: str,
    x_key: str = "env_steps",
    x_label: Optional[str] = None,
    hline_y: Optional[float] = 1.0,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in series_by_rep.items():
        rows = list(rows)
        if not rows:
            continue
        x = [float(row.get(x_key, row.get("update", 0))) for row in rows]
        y = [float(row.get(y_key, float("nan"))) for row in rows]
        ax.plot(x, y, marker="o", markersize=3, label=name)
    ax.set_title(title)
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label)
    if hline_y is not None:
        ax.axhline(hline_y, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
