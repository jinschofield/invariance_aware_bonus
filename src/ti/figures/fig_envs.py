import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ti.envs import layouts
from ti.figures.helpers import build_maze_cfg
from ti.utils import ensure_dir


def _plot_layout(ax, layout, goal, title):
    grid = np.ones_like(layout.cpu().numpy(), dtype=float)
    grid[layout.cpu().numpy().astype(bool)] = 0.0
    ax.imshow(grid, cmap="gray_r")
    ax.scatter([goal[1]], [goal[0]], c="red", s=30, marker="*", label="Goal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    maze_cfg = build_maze_cfg(cfg)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    layout = layouts.make_layout(maze_cfg["maze_size"], device)
    plate = layouts.make_open_plate_layout(maze_cfg["maze_size"], device)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    _plot_layout(axes[0], layout, maze_cfg["goal"], "Periodicity")
    _plot_layout(axes[1], layout, maze_cfg["goal"], "Slippery")
    _plot_layout(axes[2], plate, maze_cfg["goal"], "Teacup Maze")

    # Teacup cups
    centers = np.array([[3, 3], [3, 8], [8, 3], [8, 8], [6, 6]])
    for cx, cy in centers:
        circle = plt.Circle((cy, cx), radius=1.5, edgecolor="cyan", facecolor="none", lw=1.0)
        axes[2].add_patch(circle)

    fig.suptitle("Environments")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
