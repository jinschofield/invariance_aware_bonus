import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


def make_bottleneck_layout(maze_size: int, device: torch.device) -> torch.Tensor:
    layout = np.ones((maze_size, maze_size), dtype=bool)
    layout[1:-1, 1:-1] = False

    # Add vertical walls with single-cell gaps.
    v_cols = list(range(3, maze_size - 3, 4))
    for i, c in enumerate(v_cols):
        layout[1:-1, c] = True
        gap = 1 + ((i * 7) % (maze_size - 2))
        layout[gap, c] = False

    # Add horizontal walls with single-cell gaps.
    h_rows = list(range(3, maze_size - 3, 4))
    for j, r in enumerate(h_rows):
        layout[r, 1:-1] = True
        gap = 1 + ((j * 5) % (maze_size - 2))
        layout[r, gap] = False

    # Ensure common start/goal cells are free.
    layout[1, 1] = False
    layout[maze_size - 2, maze_size - 2] = False

    def _components(mask: np.ndarray):
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r, c] and not visited[r, c]:
                    q = deque([(r, c)])
                    visited[r, c] = True
                    comp = [(r, c)]
                    while q:
                        x, y = q.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                                if mask[nx, ny] and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    q.append((nx, ny))
                                    comp.append((nx, ny))
                    comps.append(comp)
        return comps

    def _carve_path(a, b):
        ar, ac = a
        br, bc = b
        for r in range(min(ar, br), max(ar, br) + 1):
            layout[r, ac] = False
        for c in range(min(ac, bc), max(ac, bc) + 1):
            layout[br, c] = False

    free = ~layout
    comps = _components(free)
    while len(comps) > 1:
        comps.sort(key=len, reverse=True)
        a = comps[0][0]
        b = comps[1][0]
        _carve_path(a, b)
        free = ~layout
        comps = _components(free)

    return torch.tensor(layout, device=device, dtype=torch.bool)


def make_layout(maze_size: int, device: torch.device) -> torch.Tensor:
    if maze_size != 12:
        return make_bottleneck_layout(maze_size, device)

    layout = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.bool,
    )
    return layout


def make_open_plate_layout(maze_size: int, device: torch.device) -> torch.Tensor:
    layout = torch.ones((maze_size, maze_size), device=device, dtype=torch.bool)
    layout[1:-1, 1:-1] = False
    return layout


def pos_norm_from_grid(pos_xy_long: torch.Tensor, maze_size: int) -> torch.Tensor:
    return (pos_xy_long.float() / float(maze_size - 1)) * 2.0 - 1.0


def step_pos_with_layout(
    pos_xy: torch.Tensor,
    action: torch.Tensor,
    layout_bool: torch.Tensor,
    maze_size: int,
    n_actions: int,
) -> torch.Tensor:
    deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=pos_xy.device, dtype=torch.long)
    a = action.long().clamp(0, n_actions - 1)
    delta = deltas[a]
    nxt = pos_xy + delta
    nxt[:, 0] = nxt[:, 0].clamp(0, maze_size - 1)
    nxt[:, 1] = nxt[:, 1].clamp(0, maze_size - 1)
    blocked = layout_bool[nxt[:, 0], nxt[:, 1]]
    return torch.where(blocked.unsqueeze(1), pos_xy, nxt)


TETRA4 = (torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
) / math.sqrt(3.0)).float()


def one_hot3(k: torch.Tensor) -> torch.Tensor:
    return F.one_hot(k.long(), num_classes=3).float()


def phase_sincos3(k: torch.Tensor, period: int) -> torch.Tensor:
    ang = (2.0 * math.pi / float(period)) * k.float()
    return torch.stack([torch.sin(ang), torch.cos(ang), torch.sin(2.0 * ang)], dim=-1)
