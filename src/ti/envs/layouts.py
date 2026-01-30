import math

import torch
import torch.nn.functional as F


def make_layout(maze_size: int, device: torch.device) -> torch.Tensor:
    if maze_size != 12:
        return make_open_plate_layout(maze_size, device)

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
