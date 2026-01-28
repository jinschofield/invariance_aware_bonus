import os
from typing import Tuple

import torch

from ti.envs import layouts


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maze_cfg_from_config(cfg) -> dict:
    return {
        "maze_size": int(cfg.maze_size),
        "obs_dim": int(cfg.obs_dim),
        "n_actions": int(cfg.n_actions),
        "max_ep_steps": int(cfg.max_ep_steps),
        "goal": list(cfg.goal),
        "periodic_P": int(cfg.periodic_P),
    }


def free_positions(maze_size: int, device: torch.device) -> torch.Tensor:
    layout = layouts.make_layout(maze_size, device)
    return torch.nonzero(~layout).long()


def build_obs_from_pos_phase(
    pos_rc: torch.Tensor, phase: torch.Tensor, maze_size: int, period: int
) -> torch.Tensor:
    xy = layouts.pos_norm_from_grid(pos_rc, maze_size)
    ph = layouts.phase_sincos3(phase, period)
    return torch.cat([xy, ph], dim=-1)
