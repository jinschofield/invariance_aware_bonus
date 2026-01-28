from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StudyConfig:
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = "periodicity_study/outputs"
    require_cuda: bool = True

    maze_size: int = 12
    obs_dim: int = 5
    n_actions: int = 4
    max_ep_steps: int = 60
    periodic_P: int = 8
    goal: Tuple[int, int] = (10, 10)

    offline_num_envs: int = 256
    offline_collect_steps: int = 40000
    offline_train_steps: int = 20000
    offline_batch_size: int = 256
    print_train_every: int = 5000

    z_dim: int = 8
    hidden_dim: int = 64
    lr: float = 3e-4
    crtr_temp: float = 2.8284271247461903
    crtr_rep_factor: int = 8
    k_cap: int = 10
    geom_p: float = 0.01

    rep_positions: int = 200
    rep_pairs_per_pos: int = 32

    bonus_lambda: float = 1.0
    bonus_beta: float = 1.0
    bonus_action_avg: bool = True
    bonus_nuis_samples: int = 16
    bonus_max_samples: int = 200000

    ppo_num_envs: int = 32
    ppo_total_steps: int = 200000
    ppo_steps_per_update: int = 128
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 1024
    ppo_hidden_dim: int = 128
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_coef: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
