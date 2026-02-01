from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StudyConfig:
    seed: int = 0
    device: Optional[str] = None
    output_dir: str = "periodicity_study/outputs"
    require_cuda: bool = True

    maze_size: int = 12
    maze_size_large: int = 32
    obs_dim: int = 5
    n_actions: int = 4
    max_ep_steps: int = 60
    periodic_P: int = 8
    slippery_D: int = 3
    teacup_P: int = 4
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

    online_eval_every_updates: int = 5
    online_eval_min_buffer: int = 512
    online_eval_buffer_size: int = 1000000
    coverage_threshold: float = 0.99

    online_rep_update_every: int = 2048
    online_rep_update_steps: int = 1
    online_rep_batch_size: int = 256
    online_rep_warmup_steps: int = 4096
    online_rep_buffer_size: int = 0

    bonus_gif_env_id: str = "periodicity"
    bonus_gif_rep: str = "crtr_learned"
    bonus_gif_every_updates: int = 2
    bonus_gif_max_frames: int = 50
    bonus_gif_fps: int = 3

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
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_alpha0: float = 1.0
    ppo_alpha_eta: float = 1.0
    ppo_alpha_rho: float = 0.05
    ppo_alpha_zero_after_hit: bool = False
    ppo_use_alpha_anneal: bool = False
    ppo_use_alpha_gate: bool = False
    ppo_use_two_critic: bool = False
    ppo_use_int_norm: bool = False
    ppo_int_clip: float = 0.0
    ppo_int_norm_eps: float = 1e-8
    bonus_gif_policy_input: str = "rep"