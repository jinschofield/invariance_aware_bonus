from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.utils import seed_everything



class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int, two_critic: bool = False):
        super().__init__()
        self.two_critic = bool(two_critic)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        if self.two_critic:
            self.value_head_ext = nn.Linear(hidden_dim, 1)
            self.value_head_int = nn.Linear(hidden_dim, 1)
        else:
            self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.policy_head(h)
        if self.two_critic:
            value_ext = self.value_head_ext(h).squeeze(-1)
            value_int = self.value_head_int(h).squeeze(-1)
            return logits, value_ext, value_int
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        logits = out[0]
        return torch.softmax(logits, dim=-1)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


def _obs_to_pos(obs: torch.Tensor, maze_size: int) -> torch.Tensor:
    xy = obs[:, :2]
    pos = torch.round(((xy + 1.0) * 0.5) * float(maze_size - 1)).long()
    return pos.clamp(0, maze_size - 1)


def _update_running_stats(count: int, mean: float, m2: float, x: torch.Tensor) -> Tuple[int, float, float]:
    if x.numel() == 0:
        return count, mean, m2
    x = x.detach().float()
    batch_count = int(x.numel())
    batch_mean = float(x.mean().item())
    batch_m2 = float(((x - batch_mean) ** 2).sum().item())
    if count == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean
    new_count = count + batch_count
    new_mean = mean + delta * batch_count / new_count
    new_m2 = m2 + batch_m2 + delta * delta * count * batch_count / new_count
    return new_count, new_mean, new_m2


def _compute_gae(rewards, values, dones, next_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def train_ppo(
    rep,
    cfg,
    device: torch.device,
    env_ctor,
    maze_cfg,
    policy_obs_fn=None,
    policy_input_dim: Optional[int] = None,
    use_extrinsic: bool = False,
    rep_updater=None,
    rep_buffer=None,
    rep_update_every: Optional[int] = None,
    rep_update_steps: int = 1,
    rep_batch_size: Optional[int] = None,
    rep_warmup_steps: int = 0,
    eval_callback=None,
    eval_every_updates: Optional[int] = None,
    eval_buffer=None,
) -> Tuple[ActorCritic, List[Dict[str, float]], List[Dict[str, float]]]:
    seed_everything(int(cfg.seed), deterministic=True)
    env = env_ctor(
        num_envs=cfg.ppo_num_envs,
        maze_size=maze_cfg["maze_size"],
        max_ep_steps=maze_cfg["max_ep_steps"],
        n_actions=maze_cfg["n_actions"],
        goal=maze_cfg["goal"],
        device=device,
    )

    if policy_input_dim is None:
        policy_input_dim = int(rep.dim)
    use_two_critic = bool(getattr(cfg, "ppo_use_two_critic", False))
    if use_extrinsic and use_two_critic:
        use_two_critic = False
    model = ActorCritic(
        policy_input_dim, cfg.n_actions, cfg.ppo_hidden_dim, two_critic=use_two_critic
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo_lr)

    bonus = EpisodicEllipticalBonus(
        z_dim=rep.dim,
        n_actions=cfg.n_actions,
        beta=cfg.bonus_beta,
        lam=cfg.bonus_lambda,
        num_envs=cfg.ppo_num_envs,
        device=device,
    )

    steps_per_update = int(cfg.ppo_steps_per_update)
    total_steps = int(cfg.ppo_total_steps)
    batch_size = steps_per_update * cfg.ppo_num_envs
    updates = max(1, total_steps // batch_size)

    obs = env.reset()
    logs: List[Dict[str, float]] = []
    metrics_log: List[Dict[str, float]] = []
    env_steps = 0
    last_rep_loss = float("nan")
    rep_update_every = int(rep_update_every or cfg.online_rep_update_every)
    rep_update_steps = int(rep_update_steps)
    rep_batch_size = int(rep_batch_size or cfg.online_rep_batch_size)
    rep_warmup_steps = int(rep_warmup_steps or cfg.online_rep_warmup_steps)
    if eval_every_updates is None:
        eval_every_updates = int(cfg.online_eval_every_updates)
    eval_buffer = eval_buffer or rep_buffer

    p_succ_hat = 0.0
    int_count = 0
    int_mean = 0.0
    int_m2 = 0.0
    use_alpha_anneal = bool(getattr(cfg, "ppo_use_alpha_anneal", False))
    use_alpha_gate = bool(getattr(cfg, "ppo_use_alpha_gate", False)) or bool(
        getattr(cfg, "ppo_alpha_zero_after_hit", False)
    )
    use_int_norm = bool(getattr(cfg, "ppo_use_int_norm", False))
    int_clip = float(getattr(cfg, "ppo_int_clip", 0.0))
    int_eps = float(getattr(cfg, "ppo_int_norm_eps", 1e-8))
    alpha0 = float(getattr(cfg, "ppo_alpha0", 1.0))
    alpha_eta = float(getattr(cfg, "ppo_alpha_eta", 1.0))
    alpha_rho = float(getattr(cfg, "ppo_alpha_rho", 0.05))
    alpha_zero_after_hit = use_alpha_gate

    goal = torch.tensor(maze_cfg["goal"], device=device, dtype=torch.long)

    for update in range(1, updates + 1):
        obs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs, policy_input_dim), device=device)
        actions_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device, dtype=torch.long)
        logprobs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        values_buf = (
            torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
            if not use_two_critic
            else None
        )
        values_ext_buf = (
            torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
            if use_two_critic
            else None
        )
        values_int_buf = (
            torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
            if use_two_critic
            else None
        )
        rewards_ext_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        rewards_int_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        dones_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        alpha_mask_buf = (
            torch.ones((steps_per_update, cfg.ppo_num_envs), device=device)
            if (alpha_zero_after_hit and use_extrinsic)
            else None
        )
        hit_mask = (
            torch.zeros((cfg.ppo_num_envs,), device=device, dtype=torch.bool)
            if alpha_mask_buf is not None
            else None
        )

        for t in range(steps_per_update):
            with torch.no_grad():
                rep_obs = rep.encode(obs).detach()
                policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
                out = model(policy_obs)
                logits = out[0]
                if use_two_critic:
                    values_ext, values_int = out[1], out[2]
                else:
                    values = out[1]
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            bonus_vals = bonus.compute_and_update(rep_obs, actions)
            rewards_int = bonus_vals

            next_obs, done, reset_obs = env.step(actions)
            if done.any():
                done_ids = torch.nonzero(done).squeeze(-1)
                bonus.reset(done_ids)
            if use_extrinsic:
                pos_next = _obs_to_pos(next_obs, int(maze_cfg["maze_size"]))
                reached = (pos_next == goal.unsqueeze(0)).all(dim=1)
                rewards_ext = reached.float()
            else:
                rewards_ext = torch.zeros_like(rewards_int)

            obs_buf[t] = policy_obs
            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            if use_two_critic:
                values_ext_buf[t] = values_ext
                values_int_buf[t] = values_int
            else:
                values_buf[t] = values
            rewards_ext_buf[t] = rewards_ext
            rewards_int_buf[t] = rewards_int
            dones_buf[t] = done.float()
            if alpha_mask_buf is not None:
                alpha_mask_buf[t] = (~hit_mask).float()
                hit_mask = hit_mask | (rewards_ext > 0)
                hit_mask = torch.where(done, torch.zeros_like(hit_mask), hit_mask)

            if eval_buffer is not None:
                eval_buffer.add_batch(obs, actions, rewards_int, next_obs, done)
            if rep_buffer is not None and rep_buffer is not eval_buffer:
                rep_buffer.add_batch(obs, actions, rewards_int, next_obs, done)
            env_steps += cfg.ppo_num_envs

            if rep_updater is not None and rep_buffer is not None and rep_buffer.size >= rep_batch_size:
                if env_steps >= rep_warmup_steps and (env_steps % rep_update_every == 0):
                    last_rep_loss = rep_updater(rep_buffer, rep_batch_size, rep_update_steps)

            obs = reset_obs

        with torch.no_grad():
            rep_obs = rep.encode(obs).detach()
            policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
            out = model(policy_obs)
            if use_two_critic:
                next_value_ext, next_value_int = out[1], out[2]
            else:
                next_value = out[1]

        rewards_int_flat = rewards_int_buf.reshape(-1)
        if use_int_norm:
            if int_count > 1:
                int_sigma = float((int_m2 / max(1, int_count - 1)) ** 0.5)
            else:
                int_sigma = float(rewards_int_flat.std(unbiased=False).item())
            int_sigma = max(int_sigma, int_eps)
            rewards_int_norm = rewards_int_buf / float(int_sigma)
            int_count, int_mean, int_m2 = _update_running_stats(
                int_count, int_mean, int_m2, rewards_int_flat
            )
        else:
            int_sigma = 1.0
            rewards_int_norm = rewards_int_buf
        if int_clip > 0:
            rewards_int_norm = rewards_int_norm.clamp(-float(int_clip), float(int_clip))

        rollout_success = float((rewards_ext_buf > 0).any().item()) if use_extrinsic else 0.0
        if use_extrinsic:
            if use_alpha_anneal:
                p_succ_hat = (1.0 - alpha_rho) * p_succ_hat + alpha_rho * rollout_success
                alpha = alpha0 * ((1.0 - p_succ_hat) ** alpha_eta)
            else:
                alpha = alpha0
        else:
            alpha = alpha0

        success_rate = float("nan")
        if use_extrinsic:
            success_rate = float((rewards_ext_buf > 0).any(dim=0).float().mean().item())

        if alpha_mask_buf is None:
            alpha_t = alpha
        else:
            alpha_t = alpha * alpha_mask_buf

        ret_mix = rewards_ext_buf + alpha_t * rewards_int_norm
        if use_two_critic:
            adv_ext, ret_ext = _compute_gae(
                rewards_ext_buf,
                values_ext_buf,
                dones_buf,
                next_value_ext,
                cfg.ppo_gamma,
                cfg.ppo_gae_lambda,
            )
            adv_int, ret_int = _compute_gae(
                rewards_int_norm,
                values_int_buf,
                dones_buf,
                next_value_int,
                cfg.ppo_gamma,
                cfg.ppo_gae_lambda,
            )
            adv = adv_ext + alpha_t * adv_int
        else:
            adv, ret = _compute_gae(
                ret_mix,
                values_buf,
                dones_buf,
                next_value,
                cfg.ppo_gamma,
                cfg.ppo_gae_lambda,
            )

        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        b_obs = obs_buf.reshape(-1, policy_input_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        if use_two_critic:
            b_returns_ext = ret_ext.reshape(-1)
            b_returns_int = ret_int.reshape(-1)
            b_values_ext = values_ext_buf.reshape(-1)
            b_values_int = values_int_buf.reshape(-1)
        else:
            b_returns = ret.reshape(-1)
            b_values = values_buf.reshape(-1)

        n_batch = b_obs.shape[0]
        mb_size = min(int(cfg.ppo_minibatch_size), n_batch)

        for _ in range(int(cfg.ppo_epochs)):
            idx = torch.randperm(n_batch, device=device)
            for start in range(0, n_batch, mb_size):
                mb = idx[start : start + mb_size]
                out = model(b_obs[mb])
                logits = out[0]
                if use_two_critic:
                    values_ext, values_int = out[1], out[2]
                else:
                    values = out[1]
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb])
                ratio = (new_logprobs - b_logprobs[mb]).exp()

                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * torch.clamp(
                    ratio, 1.0 - cfg.ppo_clip_coef, 1.0 + cfg.ppo_clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if use_two_critic:
                    v_loss_ext = ((values_ext - b_returns_ext[mb]) ** 2).mean()
                    v_loss_int = ((values_int - b_returns_int[mb]) ** 2).mean()
                    v_loss = v_loss_ext + v_loss_int
                else:
                    v_loss = ((values - b_returns[mb]) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = pg_loss + cfg.ppo_vf_coef * v_loss - cfg.ppo_ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo_max_grad_norm)
                opt.step()

        logs.append(
            {
                "update": update,
                "env_steps": int(env_steps),
                "mean_reward": float(ret_mix.mean().item()),
                "mean_reward_ext": float(rewards_ext_buf.mean().item()),
                "mean_reward_int": float(rewards_int_buf.mean().item()),
                "value_loss": float(v_loss.item()),
                "policy_loss": float(pg_loss.item()),
                "entropy": float(entropy.item()),
                "rep_loss": float(last_rep_loss),
                "alpha": float(alpha),
                "p_succ_hat": float(p_succ_hat),
                "int_sigma": float(int_sigma),
                "success_rate": float(success_rate),
                "two_critic": float(use_two_critic),
            }
        )

        if eval_callback is not None and (update == 1 or update % eval_every_updates == 0 or update == updates):
            metrics = eval_callback(update, env_steps, model)
            if use_extrinsic:
                metrics["success_rate"] = float(success_rate)
                metrics["p_succ_hat"] = float(p_succ_hat)
            metrics_log.append(metrics)

    return model, logs, metrics_log
