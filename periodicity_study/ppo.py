from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ti.envs import PeriodicMaze
from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.utils import seed_everything

from periodicity_study.common import maze_cfg_from_config


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)


@dataclass
class PPOBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


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
    policy_obs_fn=None,
    policy_input_dim: Optional[int] = None,
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
    maze_cfg = maze_cfg_from_config(cfg)

    env = PeriodicMaze(
        num_envs=cfg.ppo_num_envs,
        maze_size=maze_cfg["maze_size"],
        max_ep_steps=maze_cfg["max_ep_steps"],
        n_actions=maze_cfg["n_actions"],
        goal=maze_cfg["goal"],
        device=device,
    )

    if policy_input_dim is None:
        policy_input_dim = int(rep.dim)
    model = ActorCritic(policy_input_dim, cfg.n_actions, cfg.ppo_hidden_dim).to(device)
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

    for update in range(1, updates + 1):
        obs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs, policy_input_dim), device=device)
        actions_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device, dtype=torch.long)
        logprobs_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        values_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        rewards_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)
        dones_buf = torch.zeros((steps_per_update, cfg.ppo_num_envs), device=device)

        for t in range(steps_per_update):
            with torch.no_grad():
                rep_obs = rep.encode(obs).detach()
                policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
                logits, values = model(policy_obs)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            bonus_vals = bonus.compute_and_update(rep_obs, actions)
            rewards = bonus_vals

            next_obs, done, reset_obs = env.step(actions)
            if done.any():
                done_ids = torch.nonzero(done).squeeze(-1)
                bonus.reset(done_ids)

            obs_buf[t] = policy_obs
            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            values_buf[t] = values
            rewards_buf[t] = rewards
            dones_buf[t] = done.float()

            if eval_buffer is not None:
                eval_buffer.add_batch(obs, actions, rewards, next_obs, done)
            if rep_buffer is not None and rep_buffer is not eval_buffer:
                rep_buffer.add_batch(obs, actions, rewards, next_obs, done)
            env_steps += cfg.ppo_num_envs

            if rep_updater is not None and rep_buffer is not None and rep_buffer.size >= rep_batch_size:
                if env_steps >= rep_warmup_steps and (env_steps % rep_update_every == 0):
                    last_rep_loss = rep_updater(rep_buffer, rep_batch_size, rep_update_steps)

            obs = reset_obs

        with torch.no_grad():
            rep_obs = rep.encode(obs).detach()
            policy_obs = rep_obs if policy_obs_fn is None else policy_obs_fn(obs, rep_obs)
            _, next_value = model(policy_obs)

        adv, ret = _compute_gae(
            rewards_buf,
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
        b_returns = ret.reshape(-1)
        b_adv = adv.reshape(-1)

        n_batch = b_obs.shape[0]
        mb_size = min(int(cfg.ppo_minibatch_size), n_batch)

        for _ in range(int(cfg.ppo_epochs)):
            idx = torch.randperm(n_batch, device=device)
            for start in range(0, n_batch, mb_size):
                mb = idx[start : start + mb_size]
                logits, values = model(b_obs[mb])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb])
                ratio = (new_logprobs - b_logprobs[mb]).exp()

                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * torch.clamp(
                    ratio, 1.0 - cfg.ppo_clip_coef, 1.0 + cfg.ppo_clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                "mean_reward": float(rewards_buf.mean().item()),
                "value_loss": float(v_loss.item()),
                "policy_loss": float(pg_loss.item()),
                "entropy": float(entropy.item()),
                "rep_loss": float(last_rep_loss),
            }
        )

        if eval_callback is not None and (update == 1 or update % eval_every_updates == 0 or update == updates):
            metrics_log.append(eval_callback(update, env_steps, model))

    return model, logs, metrics_log
