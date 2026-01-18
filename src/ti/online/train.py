import os
import time
from dataclasses import dataclass

import pandas as pd
import torch

from ti.data.buffer import build_episode_index_strided
from ti.envs import make_env
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.models.rep_methods import OfflineRepLearner
from ti.online.agent_dqn import DQNAgent
from ti.online.buffer import OnlineReplayBuffer
from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.utils import ensure_dir, get_amp_settings, maybe_compile, seed_everything


@dataclass
class OnlineResult:
    env_step: int
    episode_return_extrinsic: float
    episode_return_intrinsic: float
    success: int


def _obs_to_pos(obs, maze_size):
    xy = obs[:, :2]
    pos = torch.round(((xy + 1.0) * 0.5) * float(maze_size - 1)).long()
    return pos.clamp(0, maze_size - 1)


def _compute_reward(obs, maze_size, goal):
    pos = _obs_to_pos(obs, maze_size)
    goal_t = torch.tensor(goal, device=obs.device, dtype=torch.long)
    reached = (pos == goal_t.unsqueeze(0)).all(dim=1)
    return reached.float()


def _update_rep(learner, buf, batch_size, device, use_amp, amp_dtype):
    learner.train()
    if learner.method == "CRTR":
        epi = build_episode_index_strided(buf.timestep, buf.size, buf.num_envs, device)
        if epi.num_episodes == 0:
            return None
    else:
        epi = None
    if amp_dtype in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif amp_dtype in ("fp16", "float16"):
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        loss = learner.loss(buf, epi, batch_size)
    learner.opt.zero_grad(set_to_none=True)
    loss.backward()
    learner.opt.step()
    return float(loss.item())


def run_online_training(cfg, env_id, method, seed, alpha, output_dir):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    online_cfg = methods_cfg["online"]
    maze_cfg = build_maze_cfg(cfg)

    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(int(seed), deterministic=bool(runtime.get("deterministic", True)))

    env_spec = get_env_spec(cfg, env_id)
    env = make_env(env_spec["ctor"], num_envs=online_cfg["num_envs"], maze_cfg=maze_cfg, device=device)

    obs = env.reset()
    obs_dim = maze_cfg["obs_dim"]
    z_dim = methods_cfg["model"]["z_dim"]

    buffer = OnlineReplayBuffer(obs_dim, online_cfg["buffer_size"], online_cfg["num_envs"], device=device)

    learner = OfflineRepLearner(
        method,
        obs_dim=obs_dim,
        z_dim=z_dim,
        hidden_dim=methods_cfg["model"]["hidden_dim"],
        n_actions=maze_cfg["n_actions"],
        crtr_temp=methods_cfg["model"]["crtr_temp"],
        crtr_rep=methods_cfg["model"]["crtr_rep_default"],
        k_cap=methods_cfg["model"]["k_cap"],
        geom_p=methods_cfg["model"]["geom_p"],
        device=device,
        lr=methods_cfg["model"]["lr"],
    ).to(device)
    if runtime.get("compile", False):
        learner = maybe_compile(learner, runtime)

    use_amp, amp_dtype, _ = get_amp_settings(runtime, device)

    agent = DQNAgent(
        input_dim=z_dim,
        n_actions=maze_cfg["n_actions"],
        hidden_dim=online_cfg["q_hidden"],
        lr=online_cfg["q_lr"],
        device=device,
    )

    bonus = EpisodicEllipticalBonus(
        z_dim=z_dim,
        n_actions=maze_cfg["n_actions"],
        beta=online_cfg["bonus_beta"],
        lam=online_cfg["bonus_lambda"],
        num_envs=online_cfg["num_envs"],
        device=device,
    )

    eps_start = online_cfg["eps_start"]
    epsilon = eps_start
    eps_end = online_cfg["eps_end"]
    eps_steps = max(1, int(online_cfg["eps_decay_steps"]))

    logs = []
    heatmap = torch.zeros((maze_cfg["maze_size"], maze_cfg["maze_size"]), device=device)
    ep_extr = torch.zeros((online_cfg["num_envs"],), device=device)
    ep_intr = torch.zeros((online_cfg["num_envs"],), device=device)

    start_time = time.time()
    for step in range(1, int(online_cfg["total_steps"]) + 1):
        with torch.no_grad():
            z = learner.rep_enc(obs)
        actions = agent.act(z, epsilon)
        bonus_vals = bonus.compute_and_update(z.detach(), actions)

        next_obs, done, reset_obs = env.step(actions)
        extrinsic = _compute_reward(next_obs, maze_cfg["maze_size"], maze_cfg["goal"])
        total_reward = extrinsic + float(alpha) * bonus_vals

        buffer.add_batch(obs, actions, total_reward, next_obs, done)
        ep_extr += extrinsic
        ep_intr += bonus_vals

        pos = _obs_to_pos(next_obs, maze_cfg["maze_size"])
        heatmap[pos[:, 0], pos[:, 1]] += 1.0

        if done.any():
            done_ids = torch.nonzero(done).squeeze(-1)
            for idx in done_ids.tolist():
                logs.append(
                    OnlineResult(
                        env_step=step,
                        episode_return_extrinsic=float(ep_extr[idx].item()),
                        episode_return_intrinsic=float(ep_intr[idx].item()),
                        success=int(ep_extr[idx].item() > 0),
                    )
                )
            ep_extr[done_ids] = 0.0
            ep_intr[done_ids] = 0.0
            bonus.reset(done_ids)

        obs = reset_obs

        if buffer.size >= online_cfg["batch_size"] and step % online_cfg["update_every"] == 0:
            s, a, r, sp, d = buffer.sample_with_reward(online_cfg["batch_size"])
            with torch.no_grad():
                z_sp = learner.rep_enc(sp)
            z_s = learner.rep_enc(s).detach()
            loss = agent.update((z_s, a, r, z_sp, d), gamma=online_cfg["gamma"])

        if buffer.size >= online_cfg["rep_batch_size"] and step % online_cfg["rep_update_every"] == 0:
            _update_rep(learner, buffer, online_cfg["rep_batch_size"], device, use_amp, amp_dtype)

        if step % online_cfg["target_update_every"] == 0:
            agent.sync_target()

        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - step / eps_steps)

    runtime_s = time.time() - start_time

    df = pd.DataFrame([r.__dict__ for r in logs])
    df["env"] = env_spec["name"]
    df["method"] = method
    df["seed"] = int(seed)
    df["alpha"] = float(alpha)
    df["runtime_seconds"] = runtime_s

    ensure_dir(output_dir)
    out_csv = os.path.join(output_dir, f"{env_id}_{method}_seed{seed}_alpha{alpha}.csv")
    df.to_csv(out_csv, index=False)

    heatmap_path = os.path.join(output_dir, f"{env_id}_{method}_seed{seed}_alpha{alpha}_heatmap.pt")
    torch.save({"heatmap": heatmap.detach().cpu()}, heatmap_path)

    return out_csv, heatmap_path
