import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ti.data.buffer import build_episode_index_strided
from ti.envs import make_env
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.models.biscuit import BISCUIT_VAE
from ti.models.rep_methods import OfflineRepLearner
from ti.online.agent_dqn import DQNAgent
from ti.online.buffer import OnlineReplayBuffer
from ti.online.intrinsic import EpisodicEllipticalBonus
from ti.training.cbm_train import ancestors_in_dyn_graph, train_cbm_models
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
    if use_amp and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    else:
        autocast_ctx = nullcontext()
    with autocast_ctx:
        loss = learner.loss(buf, epi, batch_size)
    learner.opt.zero_grad(set_to_none=True)
    loss.backward()
    learner.opt.step()
    return float(loss.item())


def _auc_success_from_logs(logs):
    if not logs:
        return 0.0
    steps = np.array([r.env_step for r in logs], dtype=np.float64)
    success = np.array([r.success for r in logs], dtype=np.float64)
    if steps.size < 2:
        return float(success.mean())
    cum_success = np.cumsum(success) / np.arange(1, len(success) + 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(cum_success, steps))
    return float(np.trapz(cum_success, steps))


def _metric_from_logs(logs, metric):
    if not logs:
        return 0.0
    if metric == "final_success":
        window = logs[-100:] if len(logs) >= 100 else logs
        return float(sum(r.success for r in window) / float(len(window)))
    return _auc_success_from_logs(logs)


def run_online_training(
    cfg,
    env_id,
    method,
    seed,
    alpha,
    output_dir,
    trial=None,
    prune_metric="auc_success",
    prune_every=None,
    prune_min_steps=0,
):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    online_cfg = dict(
        {
            "num_envs": 16,
            "total_steps": 200000,
            "buffer_size": 3200000,
            "batch_size": 256,
            "rep_batch_size": 256,
            "update_every": 4,
            "rep_update_every": 2,
            "target_update_every": 1000,
            "gamma": 0.99,
            "q_lr": 0.0005,
            "q_hidden": 128,
            "double_dqn": True,
            "n_step": 3,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "eps_decay_steps": 100000,
            "bonus_lambda": 1.0,
            "bonus_beta": 1.0,
            "alpha_list": [0.0, 0.1, 0.3, 1.0, 3.0],
            "cbm_warmup_steps": 5000,
            "cbm_update_every": 50000,
        }
    )
    online_cfg.update(methods_cfg.get("online", {}))
    maze_cfg = build_maze_cfg(cfg)

    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(int(seed), deterministic=bool(runtime.get("deterministic", True)))

    env_spec = get_env_spec(cfg, env_id)
    env = make_env(env_spec["ctor"], num_envs=online_cfg["num_envs"], maze_cfg=maze_cfg, device=device)

    obs = env.reset()
    obs_dim = maze_cfg["obs_dim"]
    z_dim = methods_cfg["model"]["z_dim"]

    buffer = OnlineReplayBuffer(obs_dim, online_cfg["buffer_size"], online_cfg["num_envs"], device=device)

    use_amp, amp_dtype, scaler = get_amp_settings(runtime, device)

    rep_dim = z_dim
    encode_fn = None
    rep_update_fn = None

    if method in ["CRTR", "IDM", "ICM", "RND"]:
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
        encode_fn = lambda x, L=learner: L.rep_enc(x)
        rep_update_fn = lambda _step: _update_rep(
            learner, buffer, online_cfg["rep_batch_size"], device, use_amp, amp_dtype
        )

    elif method == "BISCUIT":
        model = BISCUIT_VAE(
            obs_dim=obs_dim,
            z_dim=z_dim,
            n_actions=maze_cfg["n_actions"],
            hidden_dim=methods_cfg["model"]["hidden_dim"],
            tau_start=methods_cfg["model"]["biscuit"]["tau_start"],
            tau_end=methods_cfg["model"]["biscuit"]["tau_end"],
            interaction_reg_weight=methods_cfg["model"]["biscuit"]["interaction_reg_weight"],
            beta_kl=methods_cfg["model"]["biscuit"]["beta_kl"],
            device=device,
            lr=methods_cfg["model"]["lr"],
        ).to(device)
        if runtime.get("compile", False):
            model = maybe_compile(model, runtime)
        encode_fn = lambda x, M=model: M.encode_mean(x)
        biscuit_step = {"step": 0}

        def rep_update_fn(_step):
            s, a, sp, _ = buffer.sample(online_cfg["rep_batch_size"])
            loss = model.train_step(
                s,
                a,
                sp,
                biscuit_step["step"],
                online_cfg["total_steps"],
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            biscuit_step["step"] += 1
            return float(loss.item())

    elif method == "CBM":
        keep_mask = torch.ones((obs_dim,), device=device)
        cbm_last_update = {"step": -1}

        def encode_fn(x, mask=keep_mask):
            return x * mask

        def rep_update_fn(step):
            warmup = int(online_cfg.get("cbm_warmup_steps", 5000))
            every = int(online_cfg.get("cbm_update_every", 50000))
            if step < warmup:
                return None
            if cbm_last_update["step"] >= 0 and (step - cbm_last_update["step"]) < every:
                return None
            dyn, rew, cmi_dyn, cmi_rew, G_dyn, PR = train_cbm_models(
                buffer,
                obs_dim=obs_dim,
                n_actions=maze_cfg["n_actions"],
                maze_size=maze_cfg["maze_size"],
                goal=maze_cfg["goal"],
                steps=methods_cfg["train"]["offline_train_steps"],
                batch=methods_cfg["train"]["offline_batch_size"],
                n_neg=methods_cfg["model"]["cbm"]["n_neg"],
                eps_cmi=methods_cfg["model"]["cbm"]["eps_cmi"],
                lam1=methods_cfg["model"]["cbm"]["lam1"],
                lam2=methods_cfg["model"]["cbm"]["lam2"],
                lr=methods_cfg["model"]["cbm"]["lr"],
                device=device,
                print_every=methods_cfg["train"]["print_train_every"],
                ckpt_dir=None,
                ckpt_every=None,
                log_dir=None,
                losses_flush_every=methods_cfg["train"]["print_train_every"],
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                resume=False,
            )
            keep_state = ancestors_in_dyn_graph(G_dyn, PR, obs_dim=obs_dim, device=device)
            keep_mask.copy_(keep_state.float())
            cbm_last_update["step"] = int(step)
            return float(keep_state.sum().item())

        rep_dim = obs_dim

    else:
        raise ValueError(f"Unsupported online method: {method}")

    agent = DQNAgent(
        input_dim=rep_dim,
        n_actions=maze_cfg["n_actions"],
        hidden_dim=online_cfg["q_hidden"],
        lr=online_cfg["q_lr"],
        device=device,
        double=online_cfg.get("double_dqn", True),
    )

    bonus = EpisodicEllipticalBonus(
        z_dim=rep_dim,
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
    success_window = deque(maxlen=int(online_cfg.get("success_window", 100)))
    heatmap = torch.zeros((maze_cfg["maze_size"], maze_cfg["maze_size"]), device=device)
    ep_extr = torch.zeros((online_cfg["num_envs"],), device=device)
    ep_intr = torch.zeros((online_cfg["num_envs"],), device=device)

    total_steps = int(online_cfg["total_steps"])
    log_every = online_cfg.get("log_every", None)
    if log_every is None:
        log_every = max(1, total_steps // 20)
    else:
        log_every = int(log_every)
        if log_every < 1:
            log_every = 0

    last_q_loss = None
    last_rep_metric = None

    if log_every:
        print(
            f"[Online] {env_id}/{method} seed={seed} alpha={alpha} "
            f"steps={total_steps} num_envs={online_cfg['num_envs']} device={device}",
            flush=True,
        )

    start_time = time.time()
    for step in range(1, total_steps + 1):
        with torch.no_grad():
            z = encode_fn(obs)
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
                success_flag = int(ep_extr[idx].item() > 0)
                logs.append(
                    OnlineResult(
                        env_step=step,
                        episode_return_extrinsic=float(ep_extr[idx].item()),
                        episode_return_intrinsic=float(ep_intr[idx].item()),
                        success=success_flag,
                    )
                )
                success_window.append(success_flag)
            ep_extr[done_ids] = 0.0
            ep_intr[done_ids] = 0.0
            bonus.reset(done_ids)

        obs = reset_obs

        if buffer.size >= online_cfg["batch_size"] and step % online_cfg["update_every"] == 0:
            n_step = int(online_cfg.get("n_step", 1))
            s, a, r, sp, d = buffer.sample_nstep(online_cfg["batch_size"], n_step, online_cfg["gamma"])
            with torch.no_grad():
                z_sp = encode_fn(sp)
            z_s = encode_fn(s).detach()
            last_q_loss = agent.update((z_s, a, r, z_sp, d), gamma=online_cfg["gamma"], n_step=n_step)

        if buffer.size >= online_cfg["rep_batch_size"] and step % online_cfg["rep_update_every"] == 0:
            last_rep_metric = rep_update_fn(step)

        if step % online_cfg["target_update_every"] == 0:
            agent.sync_target()

        epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - step / eps_steps)

        if log_every and (step == 1 or step % log_every == 0 or step == total_steps):
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-9)
            eta = (total_steps - step) / max(steps_per_sec, 1e-9)
            q_loss_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            rep_str = f"{last_rep_metric:.4f}" if last_rep_metric is not None else "n/a"
            if len(success_window) > 0:
                success_rate = sum(success_window) / float(len(success_window))
                success_str = f"{success_rate:.3f} (last {len(success_window)})"
            else:
                success_str = "n/a"
            print(
                f"[Online] step {step}/{total_steps} eps={epsilon:.3f} buffer={buffer.size} "
                 f"q_loss={q_loss_str} rep_metric={rep_str} success={success_str} "
                f"{steps_per_sec:.1f} steps/s ETA {eta/60:.1f}m",
                flush=True,
            )

        if trial is not None and prune_every:
            if step >= int(prune_min_steps) and (step % int(prune_every) == 0 or step == total_steps):
                value = _metric_from_logs(logs, prune_metric)
                trial.report(value, step)
                try:
                    import optuna
                except Exception:
                    optuna = None
                if optuna is not None and trial.should_prune():
                    raise optuna.TrialPruned()

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
