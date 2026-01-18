import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.collect import collect_offline_dataset
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.metrics.invariance import invariance_metric_from_pairs
from ti.metrics.probes import run_linear_probe_any
from ti.models.rep_methods import OfflineRepLearner
from ti.utils import ensure_dir, get_amp_settings


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    train_cfg = methods_cfg["train"]
    probe_cfg = methods_cfg["probes"]
    maze_cfg = build_maze_cfg(cfg)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    env_id = fig_spec.get("env", "periodicity")
    env_spec = get_env_spec(cfg, env_id)
    steps_list = fig_spec.get("steps", list(range(0, train_cfg["offline_train_steps"] + 1, 1000)))

    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], fig_id)
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    use_amp, amp_dtype, _ = get_amp_settings(runtime, device)

    buf, env = collect_offline_dataset(
        env_spec["ctor"],
        train_cfg["offline_collect_steps"],
        train_cfg["offline_num_envs"],
        maze_cfg,
        device,
    )
    epi = build_episode_index_strided(buf.timestep, buf.size, train_cfg["offline_num_envs"], device)

    obs_all = buf.s[: buf.size]
    y_all = buf.nuis[: buf.size].long()
    num_classes = int(env_spec["classes"])

    obs1, obs2 = env.sample_invariance_pairs(2048)

    learner = OfflineRepLearner(
        "CRTR",
        obs_dim=maze_cfg["obs_dim"],
        z_dim=methods_cfg["model"]["z_dim"],
        hidden_dim=methods_cfg["model"]["hidden_dim"],
        n_actions=maze_cfg["n_actions"],
        crtr_temp=methods_cfg["model"]["crtr_temp"],
        crtr_rep=methods_cfg["model"]["crtr_rep_default"],
        k_cap=methods_cfg["model"]["k_cap"],
        geom_p=methods_cfg["model"]["geom_p"],
        device=device,
        lr=methods_cfg["model"]["lr"],
    ).to(device)

    rows = []
    for step in range(0, train_cfg["offline_train_steps"] + 1):
        if step in steps_list:
            enc = lambda x, L=learner: L.rep_enc(x)
            acc, mi = run_linear_probe_any(enc, obs_all, y_all, num_classes, probe_cfg, seed=runtime["seed"], device=device)
            inv = invariance_metric_from_pairs(enc, obs1, obs2)
            rows.append({"step": step, "nuis_mi": mi, "inv": inv})
        if step == train_cfg["offline_train_steps"]:
            break
        if amp_dtype in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif amp_dtype in ("fp16", "float16"):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
            loss = learner.loss(buf, epi, train_cfg["offline_batch_size"])
        learner.opt.zero_grad(set_to_none=True)
        loss.backward()
        learner.opt.step()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(table_dir, "crtr_over_time.csv"), index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(df["step"], df["nuis_mi"], marker="o", label="MI proxy")
    ax.plot(df["step"], df["inv"], marker="o", label="Minv")
    ax.set_xlabel("Training step")
    ax.set_title(f"CRTR convergence ({env_spec['name']})")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
