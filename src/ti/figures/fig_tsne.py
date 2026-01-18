import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from ti.data.collect import collect_offline_dataset
from ti.data.buffer import build_episode_index_strided
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.metrics.elliptical import _obs_from_pos_periodic
from ti.models.rep_methods import OfflineRepLearner
from ti.utils import ensure_dir


def _sample_periodic_obs(num_pos, num_nuis, maze_cfg, device):
    layout = torch.ones((maze_cfg["maze_size"], maze_cfg["maze_size"]), device=device, dtype=torch.bool)
    layout[1:-1, 1:-1] = False
    free = torch.nonzero(~layout).long()
    idx = torch.randint(0, free.shape[0], (num_pos,), device=device)
    pos = free[idx]
    phases = torch.randint(0, maze_cfg["periodic_P"], (num_pos, num_nuis), device=device)
    pos_rep = pos[:, None, :].expand(num_pos, num_nuis, 2).reshape(-1, 2)
    ph_rep = phases.reshape(-1)
    obs = _obs_from_pos_periodic(pos_rep, ph_rep, maze_cfg["maze_size"], maze_cfg["periodic_P"])
    labels = ph_rep.detach().cpu().numpy()
    return obs, labels


def _fit_tsne(z, seed):
    tsne = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30)
    return tsne.fit_transform(z)


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    train_cfg = methods_cfg["train"]
    maze_cfg = build_maze_cfg(cfg)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    env_id = fig_spec.get("env", "periodicity")
    env_spec = get_env_spec(cfg, env_id)
    num_pos = int(fig_spec.get("num_positions", 200))
    num_nuis = int(fig_spec.get("num_nuisances", 8))
    method_a = fig_spec.get("method_a", "CRTR_R8")
    method_b = fig_spec.get("method_b", "ICM")

    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    ensure_dir(fig_dir)

    buf, _env = collect_offline_dataset(
        env_spec["ctor"],
        train_cfg["offline_collect_steps"],
        train_cfg["offline_num_envs"],
        maze_cfg,
        device,
    )
    epi = build_episode_index_strided(buf.timestep, buf.size, train_cfg["offline_num_envs"], device)
    obs, labels = _sample_periodic_obs(num_pos, num_nuis, maze_cfg, device)

    def train_method(method_name):
        if method_name.startswith("CRTR_R"):
            rep = int(method_name.split("CRTR_R")[-1])
            method = "CRTR"
        else:
            rep = methods_cfg["model"]["crtr_rep_default"]
            method = method_name
        learner = OfflineRepLearner(
            method,
            obs_dim=maze_cfg["obs_dim"],
            z_dim=methods_cfg["model"]["z_dim"],
            hidden_dim=methods_cfg["model"]["hidden_dim"],
            n_actions=maze_cfg["n_actions"],
            crtr_temp=methods_cfg["model"]["crtr_temp"],
            crtr_rep=rep,
            k_cap=methods_cfg["model"]["k_cap"],
            geom_p=methods_cfg["model"]["geom_p"],
            device=device,
            lr=methods_cfg["model"]["lr"],
        ).to(device)
        learner.train_steps(
            buf,
            epi,
            train_cfg["offline_train_steps"],
            train_cfg["offline_batch_size"],
            train_cfg["print_train_every"],
            use_amp=runtime.get("use_amp", False),
            amp_dtype=runtime.get("amp_dtype", "bf16"),
            resume=False,
        )
        with torch.no_grad():
            z = learner.rep_enc(obs).detach().cpu().numpy()
        return z

    z_a = train_method(method_a)
    z_b = train_method(method_b)

    emb_a = _fit_tsne(z_a, runtime["seed"])
    emb_b = _fit_tsne(z_b, runtime["seed"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=False, sharey=False)
    sc1 = axes[0].scatter(emb_a[:, 0], emb_a[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    axes[0].set_title(f"{method_a}")
    sc2 = axes[1].scatter(emb_b[:, 0], emb_b[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    axes[1].set_title(f"{method_b}")
    fig.colorbar(sc2, ax=axes, shrink=0.8, label="Nuisance")
    fig.suptitle(f"t-SNE Invariance Collapse ({env_spec['name']})")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fig_id}.{ext}"))
    plt.close(fig)
