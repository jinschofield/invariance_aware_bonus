import argparse
import csv
import os
import sys
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.collect import collect_offline_dataset
from ti.envs import PeriodicMaze
from ti.online.buffer import OnlineReplayBuffer
from ti.utils import configure_torch, seed_everything

from periodicity_study.common import ensure_dir, maze_cfg_from_config
from periodicity_study.config import StudyConfig
from periodicity_study.metrics import (
    action_dist_kl_by_position,
    bonus_metrics_from_heatmaps,
    build_bonus_heatmaps,
    heatmap_similarity_metrics,
    pairwise_ttests,
    rep_invariance_by_position,
)
from periodicity_study.plotting import plot_bar, plot_heatmap, plot_heatmap_diff, plot_timeseries
from periodicity_study.ppo import train_ppo
from periodicity_study.representations import (
    CoordOnlyRep,
    CoordPhaseRep,
    init_online_crtr,
    train_or_load_crtr,
)


def apply_fast_cfg(cfg: StudyConfig) -> None:
    cfg.offline_collect_steps = 5000
    cfg.offline_train_steps = 2000
    cfg.rep_positions = 100
    cfg.rep_pairs_per_pos = 8
    cfg.ppo_total_steps = 20000
    cfg.ppo_steps_per_update = 64
    cfg.ppo_minibatch_size = 512
    cfg.online_eval_every_updates = 2
    cfg.online_eval_min_buffer = 256
    cfg.online_eval_buffer_size = 200000
    cfg.online_rep_update_every = 512
    cfg.online_rep_update_steps = 1
    cfg.online_rep_batch_size = 256
    cfg.online_rep_warmup_steps = 1024


def _save_metric_values(path: str, values_by_rep: Dict[str, torch.Tensor]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "value"])
        for rep, values in values_by_rep.items():
            for v in values.detach().cpu().tolist():
                writer.writerow([rep, float(v)])


def _save_kv(path: str, rows: Dict[str, float]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in rows.items():
            writer.writerow([k, v])


def _save_timeseries(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _eval_levels(rep, policy, cfg, device: torch.device, eval_buf) -> Dict[str, float]:
    rep_vals = rep_invariance_by_position(rep, cfg, device)
    rep_mean = float(rep_vals.mean().item())
    rep_std = float(rep_vals.std(unbiased=False).item())

    bonus_within_std = float("nan")
    bonus_between_std = float("nan")
    bonus_within_over_between = float("nan")
    if eval_buf is not None and eval_buf.size >= int(cfg.online_eval_min_buffer):
        h_mean, h_std = build_bonus_heatmaps(rep, eval_buf, cfg, device)
        bonus_metrics = bonus_metrics_from_heatmaps(h_mean, h_std)
        bonus_within_std = bonus_metrics["within_std_mean"]
        bonus_between_std = bonus_metrics["between_std"]
        bonus_within_over_between = bonus_metrics["within_over_between"]

    action_vals = action_dist_kl_by_position(policy, rep, cfg, device)
    action_mean = float(action_vals.mean().item())
    action_std = float(action_vals.std(unbiased=False).item())

    return {
        "rep_invariance_mean": rep_mean,
        "rep_invariance_std": rep_std,
        "bonus_within_std_mean": bonus_within_std,
        "bonus_between_std": bonus_between_std,
        "bonus_within_over_between": bonus_within_over_between,
        "action_kl_mean": action_mean,
        "action_kl_std": action_std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use smaller settings for quick runs.")
    parser.add_argument("--device", default=None, help="Override device, e.g., cpu or cuda:0")
    parser.add_argument("--force-retrain", action="store_true", help="Force CRTR retraining.")
    args = parser.parse_args()

    cfg = StudyConfig()
    if args.fast:
        apply_fast_cfg(cfg)
    if args.device:
        cfg.device = args.device

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    cfg.output_dir = os.path.join(repo_root, cfg.output_dir)
    fig_dir = os.path.join(cfg.output_dir, "figures")
    table_dir = os.path.join(cfg.output_dir, "tables")
    model_dir = os.path.join(cfg.output_dir, "models")
    log_dir = os.path.join(cfg.output_dir, "logs")
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this study. Pass --device cuda:0 or set require_cuda=False.")
    configure_torch(
        {
            "deterministic": True,
            "allow_tf32": True,
            "matmul_precision": "high",
        },
        device,
    )
    seed_everything(int(cfg.seed), deterministic=True)

    maze_cfg = maze_cfg_from_config(cfg)
    buf, _ = collect_offline_dataset(
        PeriodicMaze,
        cfg.offline_collect_steps,
        cfg.offline_num_envs,
        maze_cfg,
        device,
    )
    epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

    reps = {
        "coord_only": CoordOnlyRep(),
        "coord_plus_nuisance": CoordPhaseRep(cfg.periodic_P),
        "crtr_learned": train_or_load_crtr(
            cfg, device, model_dir, force_retrain=args.force_retrain, buf=buf, epi=epi
        ),
    }

    print("Stage 1: Representation invariance metrics")
    rep_metrics = {k: rep_invariance_by_position(v, cfg, device) for k, v in reps.items()}
    rep_p = pairwise_ttests(rep_metrics)
    plot_bar(
        rep_metrics,
        title="Representation Invariance (same position, different nuisance)",
        ylabel="Mean ||z1 - z2||",
        out_path=os.path.join(fig_dir, "rep_invariance.png"),
        p_values=rep_p,
    )
    _save_metric_values(os.path.join(table_dir, "rep_invariance_values.csv"), rep_metrics)
    _save_kv(os.path.join(table_dir, "rep_invariance_pvalues.csv"), rep_p)

    print("Stage 2: Elliptical bonus metrics + heatmaps")
    heat_mean = {}
    heat_std = {}
    bonus_mean_vals = {}
    bonus_std_vals = {}
    bonus_scalars = {}
    bonus_ratio_vals = {}
    for name, rep in reps.items():
        h_mean, h_std = build_bonus_heatmaps(rep, buf, cfg, device)
        heat_mean[name] = h_mean
        heat_std[name] = h_std
        bonus_scalars[name] = bonus_metrics_from_heatmaps(h_mean, h_std)

        mask = torch.isfinite(h_mean)
        bonus_mean_vals[name] = h_mean[mask]
        bonus_std_vals[name] = h_std[mask]
        between_std = h_mean[mask].std(unbiased=False)
        bonus_ratio_vals[name] = h_std[mask] / (between_std + 1e-8)

        plot_heatmap(
            h_mean,
            title=f"Elliptical bonus (mean) - {name}",
            out_path=os.path.join(fig_dir, f"heat_bonus_mean_{name}.png"),
        )
        plot_heatmap(
            h_std,
            title=f"Elliptical bonus (nuisance std) - {name}",
            out_path=os.path.join(fig_dir, f"heat_bonus_std_{name}.png"),
        )

        vals = h_mean[mask]
        mu = vals.mean()
        sigma = vals.std(unbiased=False) + 1e-8
        h_norm = (h_mean - mu) / sigma
        plot_heatmap(
            h_norm,
            title=f"Elliptical bonus (mean, normalized) - {name}",
            out_path=os.path.join(fig_dir, f"heat_bonus_mean_norm_{name}.png"),
        )
        heat_mean[name + "_norm"] = h_norm

    bonus_mean_p = pairwise_ttests(bonus_mean_vals)
    bonus_std_p = pairwise_ttests(bonus_std_vals)
    bonus_ratio_p = pairwise_ttests(bonus_ratio_vals)

    plot_bar(
        bonus_mean_vals,
        title="Elliptical Bonus Mean Across States",
        ylabel="Mean bonus",
        out_path=os.path.join(fig_dir, "bonus_mean_by_rep.png"),
        p_values=bonus_mean_p,
    )
    plot_bar(
        bonus_std_vals,
        title="Elliptical Bonus Nuisance-Std (within state)",
        ylabel="Std over nuisance",
        out_path=os.path.join(fig_dir, "bonus_nuisance_std_by_rep.png"),
        p_values=bonus_std_p,
    )
    plot_bar(
        bonus_ratio_vals,
        title="Elliptical Bonus (within-state std / between-state std)",
        ylabel="Within/Between ratio",
        out_path=os.path.join(fig_dir, "bonus_within_over_between_by_rep.png"),
        p_values=bonus_ratio_p,
    )

    _save_metric_values(os.path.join(table_dir, "bonus_mean_values.csv"), bonus_mean_vals)
    _save_metric_values(os.path.join(table_dir, "bonus_std_values.csv"), bonus_std_vals)
    _save_metric_values(os.path.join(table_dir, "bonus_ratio_values.csv"), bonus_ratio_vals)
    _save_kv(os.path.join(table_dir, "bonus_mean_pvalues.csv"), bonus_mean_p)
    _save_kv(os.path.join(table_dir, "bonus_std_pvalues.csv"), bonus_std_p)
    _save_kv(os.path.join(table_dir, "bonus_ratio_pvalues.csv"), bonus_ratio_p)

    with open(os.path.join(table_dir, "bonus_scalar_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "within_std_mean", "between_std", "within_over_between"])
        for k, v in bonus_scalars.items():
            writer.writerow([k, v["within_std_mean"], v["between_std"], v["within_over_between"]])

    names = list(reps.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = heat_mean[names[i] + "_norm"]
            b = heat_mean[names[j] + "_norm"]
            sim = heatmap_similarity_metrics(a, b)
            _save_kv(
                os.path.join(table_dir, f"heatmap_similarity_{names[i]}_vs_{names[j]}.csv"),
                sim,
            )
            plot_heatmap_diff(
                a,
                b,
                title=f"Norm bonus diff: {names[i]} - {names[j]}",
                out_path=os.path.join(fig_dir, f"heat_bonus_norm_diff_{names[i]}_minus_{names[j]}.png"),
            )

    print("Stage 3: PPO action distribution invariance (bonus-only)")
    action_kl = {}
    policies = {}
    eval_buf_size = int(cfg.online_eval_buffer_size)
    if eval_buf_size <= 0:
        eval_buf_size = int(cfg.ppo_total_steps) * int(cfg.ppo_num_envs)

    for name, rep in reps.items():
        print(f"  Training PPO for {name}...")
        eval_buf = OnlineReplayBuffer(cfg.obs_dim, eval_buf_size, cfg.ppo_num_envs, device)

        def eval_cb(update, env_steps, model, rep=rep, eval_buf=eval_buf):
            metrics = _eval_levels(rep, model, cfg, device, eval_buf)
            metrics.update({"update": int(update), "env_steps": int(env_steps)})
            return metrics

        policy, logs, metrics_log = train_ppo(
            rep,
            cfg,
            device,
            eval_callback=eval_cb,
            eval_every_updates=cfg.online_eval_every_updates,
            eval_buffer=eval_buf,
        )
        policies[name] = policy
        torch.save(policy.state_dict(), os.path.join(model_dir, f"ppo_{name}.pt"))

        with open(os.path.join(log_dir, f"ppo_logs_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
        _save_timeseries(os.path.join(log_dir, f"metrics_timeseries_{name}.csv"), metrics_log)

        metrics_to_plot = {
            "rep_invariance_mean": "Rep invariance (mean ||z1 - z2||)",
            "bonus_within_std_mean": "Bonus within-state std",
            "bonus_between_std": "Bonus between-state std",
            "action_kl_mean": "Action KL mean",
        }

        if name == "crtr_learned":
            plot_timeseries(
                metrics_log,
                title="CRTR fixed (offline) metrics over training",
                out_path=os.path.join(fig_dir, "timeseries_crtr_fixed.png"),
                metrics=metrics_to_plot,
            )
        elif name == "coord_only":
            plot_timeseries(
                metrics_log,
                title="Coord-only (xy) PPO metrics over training",
                out_path=os.path.join(fig_dir, "timeseries_coord_only.png"),
                metrics=metrics_to_plot,
            )
        elif name == "coord_plus_nuisance":
            plot_timeseries(
                metrics_log,
                title="Coord + nuisance (xy + phase) PPO metrics over training",
                out_path=os.path.join(fig_dir, "timeseries_coord_plus_nuisance.png"),
                metrics=metrics_to_plot,
            )

        action_kl[name] = action_dist_kl_by_position(policy, rep, cfg, device)

    action_kl_p = pairwise_ttests(action_kl)
    plot_bar(
        action_kl,
        title="State-conditioned action distribution change across nuisance",
        ylabel="Mean symmetric KL across nuisance",
        out_path=os.path.join(fig_dir, "ppo_action_kl_by_rep.png"),
        p_values=action_kl_p,
    )
    _save_metric_values(os.path.join(table_dir, "ppo_action_kl_values.csv"), action_kl)
    _save_kv(os.path.join(table_dir, "ppo_action_kl_pvalues.csv"), action_kl_p)

    print("Stage 4: Online joint CRTR representation + bonus + PPO")
    online_rep = init_online_crtr(cfg, device)
    rep_buf_size = int(cfg.online_rep_buffer_size)
    if rep_buf_size <= 0:
        rep_buf_size = int(cfg.ppo_total_steps) * int(cfg.ppo_num_envs)
    rep_buf = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

    def online_eval_cb(update, env_steps, model, rep=online_rep, eval_buf=rep_buf):
        metrics = _eval_levels(rep, model, cfg, device, eval_buf)
        metrics.update({"update": int(update), "env_steps": int(env_steps)})
        return metrics

    online_policy, online_logs, metrics_log = train_ppo(
        online_rep,
        cfg,
        device,
        rep_updater=online_rep.update,
        rep_buffer=rep_buf,
        rep_update_every=cfg.online_rep_update_every,
        rep_update_steps=cfg.online_rep_update_steps,
        rep_batch_size=cfg.online_rep_batch_size,
        rep_warmup_steps=cfg.online_rep_warmup_steps,
        eval_callback=online_eval_cb,
        eval_every_updates=cfg.online_eval_every_updates,
        eval_buffer=rep_buf,
    )
    policies["crtr_online_joint"] = online_policy
    torch.save(online_policy.state_dict(), os.path.join(model_dir, "ppo_crtr_online_joint.pt"))

    with open(os.path.join(log_dir, "ppo_logs_crtr_online_joint.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=online_logs[0].keys())
        writer.writeheader()
        writer.writerows(online_logs)
    _save_timeseries(os.path.join(log_dir, "metrics_timeseries_crtr_online_joint.csv"), metrics_log)

    plot_timeseries(
        metrics_log,
        title="CRTR online (joint) metrics over training",
        out_path=os.path.join(fig_dir, "timeseries_crtr_online.png"),
        metrics={
            "rep_invariance_mean": "Rep invariance (mean ||z1 - z2||)",
            "bonus_within_std_mean": "Bonus within-state std",
            "bonus_between_std": "Bonus between-state std",
            "action_kl_mean": "Action KL mean",
        },
    )

    # Final metrics including online joint representation
    rep_metrics_all = dict(rep_metrics)
    rep_metrics_all["crtr_online_joint"] = rep_invariance_by_position(online_rep, cfg, device)
    rep_p_all = pairwise_ttests(rep_metrics_all)
    plot_bar(
        rep_metrics_all,
        title="Representation Invariance (incl. online joint)",
        ylabel="Mean ||z1 - z2||",
        out_path=os.path.join(fig_dir, "rep_invariance_with_online.png"),
        p_values=rep_p_all,
    )
    _save_metric_values(
        os.path.join(table_dir, "rep_invariance_values_with_online.csv"), rep_metrics_all
    )
    _save_kv(os.path.join(table_dir, "rep_invariance_pvalues_with_online.csv"), rep_p_all)

    h_mean_online, h_std_online = build_bonus_heatmaps(online_rep, buf, cfg, device)
    plot_heatmap(
        h_mean_online,
        title="Elliptical bonus (mean) - crtr_online_joint",
        out_path=os.path.join(fig_dir, "heat_bonus_mean_crtr_online_joint.png"),
    )
    plot_heatmap(
        h_std_online,
        title="Elliptical bonus (nuisance std) - crtr_online_joint",
        out_path=os.path.join(fig_dir, "heat_bonus_std_crtr_online_joint.png"),
    )

    bonus_mean_vals_all = dict(bonus_mean_vals)
    bonus_std_vals_all = dict(bonus_std_vals)
    bonus_ratio_vals_all = dict(bonus_ratio_vals)
    mask_online = torch.isfinite(h_mean_online)
    bonus_mean_vals_all["crtr_online_joint"] = h_mean_online[mask_online]
    bonus_std_vals_all["crtr_online_joint"] = h_std_online[mask_online]
    between_std_online = h_mean_online[mask_online].std(unbiased=False)
    bonus_ratio_vals_all["crtr_online_joint"] = h_std_online[mask_online] / (between_std_online + 1e-8)
    bonus_mean_p_all = pairwise_ttests(bonus_mean_vals_all)
    bonus_std_p_all = pairwise_ttests(bonus_std_vals_all)
    bonus_ratio_p_all = pairwise_ttests(bonus_ratio_vals_all)
    plot_bar(
        bonus_mean_vals_all,
        title="Elliptical Bonus Mean Across States (incl. online joint)",
        ylabel="Mean bonus",
        out_path=os.path.join(fig_dir, "bonus_mean_with_online.png"),
        p_values=bonus_mean_p_all,
    )
    plot_bar(
        bonus_std_vals_all,
        title="Elliptical Bonus Nuisance-Std (incl. online joint)",
        ylabel="Std over nuisance",
        out_path=os.path.join(fig_dir, "bonus_nuisance_std_with_online.png"),
        p_values=bonus_std_p_all,
    )
    plot_bar(
        bonus_ratio_vals_all,
        title="Elliptical Bonus Ratio (incl. online joint)",
        ylabel="Within/Between ratio",
        out_path=os.path.join(fig_dir, "bonus_ratio_with_online.png"),
        p_values=bonus_ratio_p_all,
    )
    _save_metric_values(
        os.path.join(table_dir, "bonus_mean_values_with_online.csv"), bonus_mean_vals_all
    )
    _save_metric_values(
        os.path.join(table_dir, "bonus_std_values_with_online.csv"), bonus_std_vals_all
    )
    _save_metric_values(
        os.path.join(table_dir, "bonus_ratio_values_with_online.csv"), bonus_ratio_vals_all
    )
    _save_kv(os.path.join(table_dir, "bonus_mean_pvalues_with_online.csv"), bonus_mean_p_all)
    _save_kv(os.path.join(table_dir, "bonus_std_pvalues_with_online.csv"), bonus_std_p_all)
    _save_kv(os.path.join(table_dir, "bonus_ratio_pvalues_with_online.csv"), bonus_ratio_p_all)

    action_kl_all = dict(action_kl)
    action_kl_all["crtr_online_joint"] = action_dist_kl_by_position(
        online_policy, online_rep, cfg, device
    )
    action_kl_p_all = pairwise_ttests(action_kl_all)
    plot_bar(
        action_kl_all,
        title="Action KL across nuisance (incl. online joint)",
        ylabel="Mean symmetric KL across nuisance",
        out_path=os.path.join(fig_dir, "ppo_action_kl_with_online.png"),
        p_values=action_kl_p_all,
    )
    _save_metric_values(
        os.path.join(table_dir, "ppo_action_kl_values_with_online.csv"), action_kl_all
    )
    _save_kv(os.path.join(table_dir, "ppo_action_kl_pvalues_with_online.csv"), action_kl_p_all)

    print(f"Study complete. Outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
