import argparse
import csv
import os
import sys
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import copy
import numpy as np
import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.collect import collect_offline_dataset
from ti.envs import PeriodicMaze, SlipperyDelayMaze, TeacupMaze
from ti.online.buffer import OnlineReplayBuffer
from ti.utils import configure_torch, seed_everything

from periodicity_study.common import ensure_dir, free_positions_for_env, maze_cfg_from_config
from periodicity_study.config import StudyConfig
from periodicity_study.metrics import (
    action_dist_kl_by_position,
    bonus_metrics_from_heatmaps,
    build_bonus_heatmaps,
    coverage_from_buffer,
    heatmap_similarity_metrics,
    pairwise_ttests,
    rep_invariance_by_position,
)
from periodicity_study.plotting import (
    plot_bar,
    plot_bar_values,
    plot_heatmap,
    plot_heatmap_diff,
    plot_timeseries,
    plot_multi_timeseries,
)
from periodicity_study.ppo import train_ppo
from periodicity_study.representations import (
    CoordNuisanceRep,
    CoordOnlyRep,
    init_online_crtr,
    init_online_idm,
    train_or_load_crtr,
    train_or_load_idm,
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


def _with_env_title(title: str, env_label: str) -> str:
    return f"{title} - {env_label}"


def _coverage_time(metrics_log: list[dict], threshold: float) -> tuple[float, float]:
    if not metrics_log:
        return float("nan"), float("nan")
    for row in metrics_log:
        cov = float(row.get("coverage_fraction", float("nan")))
        if cov >= threshold:
            return float(row.get("env_steps", float("nan"))), float(
                row.get("steps_per_state", float("nan"))
            )
    last = metrics_log[-1]
    return float(last.get("env_steps", float("nan"))), float(
        last.get("steps_per_state", float("nan"))
    )


def _pairwise_ratios(values: dict) -> dict:
    keys = list(values.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = float(values[keys[i]])
            b = float(values[keys[j]])
            if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
                out[f"{keys[i]}_over_{keys[j]}"] = float("nan")
            else:
                out[f"{keys[i]}_over_{keys[j]}"] = a / b
    return out


def _eval_levels(
    rep, policy, cfg, device: torch.device, eval_buf, env_id: str, policy_obs_fn=None
) -> Dict[str, float]:
    rep_vals = rep_invariance_by_position(rep, cfg, device, env_id)
    rep_mean = float(rep_vals.mean().item())
    rep_std = float(rep_vals.std(unbiased=False).item())

    bonus_within_std = float("nan")
    bonus_between_std = float("nan")
    bonus_within_over_between = float("nan")
    coverage_frac = float("nan")
    if eval_buf is not None and eval_buf.size >= int(cfg.online_eval_min_buffer):
        h_mean, h_std = build_bonus_heatmaps(rep, eval_buf, cfg, device, env_id)
        bonus_metrics = bonus_metrics_from_heatmaps(h_mean, h_std)
        bonus_within_std = bonus_metrics["within_std_mean"]
        bonus_between_std = bonus_metrics["between_std"]
        bonus_within_over_between = bonus_metrics["within_over_between"]
        coverage_frac = coverage_from_buffer(eval_buf, cfg, device, env_id)

    action_vals = action_dist_kl_by_position(
        policy, rep, cfg, device, env_id, policy_obs_fn=policy_obs_fn
    )
    action_mean = float(action_vals.mean().item())
    action_std = float(action_vals.std(unbiased=False).item())

    return {
        "rep_invariance_mean": rep_mean,
        "rep_invariance_std": rep_std,
        "bonus_within_std_mean": bonus_within_std,
        "bonus_between_std": bonus_between_std,
        "bonus_within_over_between": bonus_within_over_between,
        "coverage_fraction": coverage_frac,
        "action_kl_mean": action_mean,
        "action_kl_std": action_std,
    }


def _build_env_specs(cfg):
    base_goal = tuple(cfg.goal)
    big_goal = (int(cfg.maze_size_large) - 2, int(cfg.maze_size_large) - 2)
    return [
        {
            "id": "periodicity",
            "name": "Periodicity",
            "ctor": PeriodicMaze,
            "maze_size": int(cfg.maze_size),
            "goal": base_goal,
            "periodic_P": int(cfg.periodic_P),
        },
        {
            "id": "slippery",
            "name": "Delay Action Queue",
            "ctor": SlipperyDelayMaze,
            "maze_size": int(cfg.maze_size),
            "goal": base_goal,
            "slippery_D": int(cfg.slippery_D),
        },
        {
            "id": "teacup",
            "name": "Teacup Maze",
            "ctor": TeacupMaze,
            "maze_size": int(cfg.maze_size),
            "goal": base_goal,
            "teacup_P": int(cfg.teacup_P),
        },
        {
            "id": "periodicity_large",
            "name": "Periodicity (Large)",
            "ctor": PeriodicMaze,
            "maze_size": int(cfg.maze_size_large),
            "goal": big_goal,
            "periodic_P": int(cfg.periodic_P),
        },
        {
            "id": "slippery_large",
            "name": "Delay Action Queue (Large)",
            "ctor": SlipperyDelayMaze,
            "maze_size": int(cfg.maze_size_large),
            "goal": big_goal,
            "slippery_D": int(cfg.slippery_D),
        },
        {
            "id": "teacup_large",
            "name": "Teacup Maze (Large)",
            "ctor": TeacupMaze,
            "maze_size": int(cfg.maze_size_large),
            "goal": big_goal,
            "teacup_P": int(cfg.teacup_P),
        },
    ]


def _apply_env_spec(cfg, env_spec):
    cfg.maze_size = int(env_spec["maze_size"])
    cfg.goal = tuple(env_spec["goal"])
    if "periodic_P" in env_spec:
        cfg.periodic_P = int(env_spec["periodic_P"])
    if "slippery_D" in env_spec:
        cfg.slippery_D = int(env_spec["slippery_D"])
    if "teacup_P" in env_spec:
        cfg.teacup_P = int(env_spec["teacup_P"])


def _run_env(cfg, env_spec, device: torch.device, args) -> None:
    env_id = env_spec["id"]
    env_name = env_spec["name"]
    env_ctor = env_spec["ctor"]
    env_label = f"{env_name} ({env_id})"

    fig_dir = os.path.join(cfg.output_dir, "figures", env_id)
    table_dir = os.path.join(cfg.output_dir, "tables", env_id)
    model_dir = os.path.join(cfg.output_dir, "models", env_id)
    log_dir = os.path.join(cfg.output_dir, "logs", env_id)
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    maze_cfg = maze_cfg_from_config(cfg)
    free_count = int(free_positions_for_env(env_id, maze_cfg["maze_size"], device).shape[0])
    buf, _ = collect_offline_dataset(
        env_ctor,
        cfg.offline_collect_steps,
        cfg.offline_num_envs,
        maze_cfg,
        device,
    )
    epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

    if env_id.startswith("slippery"):
        nuis_count = cfg.slippery_D
    elif env_id.startswith("teacup"):
        nuis_count = cfg.teacup_P
    else:
        nuis_count = cfg.periodic_P

    policy_obs_fn = lambda obs, rep_obs: obs
    policy_input_dim = int(cfg.obs_dim)

    reps = {
        "coord_only": CoordOnlyRep(),
        "coord_plus_nuisance": CoordNuisanceRep(env_id, nuis_count, device),
        "crtr_learned": train_or_load_crtr(
            cfg, device, model_dir, force_retrain=args.force_retrain, buf=buf, epi=epi
        ),
        "idm_learned": train_or_load_idm(
            cfg, device, model_dir, force_retrain=args.force_retrain, buf=buf, epi=epi
        ),
    }
    selected_reps = getattr(args, "reps_set", None)
    if selected_reps is not None:
        reps = {k: v for k, v in reps.items() if k in selected_reps}

    print(f"\n=== {env_name} ({env_id}) ===")
    print("Stage 1: Representation invariance metrics")
    rep_metrics = {k: rep_invariance_by_position(v, cfg, device, env_id) for k, v in reps.items()}
    rep_p = pairwise_ttests(rep_metrics)
    plot_bar(
        rep_metrics,
        title=_with_env_title("Representation Invariance (same position, different nuisance)", env_label),
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
        h_mean, h_std = build_bonus_heatmaps(rep, buf, cfg, device, env_id)
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
            title=_with_env_title(f"Elliptical bonus (mean) - {name}", env_label),
            out_path=os.path.join(fig_dir, f"heat_bonus_mean_{name}.png"),
        )
        plot_heatmap(
            h_std,
            title=_with_env_title(f"Elliptical bonus (nuisance std) - {name}", env_label),
            out_path=os.path.join(fig_dir, f"heat_bonus_std_{name}.png"),
        )

        vals = h_mean[mask]
        mu = vals.mean()
        sigma = vals.std(unbiased=False) + 1e-8
        h_norm = (h_mean - mu) / sigma
        plot_heatmap(
            h_norm,
            title=_with_env_title(f"Elliptical bonus (mean, normalized) - {name}", env_label),
            out_path=os.path.join(fig_dir, f"heat_bonus_mean_norm_{name}.png"),
        )
        heat_mean[name + "_norm"] = h_norm

    bonus_mean_p = pairwise_ttests(bonus_mean_vals)
    bonus_std_p = pairwise_ttests(bonus_std_vals)
    bonus_ratio_p = pairwise_ttests(bonus_ratio_vals)

    plot_bar(
        bonus_mean_vals,
        title=_with_env_title("Elliptical Bonus Mean Across States", env_label),
        ylabel="Mean bonus",
        out_path=os.path.join(fig_dir, "bonus_mean_by_rep.png"),
        p_values=bonus_mean_p,
    )
    plot_bar(
        bonus_std_vals,
        title=_with_env_title("Elliptical Bonus Nuisance-Std (within state)", env_label),
        ylabel="Std over nuisance",
        out_path=os.path.join(fig_dir, "bonus_nuisance_std_by_rep.png"),
        p_values=bonus_std_p,
    )
    plot_bar(
        bonus_ratio_vals,
        title=_with_env_title("Elliptical Bonus (within-state std / between-state std)", env_label),
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
                title=_with_env_title(f"Norm bonus diff: {names[i]} - {names[j]}", env_label),
                out_path=os.path.join(fig_dir, f"heat_bonus_norm_diff_{names[i]}_minus_{names[j]}.png"),
            )

    print("Stage 3: PPO action distribution invariance (bonus-only)")
    action_kl = {}
    policies = {}
    coverage_series = {}
    coverage_series_goal = {}
    action_kl_goal = {}
    eval_buf_size = int(cfg.online_eval_buffer_size)
    if eval_buf_size <= 0:
        eval_buf_size = int(cfg.ppo_total_steps) * int(cfg.ppo_num_envs)

    metrics_to_plot = {
        "rep_invariance_mean": "Rep invariance (mean ||z1 - z2||)",
        "bonus_within_std_mean": "Bonus within-state std",
        "bonus_between_std": "Bonus between-state std",
        "bonus_within_over_between": "Bonus within/between ratio",
        "action_kl_mean": "Action KL mean",
    }
    timeseries_meta = {
        "crtr_learned": ("timeseries_crtr_fixed", "CRTR fixed (offline) metrics over training"),
        "idm_learned": ("timeseries_idm", "IDM (offline) PPO metrics over training"),
        "coord_only": ("timeseries_coord_only", "Coord-only (xy) PPO metrics over training"),
        "coord_plus_nuisance": (
            "timeseries_coord_plus_nuisance",
            "Coord + nuisance (xy + phase) PPO metrics over training",
        ),
        "crtr_online_joint": ("timeseries_crtr_online", "CRTR online (joint) metrics over training"),
        "idm_online_joint": ("timeseries_idm_online", "IDM online (joint) metrics over training"),
    }

    def _plot_rep_timeseries(rep_name: str, metrics_log: list[dict]) -> None:
        base = rep_name.replace("_goal", "")
        if base not in timeseries_meta:
            return
        fname, title = timeseries_meta[base]
        if rep_name.endswith("_goal"):
            fname = f"{fname}_goal"
            title = f"{title} + goal"
        plot_timeseries(
            metrics_log,
            title=_with_env_title(title, env_label),
            out_path=os.path.join(fig_dir, f"{fname}.png"),
            metrics=metrics_to_plot,
        )

    compare_metrics = {
        "rep_invariance_mean": "Rep invariance (mean ||z1 - z2||)",
        "bonus_within_std_mean": "Bonus within-state std",
        "bonus_between_std": "Bonus between-state std",
        "bonus_within_over_between": "Bonus within/between ratio",
        "action_kl_mean": "Action KL mean",
    }

    def _plot_ppo_summary(
        coverage_data: dict,
        action_kl_data: dict,
        coverage_plot: str,
        coverage_time_steps: str,
        coverage_time_steps_per_state: str,
        coverage_time_ratios: str,
        action_kl_plot: str,
        action_kl_values: str,
        action_kl_pvalues: str,
        compare_suffix: str,
        title_suffix: str,
        coverage_title: str,
        coverage_ratio_title: str,
        action_kl_title: str,
    ) -> None:
        if not coverage_data:
            return
        plot_multi_timeseries(
            coverage_data,
            title=_with_env_title(f"{coverage_title}{title_suffix}", env_label),
            out_path=os.path.join(fig_dir, f"{coverage_plot}.png"),
            y_key="coverage_percent",
            y_label="Coverage (%)",
            x_key="steps_per_state",
            x_label="Steps per free state",
            hline_y=100.0,
        )

        coverage_threshold = float(getattr(cfg, "coverage_threshold", 0.99))
        coverage_steps = {}
        coverage_steps_per_state = {}
        for name, log in coverage_data.items():
            steps, steps_per_state = _coverage_time(log, coverage_threshold)
            coverage_steps[name] = steps
            coverage_steps_per_state[name] = steps_per_state
        _save_kv(os.path.join(table_dir, f"{coverage_time_steps}.csv"), coverage_steps)
        _save_kv(
            os.path.join(table_dir, f"{coverage_time_steps_per_state}.csv"),
            coverage_steps_per_state,
        )
        coverage_ratios = _pairwise_ratios(coverage_steps_per_state)
        _save_kv(os.path.join(table_dir, f"{coverage_time_ratios}.csv"), coverage_ratios)
        plot_bar_values(
            coverage_ratios,
            title=_with_env_title(f"{coverage_ratio_title}{title_suffix}", env_label),
            ylabel="Steps-per-state ratio",
            out_path=os.path.join(fig_dir, f"{coverage_time_ratios}.png"),
        )

        if action_kl_data:
            action_kl_p = pairwise_ttests(action_kl_data)
            plot_bar(
                action_kl_data,
                title=_with_env_title(f"{action_kl_title}{title_suffix}", env_label),
                ylabel="Mean symmetric KL across nuisance",
                out_path=os.path.join(fig_dir, f"{action_kl_plot}.png"),
                p_values=action_kl_p,
            )
            _save_metric_values(
                os.path.join(table_dir, f"{action_kl_values}.csv"), action_kl_data
            )
            _save_kv(
                os.path.join(table_dir, f"{action_kl_pvalues}.csv"),
                action_kl_p,
            )

        for key, label in compare_metrics.items():
            plot_multi_timeseries(
                coverage_data,
                title=_with_env_title(f"PPO {label} over time (all reps){title_suffix}", env_label),
                out_path=os.path.join(fig_dir, f"timeseries_compare_{key}{compare_suffix}.png"),
                y_key=key,
                y_label=label,
                x_key="env_steps",
                hline_y=None,
            )

    run_goal = bool(getattr(args, "goal_only", False))
    run_intrinsic = bool(getattr(args, "intrinsic_only", False))
    if run_goal and run_intrinsic:
        raise ValueError("Cannot set both --goal-only and --intrinsic-only.")
    use_extrinsic_vals = (
        (True,) if run_goal else ((False,) if run_intrinsic else (False, True))
    )

    for name, rep in reps.items():
        for use_extrinsic in use_extrinsic_vals:
            run_name = name if not use_extrinsic else f"{name}_goal"
            print(f"  Training PPO for {run_name}...")
            eval_buf = OnlineReplayBuffer(cfg.obs_dim, eval_buf_size, cfg.ppo_num_envs, device)

            def eval_cb(update, env_steps, model, rep=rep, eval_buf=eval_buf):
                metrics = _eval_levels(
                    rep, model, cfg, device, eval_buf, env_id, policy_obs_fn=policy_obs_fn
                )
                metrics.update({"update": int(update), "env_steps": int(env_steps)})
                metrics["steps_per_state"] = float(env_steps) / max(1, free_count)
                cov = float(metrics.get("coverage_fraction", float("nan")))
                metrics["coverage_percent"] = cov * 100.0 if np.isfinite(cov) else float("nan")
                return metrics

            policy, logs, metrics_log = train_ppo(
                rep,
                cfg,
                device,
                env_ctor,
                maze_cfg,
                policy_obs_fn=policy_obs_fn,
                policy_input_dim=policy_input_dim,
                use_extrinsic=use_extrinsic,
                eval_callback=eval_cb,
                eval_every_updates=cfg.online_eval_every_updates,
                eval_buffer=eval_buf,
            )
            policies[run_name] = policy
            torch.save(policy.state_dict(), os.path.join(model_dir, f"ppo_{run_name}.pt"))

            with open(os.path.join(log_dir, f"ppo_logs_{run_name}.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            _save_timeseries(os.path.join(log_dir, f"metrics_timeseries_{run_name}.csv"), metrics_log)
            _plot_rep_timeseries(run_name, metrics_log)

            action_vals = action_dist_kl_by_position(
                policy, rep, cfg, device, env_id, policy_obs_fn=policy_obs_fn
            )
            if use_extrinsic:
                coverage_series_goal[run_name] = metrics_log
                action_kl_goal[run_name] = action_vals
            else:
                coverage_series[run_name] = metrics_log
                action_kl[run_name] = action_vals

    _plot_ppo_summary(
        coverage_series,
        action_kl,
        coverage_plot="ppo_coverage_over_time",
        coverage_time_steps="ppo_coverage_time_steps",
        coverage_time_steps_per_state="ppo_coverage_time_steps_per_state",
        coverage_time_ratios="ppo_coverage_time_ratios",
        action_kl_plot="ppo_action_kl_by_rep",
        action_kl_values="ppo_action_kl_values",
        action_kl_pvalues="ppo_action_kl_pvalues",
        compare_suffix="",
        title_suffix="",
        coverage_title="PPO coverage over time (full state coverage)",
        coverage_ratio_title="PPO coverage time ratios (steps per state)",
        action_kl_title="State-conditioned action distribution change across nuisance",
    )
    _plot_ppo_summary(
        coverage_series_goal,
        action_kl_goal,
        coverage_plot="ppo_coverage_over_time_goal",
        coverage_time_steps="ppo_coverage_time_steps_goal",
        coverage_time_steps_per_state="ppo_coverage_time_steps_per_state_goal",
        coverage_time_ratios="ppo_coverage_time_ratios_goal",
        action_kl_plot="ppo_action_kl_by_rep_goal",
        action_kl_values="ppo_action_kl_values_goal",
        action_kl_pvalues="ppo_action_kl_pvalues_goal",
        compare_suffix="_goal",
        title_suffix=" + goal",
        coverage_title="PPO coverage over time (full state coverage)",
        coverage_ratio_title="PPO coverage time ratios (steps per state)",
        action_kl_title="State-conditioned action distribution change across nuisance",
    )

    print("Stage 4: Online joint representations + bonus + PPO")
    online_rep = init_online_crtr(cfg, device)
    rep_buf_size = int(cfg.online_rep_buffer_size)
    if rep_buf_size <= 0:
        rep_buf_size = int(cfg.ppo_total_steps) * int(cfg.ppo_num_envs)
    rep_buf = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

    def online_eval_cb(update, env_steps, model, rep=online_rep, eval_buf=rep_buf):
        metrics = _eval_levels(
            rep, model, cfg, device, eval_buf, env_id, policy_obs_fn=policy_obs_fn
        )
        metrics.update({"update": int(update), "env_steps": int(env_steps)})
        metrics["steps_per_state"] = float(env_steps) / max(1, free_count)
        cov = float(metrics.get("coverage_fraction", float("nan")))
        metrics["coverage_percent"] = cov * 100.0 if np.isfinite(cov) else float("nan")
        return metrics

    if not run_goal:
        online_policy, online_logs, metrics_log = train_ppo(
            online_rep,
            cfg,
            device,
            env_ctor,
            maze_cfg,
            policy_obs_fn=policy_obs_fn,
            policy_input_dim=policy_input_dim,
            use_extrinsic=False,
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

        with open(
            os.path.join(log_dir, "ppo_logs_crtr_online_joint.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=online_logs[0].keys())
            writer.writeheader()
            writer.writerows(online_logs)
        _save_timeseries(
            os.path.join(log_dir, "metrics_timeseries_crtr_online_joint.csv"), metrics_log
        )

        coverage_series["crtr_online_joint"] = metrics_log
        _plot_rep_timeseries("crtr_online_joint", metrics_log)

    if not run_intrinsic:
        online_rep_goal = init_online_crtr(cfg, device)
        rep_buf_goal = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

        def online_goal_eval_cb(update, env_steps, model, rep=online_rep_goal, eval_buf=rep_buf_goal):
            metrics = _eval_levels(
                rep, model, cfg, device, eval_buf, env_id, policy_obs_fn=policy_obs_fn
            )
            metrics.update({"update": int(update), "env_steps": int(env_steps)})
            metrics["steps_per_state"] = float(env_steps) / max(1, free_count)
            cov = float(metrics.get("coverage_fraction", float("nan")))
            metrics["coverage_percent"] = cov * 100.0 if np.isfinite(cov) else float("nan")
            return metrics

        online_policy_goal, online_logs_goal, metrics_log_goal = train_ppo(
            online_rep_goal,
            cfg,
            device,
            env_ctor,
            maze_cfg,
            policy_obs_fn=policy_obs_fn,
            policy_input_dim=policy_input_dim,
            use_extrinsic=True,
            rep_updater=online_rep_goal.update,
            rep_buffer=rep_buf_goal,
            rep_update_every=cfg.online_rep_update_every,
            rep_update_steps=cfg.online_rep_update_steps,
            rep_batch_size=cfg.online_rep_batch_size,
            rep_warmup_steps=cfg.online_rep_warmup_steps,
            eval_callback=online_goal_eval_cb,
            eval_every_updates=cfg.online_eval_every_updates,
            eval_buffer=rep_buf_goal,
        )
        policies["crtr_online_joint_goal"] = online_policy_goal
        torch.save(
            online_policy_goal.state_dict(),
            os.path.join(model_dir, "ppo_crtr_online_joint_goal.pt"),
        )

        with open(
            os.path.join(log_dir, "ppo_logs_crtr_online_joint_goal.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=online_logs_goal[0].keys())
            writer.writeheader()
            writer.writerows(online_logs_goal)
        _save_timeseries(
            os.path.join(log_dir, "metrics_timeseries_crtr_online_joint_goal.csv"),
            metrics_log_goal,
        )

        coverage_series_goal["crtr_online_joint_goal"] = metrics_log_goal
        _plot_rep_timeseries("crtr_online_joint_goal", metrics_log_goal)

    online_idm = init_online_idm(cfg, device)
    idm_buf = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

    def online_idm_eval_cb(update, env_steps, model, rep=online_idm, eval_buf=idm_buf):
        metrics = _eval_levels(
            rep, model, cfg, device, eval_buf, env_id, policy_obs_fn=policy_obs_fn
        )
        metrics.update({"update": int(update), "env_steps": int(env_steps)})
        metrics["steps_per_state"] = float(env_steps) / max(1, free_count)
        cov = float(metrics.get("coverage_fraction", float("nan")))
        metrics["coverage_percent"] = cov * 100.0 if np.isfinite(cov) else float("nan")
        return metrics

    if not run_goal:
        idm_policy, idm_logs, idm_metrics_log = train_ppo(
            online_idm,
            cfg,
            device,
            env_ctor,
            maze_cfg,
            policy_obs_fn=policy_obs_fn,
            policy_input_dim=policy_input_dim,
            use_extrinsic=False,
            rep_updater=online_idm.update,
            rep_buffer=idm_buf,
            rep_update_every=cfg.online_rep_update_every,
            rep_update_steps=cfg.online_rep_update_steps,
            rep_batch_size=cfg.online_rep_batch_size,
            rep_warmup_steps=cfg.online_rep_warmup_steps,
            eval_callback=online_idm_eval_cb,
            eval_every_updates=cfg.online_eval_every_updates,
            eval_buffer=idm_buf,
        )
        policies["idm_online_joint"] = idm_policy
        torch.save(idm_policy.state_dict(), os.path.join(model_dir, "ppo_idm_online_joint.pt"))

        with open(
            os.path.join(log_dir, "ppo_logs_idm_online_joint.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=idm_logs[0].keys())
            writer.writeheader()
            writer.writerows(idm_logs)
        _save_timeseries(
            os.path.join(log_dir, "metrics_timeseries_idm_online_joint.csv"), idm_metrics_log
        )

        coverage_series["idm_online_joint"] = idm_metrics_log
        _plot_rep_timeseries("idm_online_joint", idm_metrics_log)

    if not run_intrinsic:
        online_idm_goal = init_online_idm(cfg, device)
        idm_buf_goal = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

        def online_idm_goal_eval_cb(update, env_steps, model, rep=online_idm_goal, eval_buf=idm_buf_goal):
            metrics = _eval_levels(
                rep, model, cfg, device, eval_buf, env_id, policy_obs_fn=policy_obs_fn
            )
            metrics.update({"update": int(update), "env_steps": int(env_steps)})
            metrics["steps_per_state"] = float(env_steps) / max(1, free_count)
            cov = float(metrics.get("coverage_fraction", float("nan")))
            metrics["coverage_percent"] = cov * 100.0 if np.isfinite(cov) else float("nan")
            return metrics

        idm_policy_goal, idm_logs_goal, idm_metrics_log_goal = train_ppo(
            online_idm_goal,
            cfg,
            device,
            env_ctor,
            maze_cfg,
            policy_obs_fn=policy_obs_fn,
            policy_input_dim=policy_input_dim,
            use_extrinsic=True,
            rep_updater=online_idm_goal.update,
            rep_buffer=idm_buf_goal,
            rep_update_every=cfg.online_rep_update_every,
            rep_update_steps=cfg.online_rep_update_steps,
            rep_batch_size=cfg.online_rep_batch_size,
            rep_warmup_steps=cfg.online_rep_warmup_steps,
            eval_callback=online_idm_goal_eval_cb,
            eval_every_updates=cfg.online_eval_every_updates,
            eval_buffer=idm_buf_goal,
        )
        policies["idm_online_joint_goal"] = idm_policy_goal
        torch.save(
            idm_policy_goal.state_dict(),
            os.path.join(model_dir, "ppo_idm_online_joint_goal.pt"),
        )

        with open(
            os.path.join(log_dir, "ppo_logs_idm_online_joint_goal.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=idm_logs_goal[0].keys())
            writer.writeheader()
            writer.writerows(idm_logs_goal)
        _save_timeseries(
            os.path.join(log_dir, "metrics_timeseries_idm_online_joint_goal.csv"),
            idm_metrics_log_goal,
        )

        coverage_series_goal["idm_online_joint_goal"] = idm_metrics_log_goal
        _plot_rep_timeseries("idm_online_joint_goal", idm_metrics_log_goal)

    action_kl_with_online = dict(action_kl)
    if not run_goal:
        action_kl_with_online["crtr_online_joint"] = action_dist_kl_by_position(
            online_policy, online_rep, cfg, device, env_id, policy_obs_fn=policy_obs_fn
        )
        action_kl_with_online["idm_online_joint"] = action_dist_kl_by_position(
            idm_policy, online_idm, cfg, device, env_id, policy_obs_fn=policy_obs_fn
        )
    action_kl_with_online_goal = dict(action_kl_goal)
    if not run_intrinsic:
        action_kl_with_online_goal["crtr_online_joint_goal"] = action_dist_kl_by_position(
            online_policy_goal,
            online_rep_goal,
            cfg,
            device,
            env_id,
            policy_obs_fn=policy_obs_fn,
        )
        action_kl_with_online_goal["idm_online_joint_goal"] = action_dist_kl_by_position(
            idm_policy_goal,
            online_idm_goal,
            cfg,
            device,
            env_id,
            policy_obs_fn=policy_obs_fn,
        )

    _plot_ppo_summary(
        coverage_series,
        action_kl_with_online,
        coverage_plot="ppo_coverage_over_time_with_online",
        coverage_time_steps="ppo_coverage_time_steps_with_online",
        coverage_time_steps_per_state="ppo_coverage_time_steps_per_state_with_online",
        coverage_time_ratios="ppo_coverage_time_ratios_with_online",
        action_kl_plot="ppo_action_kl_with_online",
        action_kl_values="ppo_action_kl_values_with_online",
        action_kl_pvalues="ppo_action_kl_pvalues_with_online",
        compare_suffix="",
        title_suffix="",
        coverage_title="PPO coverage over time (incl. online CRTR/IDM)",
        coverage_ratio_title="PPO coverage time ratios (incl. online)",
        action_kl_title="Action KL across nuisance (incl. online joint)",
    )
    _plot_ppo_summary(
        coverage_series_goal,
        action_kl_with_online_goal,
        coverage_plot="ppo_coverage_over_time_with_online_goal",
        coverage_time_steps="ppo_coverage_time_steps_with_online_goal",
        coverage_time_steps_per_state="ppo_coverage_time_steps_per_state_with_online_goal",
        coverage_time_ratios="ppo_coverage_time_ratios_with_online_goal",
        action_kl_plot="ppo_action_kl_with_online_goal",
        action_kl_values="ppo_action_kl_values_with_online_goal",
        action_kl_pvalues="ppo_action_kl_pvalues_with_online_goal",
        compare_suffix="_goal",
        title_suffix=" + goal",
        coverage_title="PPO coverage over time (incl. online CRTR/IDM)",
        coverage_ratio_title="PPO coverage time ratios (incl. online)",
        action_kl_title="Action KL across nuisance (incl. online joint)",
    )

    # Final metrics including online joint representation
    rep_metrics_all = dict(rep_metrics)
    rep_metrics_all["crtr_online_joint"] = rep_invariance_by_position(online_rep, cfg, device, env_id)
    rep_metrics_all["idm_online_joint"] = rep_invariance_by_position(online_idm, cfg, device, env_id)
    rep_p_all = pairwise_ttests(rep_metrics_all)
    plot_bar(
        rep_metrics_all,
        title=_with_env_title("Representation Invariance (incl. online joint)", env_label),
        ylabel="Mean ||z1 - z2||",
        out_path=os.path.join(fig_dir, "rep_invariance_with_online.png"),
        p_values=rep_p_all,
    )
    _save_metric_values(
        os.path.join(table_dir, "rep_invariance_values_with_online.csv"), rep_metrics_all
    )
    _save_kv(os.path.join(table_dir, "rep_invariance_pvalues_with_online.csv"), rep_p_all)

    h_mean_online, h_std_online = build_bonus_heatmaps(online_rep, buf, cfg, device, env_id)
    plot_heatmap(
        h_mean_online,
        title=_with_env_title("Elliptical bonus (mean) - crtr_online_joint", env_label),
        out_path=os.path.join(fig_dir, "heat_bonus_mean_crtr_online_joint.png"),
    )
    plot_heatmap(
        h_std_online,
        title=_with_env_title("Elliptical bonus (nuisance std) - crtr_online_joint", env_label),
        out_path=os.path.join(fig_dir, "heat_bonus_std_crtr_online_joint.png"),
    )
    h_mean_idm, h_std_idm = build_bonus_heatmaps(online_idm, buf, cfg, device, env_id)
    plot_heatmap(
        h_mean_idm,
        title=_with_env_title("Elliptical bonus (mean) - idm_online_joint", env_label),
        out_path=os.path.join(fig_dir, "heat_bonus_mean_idm_online_joint.png"),
    )
    plot_heatmap(
        h_std_idm,
        title=_with_env_title("Elliptical bonus (nuisance std) - idm_online_joint", env_label),
        out_path=os.path.join(fig_dir, "heat_bonus_std_idm_online_joint.png"),
    )

    bonus_mean_vals_all = dict(bonus_mean_vals)
    bonus_std_vals_all = dict(bonus_std_vals)
    bonus_ratio_vals_all = dict(bonus_ratio_vals)
    mask_online = torch.isfinite(h_mean_online)
    bonus_mean_vals_all["crtr_online_joint"] = h_mean_online[mask_online]
    bonus_std_vals_all["crtr_online_joint"] = h_std_online[mask_online]
    between_std_online = h_mean_online[mask_online].std(unbiased=False)
    bonus_ratio_vals_all["crtr_online_joint"] = h_std_online[mask_online] / (between_std_online + 1e-8)
    mask_idm = torch.isfinite(h_mean_idm)
    bonus_mean_vals_all["idm_online_joint"] = h_mean_idm[mask_idm]
    bonus_std_vals_all["idm_online_joint"] = h_std_idm[mask_idm]
    between_std_idm = h_mean_idm[mask_idm].std(unbiased=False)
    bonus_ratio_vals_all["idm_online_joint"] = h_std_idm[mask_idm] / (between_std_idm + 1e-8)
    bonus_mean_p_all = pairwise_ttests(bonus_mean_vals_all)
    bonus_std_p_all = pairwise_ttests(bonus_std_vals_all)
    bonus_ratio_p_all = pairwise_ttests(bonus_ratio_vals_all)
    plot_bar(
        bonus_mean_vals_all,
        title=_with_env_title("Elliptical Bonus Mean Across States (incl. online joint)", env_label),
        ylabel="Mean bonus",
        out_path=os.path.join(fig_dir, "bonus_mean_with_online.png"),
        p_values=bonus_mean_p_all,
    )
    plot_bar(
        bonus_std_vals_all,
        title=_with_env_title("Elliptical Bonus Nuisance-Std (incl. online joint)", env_label),
        ylabel="Std over nuisance",
        out_path=os.path.join(fig_dir, "bonus_nuisance_std_with_online.png"),
        p_values=bonus_std_p_all,
    )
    plot_bar(
        bonus_ratio_vals_all,
        title=_with_env_title("Elliptical Bonus Ratio (incl. online joint)", env_label),
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use smaller settings for quick runs.")
    parser.add_argument("--device", default=None, help="Override device, e.g., cpu or cuda:0")
    parser.add_argument("--force-retrain", action="store_true", help="Force CRTR retraining.")
    parser.add_argument(
        "--envs",
        default=None,
        help="Comma-separated env ids to run (e.g., periodicity,teacup_large).",
    )
    parser.add_argument("--only-large", action="store_true", help="Run only *_large envs.")
    parser.add_argument("--only-small", action="store_true", help="Run only base (non-_large) envs.")
    parser.add_argument(
        "--reps",
        default=None,
        help=(
            "Comma-separated reps to run for PPO: "
            "coord_only,coord_plus_nuisance,crtr_learned,idm_learned,crtr_online_joint,idm_online_joint"
        ),
    )
    parser.add_argument("--goal-only", action="store_true", help="Run only goal (extrinsic) PPO.")
    parser.add_argument(
        "--intrinsic-only", action="store_true", help="Run only intrinsic-only PPO."
    )
    args = parser.parse_args()

    base_cfg = StudyConfig()
    if args.fast:
        apply_fast_cfg(base_cfg)
    if args.device:
        base_cfg.device = args.device

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base_dir)
    base_cfg.output_dir = os.path.join(repo_root, base_cfg.output_dir)

    device = torch.device(base_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if base_cfg.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this study. Pass --device cuda:0 or set require_cuda=False.")
    configure_torch(
        {
            "deterministic": True,
            "allow_tf32": True,
            "matmul_precision": "high",
        },
        device,
    )
    seed_everything(int(base_cfg.seed), deterministic=True)

    env_specs = _build_env_specs(base_cfg)
    if args.only_large and args.only_small:
        raise ValueError("Cannot set both --only-large and --only-small.")
    if args.only_large:
        env_specs = [spec for spec in env_specs if spec["id"].endswith("_large")]
    if args.only_small:
        env_specs = [spec for spec in env_specs if not spec["id"].endswith("_large")]
    if args.envs:
        requested = {s.strip() for s in args.envs.split(",") if s.strip()}
        env_specs = [spec for spec in env_specs if spec["id"] in requested]
        missing = requested - {spec["id"] for spec in env_specs}
        if missing:
            known = ", ".join(sorted(spec["id"] for spec in _build_env_specs(base_cfg)))
            raise ValueError(f"Unknown env ids: {sorted(missing)}. Known: {known}")
    if not env_specs:
        raise ValueError("No environments selected. Check --envs/--only-large/--only-small.")

    args.reps_set = None
    if args.reps:
        requested = {s.strip() for s in args.reps.split(",") if s.strip()}
        known_reps = {
            "coord_only",
            "coord_plus_nuisance",
            "crtr_learned",
            "idm_learned",
            "crtr_online_joint",
            "idm_online_joint",
        }
        missing = requested - known_reps
        if missing:
            known = ", ".join(sorted(known_reps))
            raise ValueError(f"Unknown reps: {sorted(missing)}. Known: {known}")
        args.reps_set = requested

    for env_spec in env_specs:
        cfg = copy.deepcopy(base_cfg)
        _apply_env_spec(cfg, env_spec)
        _run_env(cfg, env_spec, device, args)
        continue
        env_id = env_spec["id"]
        env_name = env_spec["name"]
        env_ctor = env_spec["ctor"]

        fig_dir = os.path.join(cfg.output_dir, "figures", env_id)
        table_dir = os.path.join(cfg.output_dir, "tables", env_id)
        model_dir = os.path.join(cfg.output_dir, "models", env_id)
        log_dir = os.path.join(cfg.output_dir, "logs", env_id)
        ensure_dir(fig_dir)
        ensure_dir(table_dir)
        ensure_dir(model_dir)
        ensure_dir(log_dir)

        maze_cfg = maze_cfg_from_config(cfg)
        buf, _ = collect_offline_dataset(
            env_ctor,
            cfg.offline_collect_steps,
            cfg.offline_num_envs,
            maze_cfg,
            device,
        )
        epi = build_episode_index_strided(buf.timestep, buf.size, cfg.offline_num_envs, device)

        if env_id.startswith("slippery"):
            nuis_count = cfg.slippery_D
        elif env_id.startswith("teacup"):
            nuis_count = cfg.teacup_P
        else:
            nuis_count = cfg.periodic_P

        reps = {
            "coord_only": CoordOnlyRep(),
            "coord_plus_nuisance": CoordNuisanceRep(env_id, nuis_count, device),
            "crtr_learned": train_or_load_crtr(
                cfg, device, model_dir, force_retrain=args.force_retrain, buf=buf, epi=epi
            ),
            "idm_learned": train_or_load_idm(
                cfg, device, model_dir, force_retrain=args.force_retrain, buf=buf, epi=epi
            ),
        }

        print(f"\n=== {env_name} ({env_id}) ===")
        print("Stage 1: Representation invariance metrics")
        rep_metrics = {k: rep_invariance_by_position(v, cfg, device, env_id) for k, v in reps.items()}
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

    print(f"Study complete. Outputs in: {base_cfg.output_dir}")
    return

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
    coverage_series = {}
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
        coverage_series[name] = metrics_log

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
        elif name == "idm_learned":
            plot_timeseries(
                metrics_log,
                title="IDM (offline) PPO metrics over training",
                out_path=os.path.join(fig_dir, "timeseries_idm.png"),
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

    plot_multi_timeseries(
        coverage_series,
        title="PPO coverage over time (full state coverage)",
        out_path=os.path.join(fig_dir, "ppo_coverage_over_time.png"),
        y_key="coverage_fraction",
        y_label="Coverage fraction",
    )

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

    print("Stage 4: Online joint representations + bonus + PPO")
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

    coverage_series["crtr_online_joint"] = metrics_log

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

    online_idm = init_online_idm(cfg, device)
    idm_buf = OnlineReplayBuffer(cfg.obs_dim, rep_buf_size, cfg.ppo_num_envs, device)

    def online_idm_eval_cb(update, env_steps, model, rep=online_idm, eval_buf=idm_buf):
        metrics = _eval_levels(rep, model, cfg, device, eval_buf)
        metrics.update({"update": int(update), "env_steps": int(env_steps)})
        return metrics

    idm_policy, idm_logs, idm_metrics_log = train_ppo(
        online_idm,
        cfg,
        device,
        rep_updater=online_idm.update,
        rep_buffer=idm_buf,
        rep_update_every=cfg.online_rep_update_every,
        rep_update_steps=cfg.online_rep_update_steps,
        rep_batch_size=cfg.online_rep_batch_size,
        rep_warmup_steps=cfg.online_rep_warmup_steps,
        eval_callback=online_idm_eval_cb,
        eval_every_updates=cfg.online_eval_every_updates,
        eval_buffer=idm_buf,
    )
    policies["idm_online_joint"] = idm_policy
    torch.save(idm_policy.state_dict(), os.path.join(model_dir, "ppo_idm_online_joint.pt"))

    with open(os.path.join(log_dir, "ppo_logs_idm_online_joint.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=idm_logs[0].keys())
        writer.writeheader()
        writer.writerows(idm_logs)
    _save_timeseries(os.path.join(log_dir, "metrics_timeseries_idm_online_joint.csv"), idm_metrics_log)

    coverage_series["idm_online_joint"] = idm_metrics_log
    plot_multi_timeseries(
        coverage_series,
        title="PPO coverage over time (incl. online CRTR/IDM)",
        out_path=os.path.join(fig_dir, "ppo_coverage_over_time_with_online.png"),
        y_key="coverage_fraction",
        y_label="Coverage fraction",
    )

    plot_timeseries(
        idm_metrics_log,
        title="IDM online (joint) metrics over training",
        out_path=os.path.join(fig_dir, "timeseries_idm_online.png"),
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
    rep_metrics_all["idm_online_joint"] = rep_invariance_by_position(online_idm, cfg, device)
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
    h_mean_idm, h_std_idm = build_bonus_heatmaps(online_idm, buf, cfg, device)
    plot_heatmap(
        h_mean_idm,
        title="Elliptical bonus (mean) - idm_online_joint",
        out_path=os.path.join(fig_dir, "heat_bonus_mean_idm_online_joint.png"),
    )
    plot_heatmap(
        h_std_idm,
        title="Elliptical bonus (nuisance std) - idm_online_joint",
        out_path=os.path.join(fig_dir, "heat_bonus_std_idm_online_joint.png"),
    )

    bonus_mean_vals_all = dict(bonus_mean_vals)
    bonus_std_vals_all = dict(bonus_std_vals)
    bonus_ratio_vals_all = dict(bonus_ratio_vals)
    mask_online = torch.isfinite(h_mean_online)
    bonus_mean_vals_all["crtr_online_joint"] = h_mean_online[mask_online]
    bonus_std_vals_all["crtr_online_joint"] = h_std_online[mask_online]
    between_std_online = h_mean_online[mask_online].std(unbiased=False)
    bonus_ratio_vals_all["crtr_online_joint"] = h_std_online[mask_online] / (between_std_online + 1e-8)
    mask_idm = torch.isfinite(h_mean_idm)
    bonus_mean_vals_all["idm_online_joint"] = h_mean_idm[mask_idm]
    bonus_std_vals_all["idm_online_joint"] = h_std_idm[mask_idm]
    between_std_idm = h_mean_idm[mask_idm].std(unbiased=False)
    bonus_ratio_vals_all["idm_online_joint"] = h_std_idm[mask_idm] / (between_std_idm + 1e-8)
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
    action_kl_all["idm_online_joint"] = action_dist_kl_by_position(
        idm_policy, online_idm, cfg, device
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
