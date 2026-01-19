import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from ti.config.defaults import load_config, load_yaml
from ti.online.train import run_online_training
from ti.utils import configure_torch, ensure_dir, seed_everything


def _auc_success(df: pd.DataFrame) -> float:
    df = df.sort_values("env_step")
    steps = df["env_step"].values
    success = df["success"].values
    cum_success = np.cumsum(success) / np.arange(1, len(success) + 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(cum_success, steps))
    return float(np.trapz(cum_success, steps))


def objective_factory(cfg, opt_cfg):
    runtime = cfg["runtime"]
    mode = runtime.get("throughput_mode", "never")
    if mode in ("always", "rl_only"):
        runtime.update(runtime.get("rl_profile", {}))
    methods_cfg = cfg["methods"]

    env_id = opt_cfg.get("env_id", "periodicity")
    method = opt_cfg.get("method", "CRTR")
    metric = opt_cfg.get("metric", "auc_success")
    n_steps = int(opt_cfg.get("total_steps", methods_cfg["online"]["total_steps"]))
    seed_base = int(runtime.get("seed", 0))

    def objective(trial):
        trial_cfg = deepcopy(cfg)
        online_cfg = trial_cfg["methods"]["online"]
        space = opt_cfg["search_space"]

        if "bonus_scale" in space:
            bonus_scale = trial.suggest_float(
                "bonus_scale", space["bonus_scale"][0], space["bonus_scale"][1]
            )
        else:
            bonus_scale = trial.suggest_float("alpha", space["alpha"][0], space["alpha"][1])
        online_cfg["bonus_beta"] = trial.suggest_float(
            "bonus_beta", space["bonus_beta"][0], space["bonus_beta"][1]
        )
        online_cfg["bonus_lambda"] = trial.suggest_float(
            "bonus_lambda", space["bonus_lambda"][0], space["bonus_lambda"][1]
        )
        online_cfg["q_lr"] = trial.suggest_float(
            "q_lr", space["q_lr"][0], space["q_lr"][1], log=True
        )
        online_cfg["eps_decay_steps"] = trial.suggest_int(
            "eps_decay_steps", space["eps_decay_steps"][0], space["eps_decay_steps"][1]
        )
        online_cfg["rep_update_every"] = trial.suggest_int(
            "rep_update_every", space["rep_update_every"][0], space["rep_update_every"][1]
        )
        online_cfg["update_every"] = trial.suggest_int(
            "update_every", space["update_every"][0], space["update_every"][1]
        )
        online_cfg["total_steps"] = n_steps

        seed_everything(seed_base + trial.number, deterministic=runtime.get("deterministic", True))

        out_dir = os.path.join(runtime.get("output_dir", "outputs"), "runs", "optuna_online")
        ensure_dir(out_dir)

        prune_every = int(opt_cfg.get("prune_every", max(1000, n_steps // 10)))
        prune_min_steps = int(opt_cfg.get("prune_min_steps", max(1000, n_steps // 5)))

        out_csv, _ = run_online_training(
            trial_cfg,
            env_id,
            method,
            seed_base + trial.number,
            bonus_scale,
            out_dir,
            trial=trial,
            prune_metric=metric,
            prune_every=prune_every,
            prune_min_steps=prune_min_steps,
        )
        df = pd.read_csv(out_csv)

        if metric == "final_success":
            value = float(df["success"].tail(100).mean())
        else:
            value = _auc_success(df)

        return value

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--optuna-config", default="configs/optuna_online.yaml")
    args = parser.parse_args()

    try:
        import optuna
    except Exception as e:
        raise SystemExit("Optuna is not installed. Run: pip install optuna") from e

    cfg = load_config(args.config)
    opt_cfg = load_yaml(args.optuna_config)["optuna"]

    runtime = cfg["runtime"]
    seed_everything(runtime.get("seed", 0), deterministic=runtime.get("deterministic", True))
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_torch(runtime, device)

    study_name = opt_cfg.get("study_name", f"ti_online_optuna_{time.strftime('%Y%m%d_%H%M%S')}")
    storage_path = os.path.join(runtime.get("output_dir", "outputs"), "runs", "optuna", f"{study_name}.db")
    ensure_dir(os.path.dirname(storage_path))
    storage = f"sqlite:///{storage_path}"

    pruner_kind = opt_cfg.get("pruner", "median").lower()
    if pruner_kind == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(opt_cfg.get("pruner_startup_trials", 5)),
            n_warmup_steps=int(opt_cfg.get("prune_min_steps", 0)),
            interval_steps=1,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction=opt_cfg.get("direction", "maximize"),
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    objective = objective_factory(cfg, opt_cfg)
    catch = ()
    if runtime.get("continue_on_error", False):
        catch = (RuntimeError, ValueError)
    study.optimize(objective, n_trials=int(opt_cfg.get("n_trials", 20)), catch=catch)

    best = {
        "value": float(study.best_value),
        "params": study.best_params,
        "trial": int(study.best_trial.number),
        "study_name": study.study_name,
    }
    out_dir = os.path.join(runtime.get("output_dir", "outputs"), "runs", "optuna_online")
    ensure_dir(out_dir)
    best_path = os.path.join(out_dir, "best_params.json")
    best_study_path = os.path.join(out_dir, f"best_params_{study_name}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(best_study_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Best trial:")
    print(best)
    print(f"Saved best params to: {best_path}")
    print(f"Saved best params to: {best_study_path}")


if __name__ == "__main__":
    main()
