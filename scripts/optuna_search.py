import argparse
import json
import os
import time

import torch

from ti.config.defaults import load_config, load_yaml
from ti.data.cache import load_buffer, save_buffer
from ti.data.collect import collect_offline_dataset
from ti.envs import make_env
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.metrics.invariance import invariance_metric_from_pairs
from ti.metrics.probes import run_linear_probe_any, run_xy_regression_probe
from ti.models.rep_methods import OfflineRepLearner
from ti.utils import configure_torch, ensure_dir, seed_everything


def _suggest_float(trial, name, bounds, log=False):
    low, high = float(bounds[0]), float(bounds[1])
    return trial.suggest_float(name, low, high, log=log)


def _make_sampler(optuna, opt_cfg, seed):
    sampler_name = str(opt_cfg.get("sampler", "tpe")).lower()
    n_startup = int(opt_cfg.get("n_startup_trials", 5))
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup, multivariate=True)


def _make_pruner(optuna, opt_cfg):
    pruner_name = str(opt_cfg.get("pruner", "median")).lower()
    n_startup = int(opt_cfg.get("pruner_startup_trials", 5))
    n_warmup = int(opt_cfg.get("pruner_warmup_steps", 1))
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    if pruner_name == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    return optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup)


def _metric_from_enc(enc, obs, y, obs_all, obs1, obs2, metric, probe_cfg, seed, device, num_classes):
    if metric == "inv":
        return float(invariance_metric_from_pairs(enc, obs1, obs2))
    if metric in ("nuis_acc", "nuis_mi"):
        acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=seed, device=device)
        return float(acc if metric == "nuis_acc" else mi)
    if metric == "xy_mse":
        return float(run_xy_regression_probe(enc, obs_all, probe_cfg, seed=seed, device=device))
    raise ValueError(f"Unknown metric: {metric}")


def objective_factory(cfg, opt_cfg):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    probe_cfg = methods_cfg["probes"]
    maze_cfg = build_maze_cfg(cfg)

    env_id = opt_cfg.get("env_id", "teacup")
    method = opt_cfg.get("method", "CRTR")
    metric = opt_cfg.get("metric", "nuis_acc")
    train_steps = int(opt_cfg.get("train_steps", methods_cfg["train"]["offline_train_steps"]))
    collect_steps = int(opt_cfg.get("collect_steps", methods_cfg["train"]["offline_collect_steps"]))
    num_envs = int(opt_cfg.get("num_envs", methods_cfg["train"]["offline_num_envs"]))
    batch_size = int(opt_cfg.get("batch_size", methods_cfg["train"]["offline_batch_size"]))
    eval_every = int(opt_cfg.get("eval_every", train_steps))
    inv_samples = int(opt_cfg.get("inv_samples", 2048))

    env_spec = get_env_spec(cfg, env_id)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _load_dataset():
        cache_path = os.path.join(runtime["cache_dir"], f"{env_id}_seed{runtime['seed']}.pt")
        if runtime.get("cache_datasets", True) and os.path.exists(cache_path):
            buf, epi = load_buffer(cache_path, device)
            env = make_env(env_spec["ctor"], num_envs=num_envs, maze_cfg=maze_cfg, device=device)
        else:
            buf, env = collect_offline_dataset(env_spec["ctor"], collect_steps, num_envs, maze_cfg, device)
            from ti.data.buffer import build_episode_index_strided

            epi = build_episode_index_strided(buf.timestep, buf.size, num_envs, device)
            if runtime.get("cache_datasets", True):
                save_buffer(cache_path, buf, epi)
        return buf, epi, env

    def objective(trial):
        seed_everything(int(runtime.get("seed", 0)) + int(trial.number), deterministic=runtime.get("deterministic", True))

        space = opt_cfg.get("search_space", {})
        z_dim = trial.suggest_categorical("z_dim", space.get("z_dim", [4, 8, 16]))
        hidden_dim = trial.suggest_categorical("hidden_dim", space.get("hidden_dim", [64, 128]))
        lr = _suggest_float(trial, "lr", space.get("lr", [1e-4, 1e-3]), log=True)
        crtr_rep = trial.suggest_categorical("crtr_rep", space.get("crtr_rep", [1, 2, 4, 8]))
        crtr_temp = _suggest_float(trial, "crtr_temp", space.get("crtr_temp", [1.0, 4.0]), log=True)
        geom_p = _suggest_float(trial, "geom_p", space.get("geom_p", [0.001, 0.05]), log=True)
        k_cap = trial.suggest_int("k_cap", space.get("k_cap", [5, 20])[0], space.get("k_cap", [5, 20])[1])

        buf, epi, env = _load_dataset()
        obs_all = buf.s[: buf.size]
        y_all = buf.nuis[: buf.size].long()

        if env_spec.get("probe_mask") == "special_only":
            mask = buf.special[: buf.size]
            obs = obs_all[mask]
            y = y_all[mask]
        else:
            obs = obs_all
            y = y_all

        num_classes = int(env_spec["classes"])
        obs1, obs2 = env.sample_invariance_pairs(inv_samples)

        if method != "CRTR":
            raise ValueError("Optuna script currently supports CRTR only.")

        learner = OfflineRepLearner(
            "CRTR",
            obs_dim=maze_cfg["obs_dim"],
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            n_actions=maze_cfg["n_actions"],
            crtr_temp=crtr_temp,
            crtr_rep=crtr_rep,
            k_cap=k_cap,
            geom_p=geom_p,
            device=device,
            lr=lr,
        ).to(device)

        steps_done = 0
        while steps_done < train_steps:
            chunk = min(eval_every, train_steps - steps_done)
            learner.train_steps(
                buf,
                epi,
                chunk,
                batch_size,
                log_every=0,
                use_amp=runtime.get("use_amp", False),
                amp_dtype=runtime.get("amp_dtype", "bf16"),
            )
            steps_done += chunk
            enc = lambda x, L=learner: L.rep_enc(x)
            value = _metric_from_enc(
                enc,
                obs,
                y,
                obs_all,
                obs1,
                obs2,
                metric,
                probe_cfg,
                runtime["seed"],
                device,
                num_classes,
            )
            trial.report(value, steps_done)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        return float(value)

    return objective


def _refit_best(cfg, opt_cfg, best_params, study_name):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    probe_cfg = methods_cfg["probes"]
    maze_cfg = build_maze_cfg(cfg)

    env_id = opt_cfg.get("env_id", "teacup")
    method = opt_cfg.get("method", "CRTR")
    metric = opt_cfg.get("metric", "nuis_acc")
    env_spec = get_env_spec(cfg, env_id)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_steps = int(opt_cfg.get("refit_train_steps", methods_cfg["train"]["offline_train_steps"]))
    collect_steps = int(opt_cfg.get("refit_collect_steps", methods_cfg["train"]["offline_collect_steps"]))
    num_envs = int(opt_cfg.get("refit_num_envs", methods_cfg["train"]["offline_num_envs"]))
    batch_size = int(opt_cfg.get("refit_batch_size", methods_cfg["train"]["offline_batch_size"]))
    inv_samples = int(opt_cfg.get("refit_inv_samples", 2048))

    if method != "CRTR":
        raise ValueError("Refit currently supports CRTR only.")

    cache_path = os.path.join(runtime["cache_dir"], f"{env_id}_seed{runtime['seed']}_refit.pt")
    if runtime.get("cache_datasets", True) and os.path.exists(cache_path):
        buf, epi = load_buffer(cache_path, device)
        env = make_env(env_spec["ctor"], num_envs=num_envs, maze_cfg=maze_cfg, device=device)
    else:
        buf, env = collect_offline_dataset(env_spec["ctor"], collect_steps, num_envs, maze_cfg, device)
        from ti.data.buffer import build_episode_index_strided

        epi = build_episode_index_strided(buf.timestep, buf.size, num_envs, device)
        if runtime.get("cache_datasets", True):
            save_buffer(cache_path, buf, epi)

    obs_all = buf.s[: buf.size]
    y_all = buf.nuis[: buf.size].long()

    if env_spec.get("probe_mask") == "special_only":
        mask = buf.special[: buf.size]
        obs = obs_all[mask]
        y = y_all[mask]
    else:
        obs = obs_all
        y = y_all

    num_classes = int(env_spec["classes"])
    obs1, obs2 = env.sample_invariance_pairs(inv_samples)

    learner = OfflineRepLearner(
        "CRTR",
        obs_dim=maze_cfg["obs_dim"],
        z_dim=int(best_params["z_dim"]),
        hidden_dim=int(best_params["hidden_dim"]),
        n_actions=maze_cfg["n_actions"],
        crtr_temp=float(best_params["crtr_temp"]),
        crtr_rep=int(best_params["crtr_rep"]),
        k_cap=int(best_params["k_cap"]),
        geom_p=float(best_params["geom_p"]),
        device=device,
        lr=float(best_params["lr"]),
    ).to(device)

    learner.train_steps(
        buf,
        epi,
        train_steps,
        batch_size,
        log_every=methods_cfg["train"]["print_train_every"],
        use_amp=runtime.get("use_amp", False),
        amp_dtype=runtime.get("amp_dtype", "bf16"),
    )

    enc = lambda x, L=learner: L.rep_enc(x)
    acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=runtime["seed"], device=device)
    inv = invariance_metric_from_pairs(enc, obs1, obs2)
    xy_mse = run_xy_regression_probe(enc, obs_all, probe_cfg, seed=runtime["seed"], device=device)

    refit = {
        "study_name": study_name,
        "env_id": env_id,
        "method": method,
        "seed": int(runtime.get("seed", 0)),
        "train_steps": train_steps,
        "collect_steps": collect_steps,
        "num_envs": num_envs,
        "batch_size": batch_size,
        "params": best_params,
        "metrics": {
            "nuis_acc": float(acc),
            "nuis_mi": float(mi),
            "inv": float(inv),
            "xy_mse": float(xy_mse),
        },
    }

    out_dir = os.path.join(runtime.get("output_dir", "outputs"), "runs", "optuna")
    ensure_dir(out_dir)
    refit_path = os.path.join(out_dir, f"{study_name}_refit.json")
    with open(refit_path, "w", encoding="utf-8") as f:
        json.dump(refit, f, indent=2)
    print(f"Saved refit metrics to: {refit_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--optuna-config", default="configs/optuna.yaml")
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
    ensure_dir(runtime.get("cache_dir", "outputs/cache"))

    study_name = opt_cfg.get("study_name", f"ti_optuna_{time.strftime('%Y%m%d_%H%M%S')}")
    storage_path = os.path.join(runtime.get("output_dir", "outputs"), "runs", "optuna", f"{study_name}.db")
    ensure_dir(os.path.dirname(storage_path))
    storage = f"sqlite:///{storage_path}"

    sampler = _make_sampler(optuna, opt_cfg, seed=runtime.get("seed", 0))
    pruner = _make_pruner(optuna, opt_cfg)
    study = optuna.create_study(
        study_name=study_name,
        direction=opt_cfg.get("direction", "minimize"),
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = objective_factory(cfg, opt_cfg)
    study.optimize(objective, n_trials=int(opt_cfg.get("n_trials", 20)), catch=(Exception,))

    best = {
        "value": float(study.best_value),
        "params": study.best_params,
        "trial": int(study.best_trial.number),
        "study_name": study.study_name,
    }
    best_path = os.path.join(os.path.dirname(storage_path), f"{study_name}_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print("Best trial:")
    print(best)
    print(f"Saved best params to: {best_path}")

    if opt_cfg.get("refit_best", False):
        _refit_best(cfg, opt_cfg, best["params"], study_name)


if __name__ == "__main__":
    main()
