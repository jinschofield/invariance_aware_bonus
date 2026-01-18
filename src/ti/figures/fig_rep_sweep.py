import os

import pandas as pd
import torch
from ti.data.cache import load_or_build_episode_index
from ti.data.collect import collect_offline_dataset
from ti.data.cache import load_buffer, save_buffer
from ti.envs import make_env
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.metrics.invariance import invariance_metric_from_pairs, sample_delayed_pairs_for_slippery
from ti.metrics.probes import action_probe_from_pairs, run_linear_probe_any, run_xy_regression_probe
from ti.models.biscuit import BISCUIT_VAE
from ti.models.rep_methods import OfflineRepLearner
from ti.plotting.plots import plot_env_bars
from ti.training.cbm_train import ancestors_in_dyn_graph, train_cbm_models
from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint
from ti.utils import ensure_dir, get_amp_settings, maybe_compile


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    train_cfg = methods_cfg["train"]
    probe_cfg = methods_cfg["probes"]
    maze_cfg = build_maze_cfg(cfg)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], fig_id)
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    use_amp, amp_dtype, scaler = get_amp_settings(runtime, device)

    env_ids = fig_spec.get("envs", [])
    crtr_rep_list = fig_spec.get("crtr_rep_list", methods_cfg["crtr_rep_list"])
    inv_samples = int(fig_spec.get("inv_samples", 2048))

    all_results = []

    for env_id in env_ids:
        env_spec = get_env_spec(cfg, env_id)
        env_name = env_spec["name"]
        print("\n" + "=" * 92)
        print(f"ENV = {env_name}")
        print("=" * 92)

        print("Collecting offline dataset...", flush=True)
        cache_path = os.path.join(runtime["cache_dir"], f"{env_id}_seed{runtime['seed']}.pt")
        if runtime.get("cache_datasets", True) and os.path.exists(cache_path):
            buf, epi = load_buffer(cache_path, device)
            env = make_env(env_spec["ctor"], num_envs=train_cfg["offline_num_envs"], maze_cfg=maze_cfg, device=device)
        else:
            buf, env = collect_offline_dataset(
                env_spec["ctor"],
                train_cfg["offline_collect_steps"],
                train_cfg["offline_num_envs"],
                maze_cfg,
                device,
            )
            epi = load_or_build_episode_index(buf, None, device)
            if runtime.get("cache_datasets", True):
                save_buffer(cache_path, buf, epi)

        obs_all = buf.s[: buf.size]
        y_all = buf.nuis[: buf.size].long()

        if env_spec.get("probe_mask") == "special_only":
            mask = buf.special[: buf.size]
            obs = obs_all[mask]
            y = y_all[mask]
            print(
                f"  Teacups special fraction = {float(mask.float().mean().item()):.3f} "
                f"({int(mask.sum())}/{int(mask.numel())})",
                flush=True,
            )
            if obs.shape[0] < 1000:
                print(f"  WARNING: few special samples: {obs.shape[0]}.", flush=True)
            if obs.shape[0] < 2:
                raise RuntimeError("Teacups produced too few special samples to run probes.")
        else:
            obs = obs_all
            y = y_all

        num_classes = int(env_spec["classes"])
        chance = 1.0 / float(num_classes)

        obs1, obs2 = env.sample_invariance_pairs(inv_samples)
        with torch.no_grad():
            same_xy = (obs1[:, :2] - obs2[:, :2]).abs().max().item()
        if same_xy > 1e-6:
            print(f"  WARNING: invariance pairs XY mismatch max={same_xy:.2e}", flush=True)

        env_rows = []

        def log_row(method_name, nuisance_acc, nuisance_mi, inv, xy_mse, extra=None):
            row = {
                "env": env_name,
                "method": method_name,
                "seed": int(runtime["seed"]),
                "nuis_acc": float(nuisance_acc),
                "nuis_mi": float(nuisance_mi),
                "inv": float(inv),
                "xy_mse": float(xy_mse),
            }
            if extra is not None:
                row.update(extra)
            all_results.append(row)
            env_rows.append(row)

        for r in crtr_rep_list:
            name = f"CRTR_R{r}"
            print(f"\n>>> {name}", flush=True)
            learner = OfflineRepLearner(
                "CRTR",
                obs_dim=maze_cfg["obs_dim"],
                z_dim=methods_cfg["model"]["z_dim"],
                hidden_dim=methods_cfg["model"]["hidden_dim"],
                n_actions=maze_cfg["n_actions"],
                crtr_temp=methods_cfg["model"]["crtr_temp"],
                crtr_rep=r,
                k_cap=methods_cfg["model"]["k_cap"],
                geom_p=methods_cfg["model"]["geom_p"],
                device=device,
                lr=methods_cfg["model"]["lr"],
            ).to(device)
            if runtime.get("compile", False):
                learner = maybe_compile(learner, runtime)

            print("  train:", flush=True)
            ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, name)
            losses_path = os.path.join(runtime["log_dir"], fig_id, env_id, f"{name}_loss.csv")
            ensure_dir(ckpt_dir)
            ensure_dir(os.path.dirname(losses_path))
            learner.train_steps(
                buf,
                epi,
                train_cfg["offline_train_steps"],
                train_cfg["offline_batch_size"],
                train_cfg["print_train_every"],
                ckpt_dir=ckpt_dir if runtime.get("checkpoint_every") else None,
                ckpt_every=runtime.get("checkpoint_every"),
                losses_path=losses_path,
                losses_flush_every=runtime.get("losses_flush_every", 200),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                resume=runtime.get("resume", True),
            )

            enc = lambda x, L=learner: L.rep_enc(x)

            print("  probe/eval:", flush=True)
            acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=runtime["seed"], device=device)
            inv = invariance_metric_from_pairs(enc, obs1, obs2)
            xy_mse = run_xy_regression_probe(enc, obs_all, probe_cfg, seed=runtime["seed"], device=device)

            print(
                f"  done | nuis_acc={acc:.3f} (chance={chance:.3f}) | MI={mi:.3f} | "
                f"inv={inv:.4f} | xy_mse={xy_mse:.4f}",
                flush=True,
            )
            log_row(name, acc, mi, inv, xy_mse)

        for method in ["IDM", "ICM", "RND"]:
            print(f"\n>>> {method}", flush=True)
            learner = OfflineRepLearner(
                method,
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
            if runtime.get("compile", False):
                learner = maybe_compile(learner, runtime)

            print("  train:", flush=True)
            ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, method)
            losses_path = os.path.join(runtime["log_dir"], fig_id, env_id, f"{method}_loss.csv")
            ensure_dir(ckpt_dir)
            ensure_dir(os.path.dirname(losses_path))
            learner.train_steps(
                buf,
                epi,
                train_cfg["offline_train_steps"],
                train_cfg["offline_batch_size"],
                train_cfg["print_train_every"],
                ckpt_dir=ckpt_dir if runtime.get("checkpoint_every") else None,
                ckpt_every=runtime.get("checkpoint_every"),
                losses_path=losses_path,
                losses_flush_every=runtime.get("losses_flush_every", 200),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                resume=runtime.get("resume", True),
            )

            enc = lambda x, L=learner: L.rep_enc(x)

            print("  probe/eval:", flush=True)
            acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=runtime["seed"], device=device)
            inv = invariance_metric_from_pairs(enc, obs1, obs2)
            xy_mse = run_xy_regression_probe(enc, obs_all, probe_cfg, seed=runtime["seed"], device=device)
            print(
                f"  done | nuis_acc={acc:.3f} (chance={chance:.3f}) | MI={mi:.3f} | "
                f"inv={inv:.4f} | xy_mse={xy_mse:.4f}",
                flush=True,
            )

            extra = {}
            if env_spec["ctor"] == "SlipperyDelayMaze":
                pairs = sample_delayed_pairs_for_slippery(
                    buf,
                    delay_steps=maze_cfg["slippery_D"],
                    num_envs=train_cfg["offline_num_envs"],
                    device=device,
                )
                if pairs is not None:
                    sD, spD, a0 = pairs
                    delay_acc = action_probe_from_pairs(
                        enc, sD, spD, a0, probe_cfg, seed=runtime["seed"], n_actions=maze_cfg["n_actions"], device=device
                    )
                    extra["delay_ID_acc"] = float(delay_acc)
                    print(f"       (slippery extra) delay_ID_acc={delay_acc:.3f}", flush=True)
                else:
                    extra["delay_ID_acc"] = float("nan")

            log_row(method, acc, mi, inv, xy_mse, extra=extra)

        print("\n>>> BISCUIT", flush=True)
        model = BISCUIT_VAE(
            obs_dim=maze_cfg["obs_dim"],
            z_dim=methods_cfg["model"]["z_dim"],
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

        biscuit_losses = []
        ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, "BISCUIT")
        start_step = 0
        if runtime.get("resume", True) and os.path.isdir(ckpt_dir):
            ckpt_path, step = find_latest_checkpoint(ckpt_dir, prefix="ckpt_step")
            if ckpt_path:
                payload = load_checkpoint(ckpt_path, model, optimizer=model.opt, map_location=device)
                start_step = int(payload.get("step", step))
        for t in range(start_step, train_cfg["offline_train_steps"]):
            s, a, sp, _ = buf.sample(train_cfg["offline_batch_size"])
            loss = model.train_step(
                s,
                a,
                sp,
                t,
                train_cfg["offline_train_steps"],
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            biscuit_losses.append((t + 1, float(loss.item())))
            if train_cfg["print_train_every"] and ((t + 1) % train_cfg["print_train_every"] == 0):
                print(
                    f"    BISCUIT step {t+1:>6}/{train_cfg['offline_train_steps']} | loss={float(loss.item()):.4f}",
                    flush=True,
                )
            if runtime.get("checkpoint_every") and ((t + 1) % int(runtime["checkpoint_every"]) == 0):
                from ti.training.checkpoint import save_checkpoint

                ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, "BISCUIT")
                ensure_dir(ckpt_dir)
                save_checkpoint(
                    os.path.join(ckpt_dir, f"ckpt_step{t+1:06d}.pt"),
                    model,
                    optimizer=model.opt,
                    step=t + 1,
                )

        from ti.training.logging import append_rows
        losses_path = os.path.join(runtime["log_dir"], fig_id, env_id, "BISCUIT_loss.csv")
        append_rows(losses_path, biscuit_losses, header=("step", "loss"))
        if runtime.get("checkpoint_every") is not None:
            from ti.training.checkpoint import save_checkpoint

            ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, "BISCUIT")
            ensure_dir(ckpt_dir)
            save_checkpoint(
                os.path.join(ckpt_dir, "ckpt_final.pt"),
                model,
                optimizer=model.opt,
                step=train_cfg["offline_train_steps"],
            )

        enc = lambda x, M=model: M.encode_mean(x)
        acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=runtime["seed"], device=device)
        inv = invariance_metric_from_pairs(enc, obs1, obs2)
        xy_mse = run_xy_regression_probe(enc, obs_all, probe_cfg, seed=runtime["seed"], device=device)
        print(
            f"  done | nuis_acc={acc:.3f} (chance={chance:.3f}) | MI={mi:.3f} | "
            f"inv={inv:.4f} | xy_mse={xy_mse:.4f}",
            flush=True,
        )

        extra = {}
        if env_spec["ctor"] == "SlipperyDelayMaze":
            pairs = sample_delayed_pairs_for_slippery(
                buf,
                delay_steps=maze_cfg["slippery_D"],
                num_envs=train_cfg["offline_num_envs"],
                device=device,
            )
            if pairs is not None:
                sD, spD, a0 = pairs
                delay_acc = action_probe_from_pairs(
                    enc, sD, spD, a0, probe_cfg, seed=runtime["seed"], n_actions=maze_cfg["n_actions"], device=device
                )
                extra["delay_ID_acc"] = float(delay_acc)
                print(f"       (slippery extra) delay_ID_acc={delay_acc:.3f}", flush=True)
            else:
                extra["delay_ID_acc"] = float("nan")

        log_row("BISCUIT", acc, mi, inv, xy_mse, extra=extra)

        print("\n>>> CBM", flush=True)
        ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, "CBM")
        log_dir = os.path.join(runtime["log_dir"], fig_id, env_id, "CBM")
        ensure_dir(ckpt_dir)
        ensure_dir(log_dir)
        dyn, rew, cmi_dyn, cmi_rew, G_dyn, PR = train_cbm_models(
            buf,
            obs_dim=maze_cfg["obs_dim"],
            n_actions=maze_cfg["n_actions"],
            maze_size=maze_cfg["maze_size"],
            goal=maze_cfg["goal"],
            steps=train_cfg["offline_train_steps"],
            batch=train_cfg["offline_batch_size"],
            n_neg=methods_cfg["model"]["cbm"]["n_neg"],
            eps_cmi=methods_cfg["model"]["cbm"]["eps_cmi"],
            lam1=methods_cfg["model"]["cbm"]["lam1"],
            lam2=methods_cfg["model"]["cbm"]["lam2"],
            lr=methods_cfg["model"]["cbm"]["lr"],
            device=device,
            print_every=train_cfg["print_train_every"],
            ckpt_dir=ckpt_dir if runtime.get("checkpoint_every") else None,
            ckpt_every=runtime.get("checkpoint_every"),
            log_dir=log_dir,
            losses_flush_every=runtime.get("losses_flush_every", 200),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume=runtime.get("resume", True),
        )
        if runtime.get("checkpoint_every") is not None:
            from ti.training.checkpoint import save_checkpoint

            save_checkpoint(
                os.path.join(ckpt_dir, "cbm_dyn_final.pt"),
                dyn,
                step=train_cfg["offline_train_steps"],
                extra={"component": "dyn"},
            )
            save_checkpoint(
                os.path.join(ckpt_dir, "cbm_rew_final.pt"),
                rew,
                step=train_cfg["offline_train_steps"],
                extra={"component": "rew"},
            )
        keep_state = ancestors_in_dyn_graph(G_dyn, PR, obs_dim=maze_cfg["obs_dim"], device=device)
        enc = lambda x, ks=keep_state: x[:, ks]

        acc, mi = run_linear_probe_any(enc, obs, y, num_classes, probe_cfg, seed=runtime["seed"], device=device)
        inv = invariance_metric_from_pairs(enc, obs1, obs2)
        xy_mse = run_xy_regression_probe(enc, obs_all, probe_cfg, seed=runtime["seed"], device=device)

        print(
            f"  done | nuis_acc={acc:.3f} (chance={chance:.3f}) | MI={mi:.3f} | "
            f"inv={inv:.4f} | xy_mse={xy_mse:.4f} | abstraction={int(keep_state.sum().item())}/{maze_cfg['obs_dim']}",
            flush=True,
        )

        extra = {"abstraction_size": int(keep_state.sum().item())}
        if env_spec["ctor"] == "SlipperyDelayMaze":
            pairs = sample_delayed_pairs_for_slippery(
                buf,
                delay_steps=maze_cfg["slippery_D"],
                num_envs=train_cfg["offline_num_envs"],
                device=device,
            )
            if pairs is not None:
                sD, spD, a0 = pairs
                delay_acc = action_probe_from_pairs(
                    enc, sD, spD, a0, probe_cfg, seed=runtime["seed"], n_actions=maze_cfg["n_actions"], device=device
                )
                extra["delay_ID_acc"] = float(delay_acc)
                print(f"       (slippery extra) delay_ID_acc={delay_acc:.3f}", flush=True)
            else:
                extra["delay_ID_acc"] = float("nan")

        log_row("CBM", acc, mi, inv, xy_mse, extra=extra)

        plot_env_bars(
            env_name,
            env_rows,
            "nuis_acc",
            "Nuisance probe acc (low=better)",
            chance_line=chance,
            save_path=[
                os.path.join(fig_dir, f"{env_id}_nuis_acc.png"),
                os.path.join(fig_dir, f"{env_id}_nuis_acc.pdf"),
            ],
        )
        plot_env_bars(
            env_name,
            env_rows,
            "nuis_mi",
            "MI proxy (low=better)",
            save_path=[
                os.path.join(fig_dir, f"{env_id}_nuis_mi.png"),
                os.path.join(fig_dir, f"{env_id}_nuis_mi.pdf"),
            ],
        )
        plot_env_bars(
            env_name,
            env_rows,
            "inv",
            "Geometric invariance (low=better)",
            save_path=[
                os.path.join(fig_dir, f"{env_id}_inv.png"),
                os.path.join(fig_dir, f"{env_id}_inv.pdf"),
            ],
        )
        plot_env_bars(
            env_name,
            env_rows,
            "xy_mse",
            "XY regression MSE (low=better)",
            save_path=[
                os.path.join(fig_dir, f"{env_id}_xy_mse.png"),
                os.path.join(fig_dir, f"{env_id}_xy_mse.pdf"),
            ],
        )

        if env_spec["ctor"] == "SlipperyDelayMaze":
            rows2 = [r for r in env_rows if "delay_ID_acc" in r]
            if rows2:
                plot_env_bars(
                    env_name,
                    rows2,
                    "delay_ID_acc",
                    "Delayed inverse dynamics probe acc (high=better)",
                    save_path=[
                        os.path.join(fig_dir, f"{env_id}_delay_id.png"),
                        os.path.join(fig_dir, f"{env_id}_delay_id.pdf"),
                    ],
                    higher_better=True,
                )

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(table_dir, "rep_sweep_results.csv"), index=False)
    return all_results
