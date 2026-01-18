import os

import pandas as pd
import torch

from ti.data.buffer import build_episode_index_strided
from ti.data.cache import load_buffer, save_buffer
from ti.data.collect import collect_offline_dataset
from ti.envs import layouts
from ti.figures.helpers import build_maze_cfg, get_env_spec
from ti.metrics.elliptical import (
    build_precision_A_from_buffer,
    compute_heat_and_sensitivity,
    inv_from_A,
    nuisance_predictability_from_bonus,
    scalar_scores_from_heat,
    teacup_inside_mask_grid,
)
from ti.models.biscuit import BISCUIT_VAE
from ti.models.rep_methods import OfflineRepLearner
from ti.training.cbm_train import ancestors_in_dyn_graph, train_cbm_models
from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint
from ti.utils import ensure_dir, get_amp_settings, maybe_compile


def run(cfg, fig_id, fig_spec):
    runtime = cfg["runtime"]
    methods_cfg = cfg["methods"]
    train_cfg = methods_cfg["train"]
    maze_cfg = build_maze_cfg(cfg)
    device = torch.device(runtime.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    fig_dir = os.path.join(runtime["fig_dir"], fig_id)
    table_dir = os.path.join(runtime["table_dir"], fig_id)
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    use_amp, amp_dtype, scaler = get_amp_settings(runtime, device)

    env_id = fig_spec.get("envs", ["teacup"])[0]
    env_spec = get_env_spec(cfg, env_id)
    env_name = env_spec["name"]

    crtr_rep_list = fig_spec.get("crtr_rep_list", methods_cfg["crtr_rep_list"])
    heat_cfg = methods_cfg["elliptical"]
    bonus_probe_cfg = methods_cfg["bonus_probe"]
    bonus_probe_cfg = {**bonus_probe_cfg, "seed": runtime["seed"]}

    print("\n" + "=" * 92)
    print(f"ELLIPTICAL BONUS + SCALARS | ENV = {env_name}  (ONLY)")
    print("=" * 92)

    print("Collecting offline dataset (once)...", flush=True)
    cache_path = os.path.join(runtime["cache_dir"], f"{env_id}_seed{runtime['seed']}.pt")
    if runtime.get("cache_datasets", True) and os.path.exists(cache_path):
        buf, epi = load_buffer(cache_path, device)
    else:
        buf, _env = collect_offline_dataset(
            env_spec["ctor"],
            train_cfg["offline_collect_steps"],
            train_cfg["offline_num_envs"],
            maze_cfg,
            device,
        )
        epi = build_episode_index_strided(buf.timestep, buf.size, train_cfg["offline_num_envs"], device)
        if runtime.get("cache_datasets", True):
            save_buffer(cache_path, buf, epi)

    methods = [(f"CRTR_R{r}", ("CRTR", r)) for r in crtr_rep_list] + [
        (m, (m, None)) for m in ["IDM", "ICM", "RND"]
    ] + [("BISCUIT", ("BISCUIT", None)), ("CBM", ("CBM", None))]

    teacup_inside_mask = teacup_inside_mask_grid(maze_cfg["maze_size"], device)
    rows = []

    for mname, (mkind, rep) in methods:
        print(f"\n>>> {mname}", flush=True)

        if mkind == "CRTR":
            learner = OfflineRepLearner(
                "CRTR",
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
            if runtime.get("compile", False):
                learner = maybe_compile(learner, runtime)
            ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, mname)
            losses_path = os.path.join(runtime["log_dir"], fig_id, env_id, f"{mname}_loss.csv")
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
            enc_fn = lambda x, L=learner: L.rep_enc(x)

        elif mkind in ["IDM", "ICM", "RND"]:
            learner = OfflineRepLearner(
                mkind,
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
            ckpt_dir = os.path.join(runtime["ckpt_dir"], fig_id, env_id, mname)
            losses_path = os.path.join(runtime["log_dir"], fig_id, env_id, f"{mname}_loss.csv")
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
            enc_fn = lambda x, L=learner: L.rep_enc(x)

        elif mkind == "BISCUIT":
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
            enc_fn = lambda x, M=model: M.encode_mean(x)

        elif mkind == "CBM":
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
            enc_fn = lambda x, ks=keep_state: x[:, ks]

        else:
            raise ValueError(mkind)

        A = build_precision_A_from_buffer(
            buf,
            enc_fn,
            n_actions=maze_cfg["n_actions"],
            lam=heat_cfg["lambda"],
            max_samples=heat_cfg["max_samples_for_A"],
            device=device,
        )
        Ainv = inv_from_A(A)

        heat_mean, heat_std = compute_heat_and_sensitivity(
            env_spec["ctor"], enc_fn, Ainv, maze_cfg, heat_cfg, device
        )

        bacc, bmi = nuisance_predictability_from_bonus(
            env_spec["ctor"], enc_fn, Ainv, maze_cfg, bonus_probe_cfg, heat_cfg, device
        )

        layout = layouts.make_layout(maze_cfg["maze_size"], device)
        scal = scalar_scores_from_heat(
            env_spec["ctor"],
            heat_mean,
            heat_std,
            layout=layout,
            teacup_inside_mask=teacup_inside_mask,
        )

        scal["env"] = env_name
        scal["method"] = mname
        scal["seed"] = int(runtime["seed"])
        scal["bonus_nuis_acc"] = float(bacc)
        scal["bonus_nuis_mi"] = float(bmi)
        rows.append(scal)

        print(f"  heat_std_mean (LOW)           = {scal['heat_std_mean']:.6f}")
        print(f"  orbit_ratio W/(B+eps) (LOW)   = {scal['orbit_ratio_W_over_B']:.6f}")
        print(f"  bonus→nuis acc/MI (LOW)       = {bacc:.4f} / {bmi:.4f}")
        print(f"  cup contrast in-out (HIGH)    = {scal.get('cup_contrast_in_minus_out', float('nan')):.6f}")
        print(f"  inside_std_mean (LOW)         = {scal.get('inside_std_mean', float('nan')):.6f}")
        print(f"  inside_rel_std_mean (LOW)     = {scal.get('inside_rel_std_mean', float('nan')):.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(table_dir, "teacup_elliptical_scalars.csv"), index=False)
    return rows
