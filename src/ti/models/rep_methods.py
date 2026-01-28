import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ti.data.buffer import sample_crtr_pairs_offline
from ti.models.encoders import EncoderMLP, ForwardDynamicsMLP, InverseDynamicsLinear


class OfflineRepLearner(nn.Module):
    def __init__(
        self,
        method,
        obs_dim,
        z_dim,
        hidden_dim,
        n_actions,
        crtr_temp,
        crtr_rep,
        k_cap,
        geom_p,
        device,
        lr=3e-4,
    ):
        super().__init__()
        self.method = method
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.crtr_temp = float(crtr_temp)
        self.crtr_rep = int(crtr_rep)
        self.k_cap = int(k_cap)
        self.geom_p = float(geom_p)
        self.device = device

        self.rep_enc = EncoderMLP(obs_dim, z_dim, hidden_dim).to(device)
        if method in ["IDM", "ICM"]:
            self.idm = InverseDynamicsLinear(z_dim, n_actions).to(device)
        if method == "ICM":
            self.fwd = ForwardDynamicsMLP(z_dim, hidden_dim, n_actions).to(device)
        if method == "RND":
            self.rnd_target = EncoderMLP(obs_dim, z_dim, hidden_dim).to(device)
            for p in self.rnd_target.parameters():
                p.requires_grad = False

        params = list(self.rep_enc.parameters())
        if hasattr(self, "idm"):
            params += list(self.idm.parameters())
        if hasattr(self, "fwd"):
            params += list(self.fwd.parameters())
        self.opt = optim.Adam(params, lr=lr)

    def loss(self, buf, epi, batch_size):
        if self.method == "CRTR":
            rep = int(self.crtr_rep)
            bs_eff = int(batch_size) - (int(batch_size) % rep)
            if bs_eff <= 0:
                bs_eff = rep
            s_t, s_f = sample_crtr_pairs_offline(
                buf, epi, bs_eff, rep, self.k_cap, self.geom_p, self.device
            )
            z_t = self.rep_enc(s_t)
            z_f = self.rep_enc(s_f)
            # Backward InfoNCE: anchor on z_f, predict which z_t it came from
            logits = (z_f @ z_t.T) / float(self.crtr_temp)
            return F.cross_entropy(logits, torch.arange(bs_eff, device=self.device))

        if self.method == "IDM":
            s, a, sp, _ = buf.sample(batch_size)
            z = self.rep_enc(s)
            zp = self.rep_enc(sp)
            return F.cross_entropy(self.idm(z, zp), a)

        if self.method == "ICM":
            s, a, sp, _ = buf.sample(batch_size)
            z = self.rep_enc(s)
            zp = self.rep_enc(sp)
            inv_loss = F.cross_entropy(self.idm(z, zp), a)
            pred_zp = self.fwd(z, a)
            fwd_loss = F.mse_loss(pred_zp, zp)
            return inv_loss + fwd_loss

        if self.method == "RND":
            s, _, _, _ = buf.sample(batch_size)
            z_pred = self.rep_enc(s)
            with torch.no_grad():
                z_targ = self.rnd_target(s)
            return F.mse_loss(z_pred, z_targ)

        raise ValueError(self.method)

    def train_steps(
        self,
        buf,
        epi,
        steps,
        batch_size,
        log_every,
        ckpt_dir=None,
        ckpt_every=None,
        losses_path=None,
        losses_flush_every=200,
        use_amp=False,
        amp_dtype="bf16",
        resume=False,
    ):
        self.train()
        use_amp = bool(use_amp) and self.device.type == "cuda"
        amp_dtype = str(amp_dtype).lower()
        if amp_dtype in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif amp_dtype in ("fp16", "float16"):
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")
        scaler = None
        if use_amp and dtype == torch.float16:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
            if use_amp
            else nullcontext()
        )
        logger = None
        if losses_path:
            from ti.training.logging import LossLogger

            logger = LossLogger(losses_path, flush_every=losses_flush_every)
        start_step = 0
        if resume and ckpt_dir:
            from ti.training.checkpoint import find_latest_checkpoint, load_checkpoint

            ckpt_path, step = find_latest_checkpoint(ckpt_dir, prefix="ckpt_step")
            if ckpt_path:
                payload = load_checkpoint(ckpt_path, model=self, optimizer=self.opt, map_location=self.device)
                start_step = int(payload.get("step", step))
        if start_step >= int(steps):
            if logger is not None:
                logger.flush()
            return
        for t in range(start_step, int(steps)):
            with autocast_ctx:
                loss = self.loss(buf, epi, batch_size)
            self.opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            else:
                loss.backward()
                self.opt.step()
            if logger is not None:
                logger.log(t + 1, float(loss.item()))
            if log_every and ((t + 1) % int(log_every) == 0):
                print(
                    f"    train step {t+1:>6}/{int(steps)} | loss={float(loss.item()):.4f}",
                    flush=True,
                )
            if ckpt_dir and ckpt_every and ((t + 1) % int(ckpt_every) == 0):
                from ti.training.checkpoint import save_checkpoint

                save_checkpoint(
                    os.path.join(ckpt_dir, f"ckpt_step{t+1:06d}.pt"),
                    self,
                    optimizer=self.opt,
                    step=t + 1,
                    extra={"method": self.method},
                )
        if logger is not None:
            logger.flush()
