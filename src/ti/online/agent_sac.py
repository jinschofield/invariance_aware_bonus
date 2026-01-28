import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_dim, hidden_layers):
    layers = []
    in_dim = input_dim
    for _ in range(int(hidden_layers)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    return nn.Sequential(*layers), in_dim


class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128, hidden_layers=2):
        super().__init__()
        self.trunk, trunk_dim = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.logits = nn.Linear(trunk_dim, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        return self.logits(h)


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128, hidden_layers=2):
        super().__init__()
        self.trunk, trunk_dim = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.head = nn.Linear(trunk_dim, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h)


class DiscreteSACAgent:
    def __init__(
        self,
        input_dim,
        n_actions,
        hidden_dim,
        hidden_layers,
        q_lr,
        actor_lr,
        alpha_lr,
        tau,
        entropy_alpha,
        entropy_autotune,
        target_entropy_ratio,
        alpha_max,
        max_grad_norm,
        device,
        entropy_ratio_end=None,
        entropy_anneal_steps=None,
    ):
        self.n_actions = int(n_actions)
        self.device = device
        self.tau = float(tau)
        self.autotune = bool(entropy_autotune)

        # Entropy annealing: ratio goes from target_entropy_ratio -> entropy_ratio_end
        self.entropy_ratio_start = float(target_entropy_ratio)
        self.entropy_ratio_end = float(entropy_ratio_end) if entropy_ratio_end is not None else self.entropy_ratio_start
        self.entropy_anneal_steps = int(entropy_anneal_steps) if entropy_anneal_steps is not None else 0
        self.max_entropy = float(np.log(self.n_actions))

        self.policy = CategoricalPolicy(
            input_dim, self.n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers
        ).to(device)
        self.q1 = QNetwork(
            input_dim, self.n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers
        ).to(device)
        self.q2 = QNetwork(
            input_dim, self.n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers
        ).to(device)
        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)

        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=q_lr
        )
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)

        self.target_entropy = self.max_entropy * self.entropy_ratio_start
        self.alpha_max = float(alpha_max) if alpha_max is not None else None
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm else None
        if self.autotune:
            init = float(entropy_alpha)
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(math.log(max(init, 1e-8)), device=device)
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = float(init)
        else:
            self.alpha = float(entropy_alpha)
            self.log_alpha = None
            self.alpha_opt = None

    def update_target_entropy(self, step):
        """Anneal target entropy from start ratio to end ratio over anneal_steps."""
        if self.entropy_anneal_steps <= 0 or self.entropy_ratio_start == self.entropy_ratio_end:
            return
        progress = min(1.0, float(step) / float(self.entropy_anneal_steps))
        ratio = self.entropy_ratio_start + (self.entropy_ratio_end - self.entropy_ratio_start) * progress
        self.target_entropy = self.max_entropy * ratio

    def _policy_dist(self, state):
        logits = self.policy(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            probs, _ = self._policy_dist(state)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
            if epsilon > 0.0:
                rand = torch.rand(actions.shape[0], device=actions.device)
                random_actions = torch.randint(0, self.n_actions, (actions.shape[0],), device=actions.device)
                actions = torch.where(rand < float(epsilon), random_actions, actions)
        return actions

    def _soft_update(self):
        with torch.no_grad():
            for target, src in zip(self.q1_target.parameters(), self.q1.parameters()):
                target.data.mul_(1.0 - self.tau).add_(self.tau * src.data)
            for target, src in zip(self.q2_target.parameters(), self.q2.parameters()):
                target.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

    def sync_target(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def update(self, batch, gamma, n_step=1):
        s, a, r, sp, d = batch
        a = a.long()
        r = r.float()
        d = d.float()

        with torch.no_grad():
            next_probs, next_log_probs = self._policy_dist(sp)
            q1_next = self.q1_target(sp)
            q2_next = self.q2_target(sp)
            q_next = torch.min(q1_next, q2_next)
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1)
            target = r + (1.0 - d) * (float(gamma) ** int(n_step)) * v_next

        q1 = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.max_grad_norm,
            )
        self.q_opt.step()

        probs, log_probs = self._policy_dist(s)
        q1_pi = self.q1(s)
        q2_pi = self.q2(s)
        q_pi = torch.min(q1_pi, q2_pi).detach()
        pi_loss = (probs * (self.alpha * log_probs - q_pi)).sum(dim=1).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.pi_opt.step()

        alpha_loss = None
        if self.autotune and self.log_alpha is not None:
            entropy = -(probs * log_probs).sum(dim=1)
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            if self.alpha_max is not None:
                self.log_alpha.data.clamp_(max=math.log(self.alpha_max))
                self.alpha = float(self.log_alpha.exp().clamp(max=self.alpha_max).item())
            else:
                self.alpha = float(self.log_alpha.exp().item())

        self._soft_update()

        return float(q_loss.item()), float(pi_loss.item()), float(self.alpha), (
            float(alpha_loss.item()) if alpha_loss is not None else None
        )
