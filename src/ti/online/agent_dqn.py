import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, input_dim, n_actions, hidden_dim, lr, device):
        self.n_actions = int(n_actions)
        self.device = device

        self.q = QNetwork(input_dim, self.n_actions, hidden_dim=hidden_dim).to(device)
        self.q_target = copy.deepcopy(self.q).to(device)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    def act(self, z, epsilon):
        if torch.rand((), device=z.device) < float(epsilon):
            return torch.randint(0, self.n_actions, (z.shape[0],), device=z.device)
        with torch.no_grad():
            q = self.q(z)
        return q.argmax(dim=-1)

    def update(self, batch, gamma):
        s, a, r, sp, d = batch
        q = self.q(s)
        q_a = q.gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(sp).max(dim=1).values
            target = r + (1.0 - d.float()) * gamma * q_next
        loss = F.mse_loss(q_a, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
