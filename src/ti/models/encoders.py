import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(7):  # 7 more hidden layers (8 total)
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1, eps=1e-8)


class InverseDynamicsLinear(nn.Module):
    def __init__(self, z_dim, n_actions):
        super().__init__()
        self.fc = nn.Linear(2 * z_dim, n_actions)

    def forward(self, z, zp):
        return self.fc(torch.cat([z, zp], dim=-1))


class ForwardDynamicsMLP(nn.Module):
    def __init__(self, z_dim, hidden, n_actions):
        super().__init__()
        self.n_actions = int(n_actions)
        self.net = nn.Sequential(
            nn.Linear(z_dim + self.n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, a):
        a1 = F.one_hot(a, num_classes=self.n_actions).float()
        return self.net(torch.cat([z, a1], dim=-1))


class BiscuitEncoder(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden), nn.SiLU()]
        for _ in range(7):  # 7 more hidden layers (8 total)
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.mu(h), self.logvar(h)


class BiscuitDecoder(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class ActionEmbed(nn.Module):
    def __init__(self, n_actions, a_dim=8):
        super().__init__()
        self.emb = nn.Embedding(n_actions, a_dim)

    def forward(self, a):
        return self.emb(a.long())
