import torch


class EpisodicEllipticalBonus:
    def __init__(self, z_dim, n_actions, beta, lam, num_envs, device):
        self.z_dim = int(z_dim)
        self.n_actions = int(n_actions)
        self.beta = float(beta)
        self.lam = float(lam)
        self.num_envs = int(num_envs)
        self.device = device

        self.d = self.z_dim + self.n_actions
        inv_lam = 1.0 / max(self.lam, 1e-8)
        eye = torch.eye(self.d, device=device) * inv_lam
        self.Ainv = eye.unsqueeze(0).repeat(self.num_envs, 1, 1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        inv_lam = 1.0 / max(self.lam, 1e-8)
        self.Ainv[env_ids] = torch.eye(self.d, device=self.device) * inv_lam

    def _phi(self, z, a):
        a_onehot = torch.nn.functional.one_hot(a.long(), num_classes=self.n_actions).float()
        return torch.cat([z, a_onehot], dim=-1)

    def compute_and_update(self, z, a):
        phi = self._phi(z, a)
        phi_col = phi.unsqueeze(-1)
        Ainv_phi = torch.bmm(self.Ainv, phi_col)
        v = torch.bmm(phi.unsqueeze(1), Ainv_phi).squeeze(-1).squeeze(-1)
        bonus = self.beta * torch.sqrt(v.clamp_min(1e-12))

        denom = (1.0 + v).view(-1, 1, 1)
        self.Ainv = self.Ainv - torch.bmm(Ainv_phi, Ainv_phi.transpose(1, 2)) / denom
        return bonus
