import torch


class OnlineReplayBuffer:
    def __init__(self, obs_dim, max_size, num_envs, device):
        self.obs_dim = int(obs_dim)
        self.max_size = int(max_size)
        self.num_envs = int(num_envs)
        self.device = device

        self.s = torch.empty((self.max_size, self.obs_dim), device=device)
        self.sp = torch.empty((self.max_size, self.obs_dim), device=device)
        self.a = torch.empty((self.max_size,), device=device, dtype=torch.long)
        self.r = torch.empty((self.max_size,), device=device)
        self.d = torch.empty((self.max_size,), device=device, dtype=torch.bool)
        self.timestep = torch.empty((self.max_size,), device=device, dtype=torch.long)
        self.current_timestep = torch.zeros((self.num_envs,), device=device, dtype=torch.long)

        self.size = 0

    def add_batch(self, s, a, r, sp, done):
        b = int(s.shape[0])
        end = self.size + b
        if end > self.max_size:
            raise RuntimeError("buffer overflow")
        idx = torch.arange(self.size, end, device=self.device)
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.sp[idx] = sp
        self.d[idx] = done
        self.timestep[idx] = self.current_timestep
        self.current_timestep = self.current_timestep + 1
        self.current_timestep[done] = 0
        self.size = end

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.sp[idx], self.d[idx]

    def sample_with_reward(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]
