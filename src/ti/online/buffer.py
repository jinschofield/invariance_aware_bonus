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
        self.ptr = 0  # Write pointer for circular buffer
        self.wrapped = False  # Track if buffer has wrapped

    def add_batch(self, s, a, r, sp, done):
        b = int(s.shape[0])
        # Circular buffer: wrap around when full, but maintain strided layout
        # Each env writes at ptr + env_id, then ptr advances by num_envs
        end_ptr = self.ptr + b
        if end_ptr <= self.max_size:
            # No wrap needed
            idx = torch.arange(self.ptr, end_ptr, device=self.device)
            self.s[idx] = s
            self.a[idx] = a
            self.r[idx] = r
            self.sp[idx] = sp
            self.d[idx] = done
            self.timestep[idx] = self.current_timestep
            self.ptr = end_ptr % self.max_size
        else:
            # Wrap around
            self.wrapped = True
            first_part = self.max_size - self.ptr
            idx1 = torch.arange(self.ptr, self.max_size, device=self.device)
            idx2 = torch.arange(0, b - first_part, device=self.device)
            self.s[idx1] = s[:first_part]
            self.s[idx2] = s[first_part:]
            self.a[idx1] = a[:first_part]
            self.a[idx2] = a[first_part:]
            self.r[idx1] = r[:first_part]
            self.r[idx2] = r[first_part:]
            self.sp[idx1] = sp[:first_part]
            self.sp[idx2] = sp[first_part:]
            self.d[idx1] = done[:first_part]
            self.d[idx2] = done[first_part:]
            self.timestep[idx1] = self.current_timestep[:first_part]
            self.timestep[idx2] = self.current_timestep[first_part:]
            self.ptr = b - first_part
        self.size = min(self.size + b, self.max_size)
        self.current_timestep = self.current_timestep + 1
        self.current_timestep[done] = 0

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.sp[idx], self.d[idx]

    def sample_with_reward(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.d[idx]

    def sample_nstep(self, batch_size, n_step, gamma):
        n_step = int(n_step)
        if n_step <= 1:
            s, a, r, sp, d = self.sample_with_reward(batch_size)
            return s, a, r, sp, d, 1
        # Fall back to 1-step if buffer has wrapped (n-step logic assumes contiguous storage)
        if self.wrapped:
            s, a, r, sp, d = self.sample_with_reward(batch_size)
            return s, a, r, sp, d, 1
        nenv = int(self.num_envs)
        max_base = self.size - n_step * nenv
        if max_base <= 0:
            s, a, r, sp, d = self.sample_with_reward(batch_size)
            return s, a, r, sp, d, 1

        collected = []
        collected_count = 0
        tries = 0
        while collected_count < batch_size and tries < 10:
            tries += 1
            cand = torch.randint(0, max_base, (batch_size * 2,), device=self.device)
            t0 = self.timestep[cand]
            valid = torch.ones_like(cand, dtype=torch.bool)
            for k in range(n_step):
                idx_k = cand + k * nenv
                valid &= idx_k < self.size
                valid &= self.timestep[idx_k] == (t0 + k)
                if k < n_step - 1:
                    valid &= ~self.d[idx_k]
            valid_idx = cand[valid]
            if valid_idx.numel() > 0:
                collected.append(valid_idx)
                collected_count += int(valid_idx.numel())

        if collected_count < batch_size:
            s, a, r, sp, d = self.sample_with_reward(batch_size)
            return s, a, r, sp, d, 1

        base = torch.cat(collected, dim=0)[:batch_size]

        s = self.s[base]
        a = self.a[base]

        returns = torch.zeros((base.shape[0],), device=self.device)
        not_done = torch.ones((base.shape[0],), device=self.device, dtype=torch.bool)
        last_idx = base.clone()
        discount = 1.0
        for k in range(n_step):
            idx_k = base + k * nenv
            r_k = self.r[idx_k]
            returns += discount * r_k * not_done.float()
            done_k = self.d[idx_k] & not_done
            last_idx = torch.where(done_k, idx_k, last_idx)
            not_done = not_done & ~done_k
            discount *= gamma

        last_idx = torch.where(not_done, base + (n_step - 1) * nenv, last_idx)
        sp = self.sp[last_idx]
        d = (~not_done).to(self.d.dtype)
        return s, a, returns, sp, d, n_step
