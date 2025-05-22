# algorithms/dueling_ddqn_prioritized.py
import math, random, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from algorithms.base_trainer import BaseTrainer
from core import device

# ======= Prioritized Replay Buffer =======
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-5):
        self.capacity = capacity
        self.alpha    = alpha
        self.eps      = eps
        self.data     = []
        self.prior    = []
        self.pos      = 0
        self.max_prior = 1.0

    def __len__(self): return len(self.data)

    def _probabilities(self):
        p = np.array(self.prior, dtype=np.float32) ** self.alpha
        p_sum = p.sum()
        return p / p_sum if p_sum > 0 else np.ones_like(p) / len(p)

    def push(self, s, a, r, ns, d):
        entry = (s, a, r, ns, d)
        if len(self.data) < self.capacity:
            self.data.append(entry)
            self.prior.append(self.max_prior)
        else:
            idx = self.pos % self.capacity
            self.data[idx] = entry
            self.prior[idx] = self.max_prior
        self.pos += 1

    def sample(self, batch, beta=0.4):
        probs = self._probabilities()
        idxs = np.random.choice(len(self.data), batch, p=probs)
        samples = [self.data[i] for i in idxs]

        weights = (len(self.data) * probs[idxs]) ** (-beta)
        weights /= weights.max()

        S, A, R, NS, D = zip(*samples)
        return (
            torch.stack(S ).to(device),
            torch.tensor(A, dtype=torch.long,  device=device).unsqueeze(1),
            torch.tensor(R, dtype=torch.float, device=device),
            torch.stack(NS).to(device),
            torch.tensor(D, dtype=torch.float, device=device),
            idxs,
            torch.tensor(weights, dtype=torch.float, device=device)
        )

    def update_priorities(self, idxs, td_errors):
        for i, err in zip(idxs, td_errors):
            p = float(abs(err) + self.eps)
            self.prior[i] = p
            self.max_prior = max(self.max_prior, p)

# ======= Dueling CNN =======
class DuelingCNN(nn.Module):
    def __init__(self, in_ch:int, n_act:int, view:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten())
        feat = 128 * view * view
        self.val = nn.Sequential(nn.Linear(feat,256), nn.ReLU(), nn.Linear(256,1))
        self.adv = nn.Sequential(nn.Linear(feat,256), nn.ReLU(), nn.Linear(256, n_act))

    def forward(self, x):
        f = self.conv(x)
        v = self.val(f)
        a = self.adv(f)
        return v + a - a.mean(1, keepdim=True)

# ======= Dueling DDQN + PER Trainer =======
class DuelingDDQNPrioritizedTrainer(BaseTrainer):
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)

        self.online = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        self.target = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.replay = PrioritizedReplayBuffer(
            self.memory_size, alpha=cfg.get("PER_ALPHA", 0.6)
        )
        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)

        self.beta_start  = cfg.get("PER_BETA_START", 0.4)
        self.beta_frames = cfg.get("PER_BETA_FRAMES", cfg["NUM_EPISODES"] * self.max_steps)
        self.frame_idx   = 0
        self.beta        = self.beta_start

        self.epsilon = self.eps_start

    def select_actions(self, views, mask):
        B, N, C, V, _ = views.shape
        flat = views.view(B * N, C, V, V)
        with torch.no_grad():
            q = self.online(flat)
        greedy = q.argmax(1).view(B, N)

        rand = torch.randint(0, 4, (B, N), device=self.device)
        eps_mask = (torch.rand(B, N, device=self.device) < self.epsilon) & mask
        return torch.where(eps_mask, rand, greedy)

    def learn_step(self):
        if len(self.replay) < self.batch_size:
            return None

        self.frame_idx += 1
        self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

        S, A, R, NS, D, idxs, w = self.replay.sample(self.batch_size, beta=self.beta)

        q_curr = self.online(S).gather(1, A).squeeze(1)
        with torch.no_grad():
            best = self.online(NS).argmax(1, keepdim=True)
            q_next = self.target(NS).gather(1, best).squeeze(1)
            tgt = R + self.gamma * q_next * (1.0 - D)

        td_errors = tgt - q_curr
        loss = (w * td_errors.pow(2)).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.replay.update_priorities(idxs, td_errors.detach().cpu().numpy())

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return loss.item()

    def after_episode(self, ep):
        if (ep + 1) % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

    def store_transition(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    def build_model(self):
        return self.online, self.target
