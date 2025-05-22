
import torch, random, time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from algorithms.base_trainer import BaseTrainer
from core import (device, batched_update_fire_exit, batched_update_agents,
                  batched_get_views, batched_step)
from core import ReplayBuffer


# ---------- модель QR‑DQN ---------------------------------
class QuantileCNN(nn.Module):
    def __init__(self, in_ch:int, n_actions:int, view_size:int,
                 n_atoms:int, tau_embed_dim:int):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,128, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten())
        self.feat_dim = 128*view_size*view_size
        self.tau_fc = nn.Sequential(
            nn.Linear(1, tau_embed_dim), nn.ReLU(),
            nn.Linear(tau_embed_dim, self.feat_dim), nn.ReLU())
        self.head = nn.Linear(self.feat_dim, n_actions)

    def forward(self, x:torch.Tensor, taus:torch.Tensor):
        B = x.size(0)
        h = self.conv(x)                                 # [B, feat]
        h = h.unsqueeze(1).expand(-1, self.n_atoms, -1)  # [B,N,feat]
        tau_emb = self.tau_fc(taus.unsqueeze(-1))        # [B,N,feat]
        z = h * tau_emb                                  # Hadamard
        q = self.head(z)                                 # [B,N,act]
        return q


# -------- вспомогательные функции -----------------
def sample_taus(batch:int, n_atoms:int, device):
    taus = torch.arange(n_atoms, device=device, dtype=torch.float32)
    taus = (taus + 0.5) / n_atoms
    return taus.unsqueeze(0).expand(batch, -1)

def quantile_huber_loss(pred, target, taus, k=1.0):
    d = target - pred
    huber = F.smooth_l1_loss(pred, target, reduction='none')
    weight = torch.abs(taus - (d < 0).float())
    return (weight * huber).mean()

# ------------- тренер QR‑DQN ------------------------
class QRDQNTrainer(BaseTrainer):
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)  # <-- первым!

        self.replay = ReplayBuffer(self.memory_size)

        self.n_atoms = cfg.get("N_ATOMS", 16)
        self.tau_embed_dim = cfg.get("TAU_EMBED_DIM", 128)

        self.online_net = QuantileCNN(
            self.in_ch, 4, self.view, self.n_atoms, self.tau_embed_dim
        ).to(device)

        self.target_net = QuantileCNN(
            self.in_ch, 4, self.view, self.n_atoms, self.tau_embed_dim
        ).to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = quantile_huber_loss

        self.online, self.target = self.build_model()

    def store_transition(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    # ------ выбор действия ε‑жадно -------------
    def act(self, active_views, eps):
        B = active_views.size(0)
        taus = sample_taus(B, self.n_atoms, device)
        with torch.no_grad():
            q_atoms = self.online_net(active_views, taus)   # [B,N,act]
            q_mean  = q_atoms.mean(dim=1)                   # [B,act]
        return q_mean.argmax(dim=1)

    def select_actions(self, views, mask):
        eps = self.cfg.get("EPSILON", 0.0)
        B, N, C, V, _ = views.shape
        flat = views.view(B * N, C, V, V)
        actions = self.act(flat, eps)
        return actions.view(B, N)
    
    
    def learn_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = self.replay.sample(self.batch_size)
        return self.train_step(batch)

    # --------- шаг обучения -------------
    def train_step(self, batch):
        S, A, R, NS, D = batch
        B = S.size(0)
        taus_s  = sample_taus(B, self.n_atoms, device)
        taus_ns = sample_taus(B, self.n_atoms, device)

        # онлайн
        q_atoms_s = self.online_net(S, taus_s)                    # [B,N,act]
        q_atoms_sa= q_atoms_s[torch.arange(B), :, A.squeeze()]    # [B,N]

        # Double: argmax через онлайн, оценка через target
        with torch.no_grad():
            q_atoms_ns = self.online_net(NS, taus_ns)             # [B,N,act]
            q_mean_ns  = q_atoms_ns.mean(dim=1)                   # [B,act]
            best = q_mean_ns.argmax(dim=1)
            q_atoms_target = self.target_net(NS, taus_ns)
            q_atoms_next = q_atoms_target[torch.arange(B), :, best]  # [B,N]
            Tz = R.unsqueeze(1) + self.gamma * q_atoms_next * (1 - D.unsqueeze(1))

        loss = self.loss_fn(q_atoms_sa, Tz, taus_s)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def build_model(self):
        return self.online_net, self.target_net

    def after_episode(self, ep_idx):
        if (ep_idx + 1) % self.cfg["TARGET_UPDATE_FREQ"] == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()