# algorithms/dueling_ddqn.py
import torch, torch.nn as nn, torch.optim as optim
from algorithms.base_trainer import BaseTrainer
from core import ReplayBuffer, DIRS
from core import batched_update_fire_exit, batched_update_agents, \
                 batched_get_views, batched_step     # последнюю добавим чуть позже

# ===== модель =====
class DuelingCNN(nn.Module):
    def __init__(self,in_ch, actions, view):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.Flatten()
        )
        dim=128*view*view
        self.v=nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Linear(256,1))
        self.a=nn.Sequential(nn.Linear(dim,256), nn.ReLU(), nn.Linear(256,actions))
    def forward(self,x):
        f=self.conv(x)
        v=self.v(f); a=self.a(f)
        return v + a - a.mean(1,keepdim=True)

# ===== тренер =====
class DuelingDDQNTrainer(BaseTrainer):
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)
        self.online, self.target = self.build_model()
        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        self.buf = ReplayBuffer(self.memory_size)
        self.eps = self.eps_start
        self.epsilon = self.eps_start

    def build_model(self):
        m = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        t = DuelingCNN(self.in_ch, 4, self.view).to(self.device)
        t.load_state_dict(m.state_dict())
        t.eval()
        return m, t

    def select_actions(self, views, mask):
        B, N, *rest = views.shape
        flat = views.view(B * N, *rest)
        with torch.no_grad():
            q = self.online(flat)
        q = q.view(B, N, -1).argmax(dim=2)

        rand = torch.randint(0, 4, (B, N), device=self.device)
        choose = torch.where(torch.rand_like(q.float()) < self.eps, rand, q)

        acts = torch.zeros_like(q)
        acts[mask] = choose[mask]
        return acts

    def learn_step(self):
        if len(self.buf) < self.batch_size:
            return None

        S, A, R, NS, D = self.buf.sample(self.batch_size)
        qc = self.online(S).gather(1, A).squeeze()
        with torch.no_grad():
            best = self.online(NS).argmax(1, keepdim=True)
            qn = self.target(NS).gather(1, best).squeeze()
        tgt = R + self.gamma * qn * (1 - D)

        loss = nn.functional.mse_loss(qc, tgt)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if hasattr(self, "epsilon"):
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        return loss.item()

    def after_episode(self, ep):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        if (ep + 1) % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
