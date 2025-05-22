# algorithms/noisy_dueling_ddqn.py
import torch
import torch.optim as optim
from algorithms.base_trainer import BaseTrainer
from models.noisy_dueling_cnn import NoisyDuelingCNN
from core import device   # loss уже есть, можно MSE
from core import ReplayBuffer                   # обычный буфер

class NoisyDuelingDDQNTrainer(BaseTrainer):
    """Dueling Double DQN с factorised Noisy-Net (без ε)."""
    def __init__(self, envs, cfg):
        super().__init__(envs, cfg)

        self.online = NoisyDuelingCNN(self.in_ch, 4, self.view).to(device)
        self.target = NoisyDuelingCNN(self.in_ch, 4, self.view).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.replay = ReplayBuffer(self.memory_size)
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)
        self.loss_fn   = torch.nn.MSELoss()

        self.epsilon = float("nan") 

    # ---------- действий ----------
    def select_actions(self, views, mask):
        B, N, C, V, _ = views.shape
        flat = views.view(B * N, C, V, V)

        self.online.reset_noise()          # ключевая строка
        q = self.online(flat)
        acts = q.argmax(dim=1)
        return acts.view(B, N)

    # ---------- обучение ----------
    def learn_step(self):
        if len(self.replay) < self.batch_size:
            return None

        S, A, R, NS, D = self.replay.sample(self.batch_size)

        # 🎲 Обновляем шум (для обоих сетей)
        self.online.reset_noise()
        self.target.reset_noise()

        # Q(s, a)
        q_curr = self.online(S).gather(1, A).squeeze()

        with torch.no_grad():
            best = self.online(NS).argmax(dim=1, keepdim=True)
            q_next = self.target(NS).gather(1, best).squeeze()
            tgt = R + self.gamma * q_next * (1 - D)

        loss = self.loss_fn(q_curr, tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    # ---------- target sync ----------
    def after_episode(self, ep):
        if (ep + 1) % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    def build_model(self):
        return self.online, self.target
