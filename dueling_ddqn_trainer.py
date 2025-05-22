"""dueling_ddqn_trainer.py
Минимальная реализация тренера Dueling Double DQN, использующего core.py.
Позволяет проверить, что общий каркас работает с новым ядром.

Ограничения:
* Нет логирования метрик/видео.
* Обучается 2 эпизода × 50 шагов на одной среде (ENV=1) — быстрый smoke‑test.
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from core import (
    DIRS,
    batched_get_views,
    batched_step,
    batched_update_agents,
    batched_update_fire_exit,
    ReplayBuffer,
)

# ------------------- simplified Environment (wrapper around core.Environment) ---------
from typing import List

class SimpleEnv:
    """Одно‑средовое упрощение: берём Environment из 1.3, но только то, что нужно."""

    def __init__(self, env_cls, *args, **kwargs):
        self._env = env_cls(*args, **kwargs)
        self.N = self._env.N

    def reset(self):
        self._env.reset()

    # property proxies
    def __getattr__(self, item):
        return getattr(self._env, item)


# ------------------- Network ----------------------------------------------------------------

class DuelingCNN(nn.Module):
    def __init__(self, in_ch: int, n_actions: int, view: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        feat = 64 * view * view
        self.v = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        h = self.conv(x)
        v = self.v(h)
        a = self.a(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ------------------- Trainer -----------------------------------------------------------------

def smoke_test(env_cls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VIEW = 5
    AG_CH = 6
    ACTIONS = 4

    # env
    env = SimpleEnv(env_cls, 20, [(5, 1)], 1, [(19, 0)])
    env.reset()

    # net & target
    net = DuelingCNN(AG_CH, ACTIONS, VIEW).to(device)
    tgt = DuelingCNN(AG_CH, ACTIONS, VIEW).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=1e-3)
    buf = ReplayBuffer(10_000, device)

    EPS = 1.0
    GAMMA = 0.99
    print("▶ smoke‑train start …")

    for ep in range(2):
        positions, alive, knows, health, size, speed, fire, exit_m = (
            env.positions.unsqueeze(0),
            env.alive.unsqueeze(0),
            env.knows_exit.unsqueeze(0),
            env.health.unsqueeze(0),
            env.size.unsqueeze(0),
            env.speed.unsqueeze(0),
            env.fire_mask.unsqueeze(0),
            env.exit_mask.unsqueeze(0),
        )
        total = 0.0
        for step in range(50):
            fire, exit_m = batched_update_fire_exit(fire, exit_m, 0.1)
            ag, sz, sp, inf = batched_update_agents(positions, size, speed, knows, alive)
            views = batched_get_views(ag, fire, exit_m, sz, sp, inf, positions, view_size=VIEW)
            flat = views.view(-1, AG_CH, VIEW, VIEW)
            q = net(flat)
            greedy = q.argmax(dim=1).view(1, -1)
            rand = torch.randint(0, ACTIONS, greedy.shape, device=greedy.device)
            actions = torch.where(torch.rand_like(greedy, dtype=torch.float) < EPS, rand, greedy)

            positions, rew, done, alive, health, fire, exit_m, *_ = batched_step(
                positions, actions, size, speed, fire, exit_m, health
            )
            total += rew.sum().item()

            buf.push(views[0, 0].cpu(), actions[0, 0].item(), rew[0, 0].item(), views[0, 0].cpu(), done[0].item())
            if len(buf) >= 32:
                s, a, r, ns, d = buf.sample(32)
                qc = net(s).gather(1, a).squeeze()
                with torch.no_grad():
                    qn = tgt(ns).max(1)[0]
                tgt_q = r + GAMMA * qn * (1 - d)
                loss = nn.functional.mse_loss(qc, tgt_q)
                opt.zero_grad(); loss.backward(); opt.step()
            EPS *= 0.99
            if done.all():
                break
        print(f"Episode {ep+1}: total reward {total:.1f}")
    print("✔ smoke test finished")


if __name__ == "__main__":
    # Можем импортировать environment из оригинального кода 1.3 (здесь псевдо‑импорт)
    from environment import Environment  # TODO заменить на core.env.Environment после интеграции

    smoke_test(Environment)
