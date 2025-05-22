# models/noisy_dueling_cnn.py
import torch.nn as nn
from nn_modules.noisy_linear import NoisyLinear

class NoisyDuelingCNN(nn.Module):
    """
    Dueling CNN-бэкбон с NoisyLinear головами.
    """
    def __init__(self, in_ch: int, n_actions: int, view: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten()
        )
        feat = 128 * view * view
        # value & advantage потоки
        self.value = nn.Sequential(
            NoisyLinear(feat, 256), nn.ReLU(),
            NoisyLinear(256, 1)
        )
        self.adv   = nn.Sequential(
            NoisyLinear(feat, 256), nn.ReLU(),
            NoisyLinear(256, n_actions)
        )

    def forward(self, x):
        z = self.conv(x)
        v = self.value(z)
        a = self.adv(z)
        return v + (a - a.mean(dim=1, keepdim=True))

    # вызовем из трейнера
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
