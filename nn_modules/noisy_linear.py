# nn_modules/noisy_linear.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Factorised Gaussian Noisy layer (Fortunato et al., 2018).
    На inference ведёт себя как обычный Linear, если не звать reset_noise().
    """
    def __init__(self, in_f: int, out_f: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f

        # μ-параметры
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.bias_mu   = nn.Parameter(torch.empty(out_f))

        # σ-параметры
        self.weight_sigma = nn.Parameter(
            torch.full((out_f, in_f), sigma_init / math.sqrt(in_f))
        )
        self.bias_sigma = nn.Parameter(
            torch.full((out_f,),     sigma_init / math.sqrt(in_f))
        )

        # буферы ε
        self.register_buffer("eps_in",  torch.zeros(1,  in_f))
        self.register_buffer("eps_out", torch.zeros(out_f, 1))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_f)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu,   -bound, bound)

    @staticmethod
    def _f(x):         # f(u) из статьи
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        self.eps_in .normal_(); self.eps_in .copy_(self._f(self.eps_in))
        self.eps_out.normal_(); self.eps_out.copy_(self._f(self.eps_out))

    def forward(self, x):
        # W = μ_W + σ_W ⊙ ε_out ε_inᵀ
        w = self.weight_mu + self.weight_sigma * (self.eps_out @ self.eps_in)
        b = self.bias_mu   + self.bias_sigma   * self.eps_out.squeeze()
        return F.linear(x, w, b)
