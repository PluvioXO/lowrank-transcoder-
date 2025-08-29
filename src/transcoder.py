\
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TranscoderConfig:
    d_in: int
    d_out: int
    rank: int = 64
    residual: bool = True
    l1: float = 0.0
    prox_every: int = 0  # steps between proximal soft-thresholding; 0 disables

class LinearMap(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        return self.W(x)

class LowRankMap(nn.Module):
    """
    LoRA-style factorization: W = U V^T, with optional residual skip if d_in==d_out.
    """
    def __init__(self, cfg: TranscoderConfig):
        super().__init__()
        self.cfg = cfg
        self.U = nn.Parameter(torch.empty(cfg.d_out, cfg.rank))
        self.V = nn.Parameter(torch.empty(cfg.d_in, cfg.rank))
        nn.init.kaiming_uniform_(self.U, a=5**0.5)
        nn.init.kaiming_uniform_(self.V, a=5**0.5)

    def forward(self, x):
        y = x @ self.V @ self.U.T  # [N, d_out]
        if self.cfg.residual and self.cfg.d_in == self.cfg.d_out:
            y = x + y
        return y

    def l1_penalty(self):
        return self.cfg.l1 * (self.U.abs().mean() + self.V.abs().mean())

    @torch.no_grad()
    def prox(self, t: float):
        if self.cfg.l1 <= 0: return
        lam = t * self.cfg.l1
        self.U.data = torch.sign(self.U.data) * torch.clamp(self.U.data.abs() - lam, min=0.0)
        self.V.data = torch.sign(self.V.data) * torch.clamp(self.V.data.abs() - lam, min=0.0)

class TinyMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
