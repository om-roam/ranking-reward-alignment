import torch.nn as nn

from coral.models.reward.norms import RMSNorm

class QwenStyleCORALHead(nn.Module):
    def __init__(self, hidden_dim: int, k: int):
        super().__init__()

        self.norm = RMSNorm(hidden_dim, eps=1e-6)

        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)

        self.act = nn.SiLU()

        self.out_proj = nn.Linear(hidden_dim, k, bias=True)

    def forward(self, x):
        x = self.norm(x)

        gated = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(gated)

        return self.out_proj(x)