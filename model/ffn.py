import torch
from torch import nn


class DenseFFN(nn.Module):
    """Two-layer SiLU FFN used for both dense FFNs and MoE experts."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class ExpertFFN(DenseFFN):
    pass
