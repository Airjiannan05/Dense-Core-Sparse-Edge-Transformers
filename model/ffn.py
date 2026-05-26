import torch
from torch import nn


class DenseFFN(nn.Module):
    """SwiGLU FFN: W_down(SiLU(W_gate(x)) * W_up(x)).

    This matches modern LLM architectures (LLaMA, Mistral, etc.) and ensures
    a fair comparison between Dense FFN and MoE expert FFNs.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ExpertFFN(DenseFFN):
    """Expert FFN sharing the same SwiGLU structure as DenseFFN.

    Kept as a separate class for clarity and potential future specialization
    (e.g., expert-specific initialization or dropout).
    """
    pass
