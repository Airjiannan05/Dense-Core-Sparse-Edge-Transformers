from __future__ import annotations

import torch
from torch import nn


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.router_jitter_noise > 0.0:
            noise = torch.empty_like(tokens).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )
            tokens = tokens * noise

        router_logits = self.router(tokens)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return router_probs, topk_probs, topk_indices
