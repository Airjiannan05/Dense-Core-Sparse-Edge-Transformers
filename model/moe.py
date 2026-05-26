from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .ffn import ExpertFFN
from .router import TopKRouter


def load_balancing_loss(
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    expert_mask = F.one_hot(topk_indices, num_classes=num_experts).sum(dim=1)
    expert_mask = expert_mask.clamp(max=1).to(router_probs.dtype)
    f = expert_mask.mean(dim=0)
    p = router_probs.mean(dim=0)
    return num_experts * torch.sum(f * p)


class MoEFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        shared_expert: bool = True,
        router_jitter_noise: float = 0.0,
        router_aux_loss_coef: float = 0.01,
        capacity_factor: float = 1.25,
        drop_tokens: bool = False,
    ):
        super().__init__()
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert_enabled = shared_expert
        self.router_aux_loss_coef = router_aux_loss_coef
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.record_routing_trace = False

        self.shared_ffn = ExpertFFN(hidden_size, intermediate_size) if shared_expert else None
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size) for _ in range(num_experts)
        )
        self.router = TopKRouter(hidden_size, num_experts, top_k, router_jitter_noise)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, hidden_size = x.shape
        tokens = x.reshape(batch_size * seq_len, hidden_size)

        router_probs, topk_probs, topk_indices = self.router(tokens)
        routed = torch.zeros_like(tokens)
        dropped = torch.zeros((), device=x.device, dtype=torch.long)
        capacity = math.ceil(self.capacity_factor * tokens.shape[0] * self.top_k / self.num_experts)

        for expert_idx, expert in enumerate(self.experts):
            token_pos, choice_pos = torch.where(topk_indices == expert_idx)
            if token_pos.numel() == 0:
                continue

            if self.drop_tokens and token_pos.numel() > capacity:
                keep = torch.arange(capacity, device=x.device)
                dropped = dropped + (token_pos.numel() - capacity)
                token_pos = token_pos[keep]
                choice_pos = choice_pos[keep]

            expert_out = expert(tokens.index_select(0, token_pos))
            weights = topk_probs[token_pos, choice_pos].unsqueeze(-1).to(expert_out.dtype)
            routed.index_add_(0, token_pos, expert_out * weights)

        if self.shared_ffn is not None:
            y = self.shared_ffn(tokens) + routed
        else:
            y = routed

        aux_loss = load_balancing_loss(router_probs, topk_indices, self.num_experts)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        expert_mask = expert_mask.clamp(max=1).to(router_probs.dtype)
        expert_counts = expert_mask.sum(dim=0)
        expert_probs = router_probs.mean(dim=0)
        entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()

        router_info: dict[str, torch.Tensor] = {
            "expert_counts": expert_counts.detach(),
            "expert_probs": expert_probs.detach(),
            "router_entropy": entropy.detach(),
            "aux_loss": aux_loss.detach(),
            "dropped_tokens": dropped.detach(),
        }
        if self.record_routing_trace:
            router_info["topk_indices"] = topk_indices.detach().cpu()
            router_info["topk_probs"] = topk_probs.detach().cpu()

        return y.reshape(batch_size, seq_len, hidden_size), aux_loss, router_info
