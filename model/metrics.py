from __future__ import annotations

import math

from .moe import MoEFFN


def estimate_active_flops_per_token(model) -> int:
    cfg = model.config
    h = cfg.hidden_size
    total = 0
    for layer in model.layers:
        total += 4 * h * h
        if isinstance(layer.ffn, MoEFFN):
            expert_h = layer.ffn.intermediate_size
            active_experts = layer.ffn.top_k + (1 if layer.ffn.shared_ffn is not None else 0)
            total += active_experts * 2 * h * expert_h
            total += h * layer.ffn.num_experts
        else:
            total += 2 * h * cfg.intermediate_size
    total += h * cfg.vocab_size
    return total


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
