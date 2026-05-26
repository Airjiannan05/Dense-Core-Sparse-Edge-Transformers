from __future__ import annotations

import math

from .moe import MoEFFN


def estimate_active_flops_per_token(model) -> int:
    cfg = model.config
    h = cfg.hidden_size
    total = 0
    for layer in model.layers:
        # Dense self-attention: QKV projection (3*h*h) + output (h*h)
        total += 4 * h * h
        if isinstance(layer.ffn, MoEFFN):
            expert_h = layer.ffn.intermediate_size
            # SwiGLU: gate(h->e) + up(h->e) + down(e->h) = 3 * h * e
            active_experts = layer.ffn.top_k + (1 if layer.ffn.shared_ffn is not None else 0)
            total += active_experts * 3 * h * expert_h
            # Router: h * num_experts
            total += h * layer.ffn.num_experts
        else:
            # Dense SwiGLU FFN: gate + up + down = 3 * h * intermediate
            total += 3 * h * cfg.intermediate_size
    # LM head: h * vocab_size
    total += h * cfg.vocab_size
    return total


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
