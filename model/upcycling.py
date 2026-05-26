from __future__ import annotations

import torch

from .ffn import DenseFFN
from .moe import MoEFFN


@torch.no_grad()
def upcycle_dense_ffn_to_moe(
    dense_ffn: DenseFFN,
    moe_ffn: MoEFFN,
    expert_noise_std: float = 0.0,
) -> None:
    if moe_ffn.shared_ffn is not None:
        moe_ffn.shared_ffn.load_state_dict(dense_ffn.state_dict())

    for expert in moe_ffn.experts:
        expert.load_state_dict(dense_ffn.state_dict())
        if expert_noise_std > 0.0:
            for param in expert.parameters():
                param.add_(torch.randn_like(param) * expert_noise_std)

    for param in moe_ffn.router.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)


@torch.no_grad()
def upcycle_transformer_ffns(
    dense_model,
    moe_model,
    expert_noise_std: float = 0.0,
) -> None:
    dense_layers = getattr(dense_model, "layers")
    moe_layers = getattr(moe_model, "layers")
    if len(dense_layers) != len(moe_layers):
        raise ValueError("dense and MoE models must have the same number of layers")

    dense_state = dense_model.state_dict()
    compatible = {
        k: v for k, v in dense_state.items()
        if k in moe_model.state_dict() and moe_model.state_dict()[k].shape == v.shape
    }
    moe_model.load_state_dict(compatible, strict=False)

    for dense_layer, moe_layer in zip(dense_layers, moe_layers):
        if isinstance(getattr(moe_layer, "ffn"), MoEFFN):
            if not isinstance(getattr(dense_layer, "ffn"), DenseFFN):
                raise TypeError("source layer FFN must be DenseFFN")
            upcycle_dense_ffn_to_moe(dense_layer.ffn, moe_layer.ffn, expert_noise_std)
