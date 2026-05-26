import torch

from model import MoEFFN


def test_moe_shapes():
    moe = MoEFFN(hidden_size=16, intermediate_size=32, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 16)
    y, aux_loss, router_info = moe(x)
    assert y.shape == x.shape
    assert aux_loss.ndim == 0
    assert router_info["expert_counts"].shape == (4,)
    assert router_info["expert_probs"].shape == (4,)
