import torch

from model import MoEFFN


def test_moe_shapes_with_shared():
    moe = MoEFFN(hidden_size=16, intermediate_size=32, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 16)
    y, aux_loss, router_info = moe(x)
    assert y.shape == x.shape
    assert aux_loss.ndim == 0
    assert router_info["expert_counts"].shape == (4,)
    assert router_info["expert_probs"].shape == (4,)


def test_moe_shapes_without_shared():
    moe = MoEFFN(hidden_size=16, intermediate_size=32, num_experts=4, top_k=2, shared_expert=False)
    x = torch.randn(2, 5, 16)
    y, aux_loss, router_info = moe(x)
    assert y.shape == x.shape
    assert aux_loss.ndim == 0


def test_moe_gradient_flow():
    moe = MoEFFN(hidden_size=16, intermediate_size=32, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y, aux_loss, _ = moe(x)
    loss = y.sum() + aux_loss
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
