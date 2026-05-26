import torch

from model import load_balancing_loss


def test_load_balance_loss_is_finite_and_differentiable():
    logits = torch.randn(13, 4, requires_grad=True)
    router_probs = torch.softmax(logits, dim=-1)
    topk_indices = torch.topk(router_probs, k=2, dim=-1).indices
    loss = load_balancing_loss(router_probs, topk_indices, num_experts=4)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
