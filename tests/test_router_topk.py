import torch

from model.router import TopKRouter


def test_router_topk_each_token_routes_to_exactly_top_k_experts():
    router = TopKRouter(hidden_size=8, num_experts=5, top_k=2)
    tokens = torch.randn(11, 8)
    router_probs, topk_probs, topk_indices = router(tokens)
    assert router_probs.shape == (11, 5)
    assert topk_probs.shape == (11, 2)
    assert topk_indices.shape == (11, 2)
    assert torch.allclose(topk_probs.sum(dim=-1), torch.ones(11), atol=1e-6)
    assert all(len(set(row.tolist())) == 2 for row in topk_indices)
