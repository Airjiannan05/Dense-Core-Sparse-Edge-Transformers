import torch

from model import DenseFFN, MoEFFN, upcycle_dense_ffn_to_moe


def test_dense_ffn_weights_copied_into_shared_and_routed_experts():
    dense = DenseFFN(hidden_size=8, intermediate_size=16)
    moe = MoEFFN(hidden_size=8, intermediate_size=16, num_experts=3, top_k=1)
    upcycle_dense_ffn_to_moe(dense, moe, expert_noise_std=0.0)

    for key, value in dense.state_dict().items():
        assert key in moe.shared_ffn.state_dict()
        assert torch.equal(value, moe.shared_ffn.state_dict()[key])
        for expert in moe.experts:
            assert torch.equal(value, expert.state_dict()[key])

    # Verify router was re-initialized (not copied from dense)
    assert any(not torch.equal(p, torch.zeros_like(p)) for p in moe.router.parameters())


def test_upcycling_with_noise_breaks_symmetry():
    dense = DenseFFN(hidden_size=8, intermediate_size=16)
    moe = MoEFFN(hidden_size=8, intermediate_size=16, num_experts=3, top_k=1)
    upcycle_dense_ffn_to_moe(dense, moe, expert_noise_std=1e-3)

    # After noise, experts should differ from each other
    state0 = moe.experts[0].state_dict()
    state1 = moe.experts[1].state_dict()
    any_diff = any(not torch.allclose(state0[k], state1[k]) for k in state0)
    assert any_diff, "experts should differ after adding noise"
