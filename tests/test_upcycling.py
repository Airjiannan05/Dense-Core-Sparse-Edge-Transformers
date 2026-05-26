import torch

from model import DenseFFN, MoEFFN, upcycle_dense_ffn_to_moe


def test_dense_ffn_weights_copied_into_shared_and_routed_experts():
    dense = DenseFFN(hidden_size=8, intermediate_size=16)
    moe = MoEFFN(hidden_size=8, intermediate_size=16, num_experts=3, top_k=1)
    upcycle_dense_ffn_to_moe(dense, moe, expert_noise_std=0.0)

    for key, value in dense.state_dict().items():
        assert torch.equal(value, moe.shared_ffn.state_dict()[key])
        for expert in moe.experts:
            assert torch.equal(value, expert.state_dict()[key])
