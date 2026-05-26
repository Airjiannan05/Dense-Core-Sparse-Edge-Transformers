from model import DenseCoreSparseEdgeConfig, DenseCoreSparseEdgeTransformer
from model.attention import DenseSelfAttention


def test_all_layers_use_dense_self_attention():
    cfg = DenseCoreSparseEdgeConfig(
        hidden_size=16,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=64,
        max_position_embeddings=16,
        ffn={"placement": "full_moe"},
        moe={"num_experts": 4, "top_k": 2, "expert_intermediate_size": 16},
    )
    model = DenseCoreSparseEdgeTransformer(cfg)
    assert all(isinstance(layer.attn, DenseSelfAttention) for layer in model.layers)
