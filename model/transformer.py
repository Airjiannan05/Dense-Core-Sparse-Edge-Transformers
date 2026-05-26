from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import DenseSelfAttention
from .ffn import DenseFFN
from .moe import MoEFFN
from .placement import get_moe_layers


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


@dataclass
class DenseCoreSparseEdgeConfig:
    architecture: str = "dense_core_sparse_edge"
    hidden_size: int = 768
    num_layers: int = 24
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    attention: dict[str, Any] = field(default_factory=lambda: {"type": "dense"})
    ffn: dict[str, Any] = field(
        default_factory=lambda: {
            "placement": "edge_moe",
            "front_moe_layers": 4,
            "back_moe_layers": 6,
        }
    )
    moe: dict[str, Any] = field(
        default_factory=lambda: {
            "num_experts": 8,
            "top_k": 2,
            "shared_expert": True,
            "expert_intermediate_size": 1536,
            "router_aux_loss_coef": 0.01,
            "capacity_factor": 1.25,
            "router_jitter_noise": 0.0,
            "drop_tokens": False,
        }
    )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "DenseCoreSparseEdgeConfig":
        cfg = dict(cfg)
        return cls(**cfg)


class TransformerBlock(nn.Module):
    def __init__(self, config: DenseCoreSparseEdgeConfig, layer_idx: int, moe_layers: set[int]):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = DenseSelfAttention(config.hidden_size, config.num_attention_heads)
        self.norm2 = RMSNorm(config.hidden_size)

        if layer_idx in moe_layers:
            moe_cfg = config.moe
            self.ffn = MoEFFN(
                hidden_size=config.hidden_size,
                intermediate_size=moe_cfg.get("expert_intermediate_size", config.intermediate_size),
                num_experts=moe_cfg.get("num_experts", 8),
                top_k=moe_cfg.get("top_k", 2),
                shared_expert=moe_cfg.get("shared_expert", True),
                router_jitter_noise=moe_cfg.get("router_jitter_noise", 0.0),
                router_aux_loss_coef=moe_cfg.get("router_aux_loss_coef", 0.01),
                capacity_factor=moe_cfg.get("capacity_factor", 1.25),
                drop_tokens=moe_cfg.get("drop_tokens", False),
            )
        else:
            self.ffn = DenseFFN(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.attn(self.norm1(x))
        ffn_input = self.norm2(x)
        if isinstance(self.ffn, MoEFFN):
            ffn_out, aux_loss, router_info = self.ffn(ffn_input)
        else:
            ffn_out = self.ffn(ffn_input)
            aux_loss = x.new_zeros(())
            router_info = {}
        x = x + ffn_out
        return x, aux_loss, router_info


class DenseCoreSparseEdgeTransformer(nn.Module):
    def __init__(self, config: DenseCoreSparseEdgeConfig | dict[str, Any]):
        super().__init__()
        if isinstance(config, dict):
            config = DenseCoreSparseEdgeConfig.from_dict(config)
        self.config = config

        attn_type = config.attention.get("type", "dense")
        if attn_type != "dense":
            raise ValueError("This prototype intentionally supports dense self-attention only")

        ffn_cfg = config.ffn
        placement = ffn_cfg.get("placement")
        if placement is None:
            placement = "edge_moe" if ffn_cfg.get("type") == "dense_core_sparse_edge" else ffn_cfg.get("type", "edge_moe")
        n_moe_layers = ffn_cfg.get("n_moe_layers")
        if n_moe_layers is None and placement != "full_moe":
            n_moe_layers = ffn_cfg.get("front_moe_layers", 0) + ffn_cfg.get("back_moe_layers", 0)

        self.moe_layers = set(
            get_moe_layers(
                placement,
                config.num_layers,
                n_moe_layers=n_moe_layers,
                seed=ffn_cfg.get("random_seed", 0),
                front_moe_layers=ffn_cfg.get("front_moe_layers") if placement == "edge_moe" else None,
                back_moe_layers=ffn_cfg.get("back_moe_layers") if placement == "edge_moe" else None,
            )
        )

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx, self.moe_layers)
            for layer_idx in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)

        router_info: dict[str, dict[str, torch.Tensor]] = {}
        aux_losses = []
        for idx, layer in enumerate(self.layers):
            x, aux_loss, info = layer(x)
            if info:
                router_info[f"layer_{idx}"] = info
                coef = getattr(layer.ffn, "router_aux_loss_coef", 1.0)
                aux_losses.append(aux_loss * coef)

        x = self.norm(x)
        logits = self.lm_head(x)
        aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
        lm_loss = None
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + aux_loss

        return {
            "logits": logits,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "loss": loss,
            "router_info": router_info,
        }

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameters(self) -> int:
        total = self.tok_embeddings.weight.numel() + self.pos_embeddings.weight.numel()
        total += sum(p.numel() for p in self.norm.parameters())
        if not self.config.tie_word_embeddings:
            total += sum(p.numel() for p in self.lm_head.parameters())

        for layer in self.layers:
            total += sum(p.numel() for p in layer.norm1.parameters())
            total += sum(p.numel() for p in layer.norm2.parameters())
            total += sum(p.numel() for p in layer.attn.parameters())
            if isinstance(layer.ffn, MoEFFN):
                total += sum(p.numel() for p in layer.ffn.router.parameters())
                if layer.ffn.shared_ffn is not None:
                    total += sum(p.numel() for p in layer.ffn.shared_ffn.parameters())
                routed_one = sum(p.numel() for p in layer.ffn.experts[0].parameters())
                total += layer.ffn.top_k * routed_one
            else:
                total += sum(p.numel() for p in layer.ffn.parameters())
        return total

    def set_record_routing_trace(self, enabled: bool) -> None:
        for layer in self.layers:
            if isinstance(layer.ffn, MoEFFN):
                layer.ffn.record_routing_trace = enabled
