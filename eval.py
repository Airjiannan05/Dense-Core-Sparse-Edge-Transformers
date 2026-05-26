from __future__ import annotations

import argparse
import time

import torch

from model import DenseCoreSparseEdgeConfig, DenseCoreSparseEdgeTransformer
from model.metrics import estimate_active_flops_per_token, perplexity
from utils import build_token_stream, load_yaml, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dataset", default="random")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg_dict = load_yaml(args.config)["model"]
    config = DenseCoreSparseEdgeConfig.from_dict(cfg_dict)
    seq_len = args.seq_len or min(1024, config.max_position_embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseCoreSparseEdgeTransformer(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    stream = build_token_stream(args.dataset, config.vocab_size, seq_len, args.batch_size, args.seed)
    losses = []
    start = time.perf_counter()
    for _ in range(args.steps):
        input_ids = stream.next_batch(device)
        out = model(input_ids, labels=input_ids)
        losses.append(out["lm_loss"].item())
    elapsed = max(time.perf_counter() - start, 1e-6)
    mean_loss = sum(losses) / len(losses)
    tokens = args.steps * args.batch_size * seq_len

    print(f"validation_loss={mean_loss:.4f}")
    print(f"validation_perplexity={perplexity(mean_loss):.2f}")
    print(f"tokens_sec={tokens / elapsed:.1f}")
    print(f"total_parameters={model.total_parameters()}")
    print(f"active_parameters={model.active_parameters()}")
    print(f"active_flops_per_token_estimate={estimate_active_flops_per_token(model)}")
    if torch.cuda.is_available():
        print(f"gpu_memory_mb={torch.cuda.max_memory_allocated() / (1024 ** 2):.1f}")
    else:
        print("gpu_memory_mb=0.0")


if __name__ == "__main__":
    main()
