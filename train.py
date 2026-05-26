from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from model import DenseCoreSparseEdgeConfig, DenseCoreSparseEdgeTransformer
from model.metrics import estimate_active_flops_per_token, perplexity
from model.upcycling import upcycle_transformer_ffns
from utils import build_token_stream, cosine_lr, load_yaml, parse_token_count, seed_everything, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="random")
    parser.add_argument("--tokens", default="10M")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--routing_trace_interval", type=int, default=0)
    parser.add_argument("--upcycle_from_dense_checkpoint", default=None)
    parser.add_argument("--upcycling_noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg_dict = load_yaml(args.config)["model"]
    config = DenseCoreSparseEdgeConfig.from_dict(cfg_dict)
    seq_len = args.seq_len or min(1024, config.max_position_embeddings)
    target_tokens = parse_token_count(args.tokens)
    total_steps = max(1, target_tokens // (args.batch_size * seq_len))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseCoreSparseEdgeTransformer(config).to(device)
    if args.upcycle_from_dense_checkpoint:
        checkpoint = torch.load(args.upcycle_from_dense_checkpoint, map_location=device)
        dense_cfg = DenseCoreSparseEdgeConfig.from_dict(checkpoint.get("config", cfg_dict))
        dense_model = DenseCoreSparseEdgeTransformer(dense_cfg).to(device)
        dense_model.load_state_dict(checkpoint["model"])
        upcycle_transformer_ffns(dense_model, model, expert_noise_std=args.upcycling_noise_std)
        del dense_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    stream = build_token_stream(args.dataset, config.vocab_size, seq_len, args.batch_size, args.seed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    print(f"dataset={args.dataset} device={device} steps={total_steps}")
    print(f"total_parameters={model.total_parameters()}")
    print(f"active_parameters={model.active_parameters()}")
    print(f"active_flops_per_token_estimate={estimate_active_flops_per_token(model)}")

    model.train()
    start = time.perf_counter()
    pbar = tqdm(range(total_steps), desc="training", unit="step", dynamic_ncols=True)

    for step in pbar:
        lr = cosine_lr(args.learning_rate, step, total_steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids = stream.next_batch(device)
        model.set_record_routing_trace(
            args.routing_trace_interval > 0 and step % args.routing_trace_interval == 0
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(input_ids, labels=input_ids)
            loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.routing_trace_interval > 0 and step % args.routing_trace_interval == 0:
            rows = build_routing_trace(step, input_ids.detach().cpu(), out["router_info"])
            if rows:
                write_jsonl(output_dir / "routing_trace.jsonl", rows)

        elapsed = max(time.perf_counter() - start, 1e-6)
        tokens_seen = (step + 1) * args.batch_size * seq_len
        lm_loss_val = out["lm_loss"].item()
        aux_loss_val = out["aux_loss"].item()
        tok_sec = tokens_seen / elapsed

        if step % args.log_interval == 0:
            log_router_metrics(out["router_info"], prefix=f"step={step}")

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            lm=f"{lm_loss_val:.3f}",
            aux=f"{aux_loss_val:.3f}",
            ppl=f"{perplexity(lm_loss_val):.1f}",
            tok_s=f"{tok_sec:.0f}",
            mem=f"{gpu_memory_mb():.0f}M",
        )

    torch.save({"model": model.state_dict(), "config": cfg_dict}, output_dir / "checkpoint.pt")


def log_router_metrics(router_info: dict, prefix: str) -> None:
    for layer_name, info in router_info.items():
        counts = info["expert_counts"].float()
        loads = counts / counts.sum().clamp_min(1.0)
        entropy = info["router_entropy"].item()
        aux = info["aux_loss"].item()
        load_text = " ".join(f"{layer_name}/expert_{i}_load={v:.4f}" for i, v in enumerate(loads.tolist()))
        print(f"{prefix} {load_text} {layer_name}/router_entropy={entropy:.4f} {layer_name}/aux_loss={aux:.4f}")


def build_routing_trace(step: int, input_ids: torch.Tensor, router_info: dict) -> list[dict]:
    rows = []
    tokens = input_ids.reshape(-1).tolist()
    for layer_name, info in router_info.items():
        if "topk_indices" not in info:
            continue
        layer_idx = int(layer_name.split("_")[-1])
        topk_indices = info["topk_indices"].tolist()
        topk_probs = info["topk_probs"].tolist()
        rows.append(
            {
                "step": step,
                "layer": layer_idx,
                "tokens": tokens,
                "topk_experts": topk_indices,
                "topk_probs": topk_probs,
                "task_label": "unknown",
            }
        )
    return rows


def gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


if __name__ == "__main__":
    main()
