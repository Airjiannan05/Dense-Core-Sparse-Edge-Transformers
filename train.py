from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from infrastructure.metrics_logger import MetricsLogger
from model import DenseCoreSparseEdgeConfig, DenseCoreSparseEdgeTransformer
from model.metrics import estimate_active_flops_per_token, perplexity
from model.upcycling import upcycle_transformer_ffns
from infrastructure.run_manager import RunManager
from utils import build_token_stream, cosine_lr, load_yaml, parse_token_count, seed_everything, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="data/wikitext2_train.txt")
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

    # Initialize RunManager (creates run_dir under runs/<name>_<timestamp>)
    mgr = RunManager.init(
        run_name=Path(args.config).stem,
        args=args,
        config_dict=cfg_dict,
    )

    # Backward-compatible output_dir for checkpoint.pt and routing_trace.jsonl
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

    # Create MetricsLogger (JSONL writer under mgr.run_dir)
    mlog = MetricsLogger(mgr.run_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    stream = build_token_stream(args.dataset, config.vocab_size, seq_len, args.batch_size, args.seed)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Write dataset and model metadata
    mgr.write_dataset_info(args.dataset, config.vocab_size)
    mgr.write_model_info({
        "total_parameters": model.total_parameters(),
        "active_parameters": model.active_parameters(),
        "active_flops_per_token_estimate": estimate_active_flops_per_token(model),
        "moe_layer_indices": sorted(model.moe_layers),
        "dense_layer_indices": sorted(set(range(config.num_layers)) - model.moe_layers),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_experts": config.moe.get("num_experts"),
        "top_k": config.moe.get("top_k"),
        "shared_expert": config.moe.get("shared_expert"),
        "expert_intermediate_size": config.moe.get("expert_intermediate_size"),
    })

    print(f"dataset={args.dataset} device={device} steps={total_steps}")
    print(f"total_parameters={model.total_parameters()}")
    print(f"active_parameters={model.active_parameters()}")
    print(f"active_flops_per_token_estimate={estimate_active_flops_per_token(model)}")

    model.train()
    start = time.perf_counter()
    pbar = tqdm(range(total_steps), desc="training", unit="step", dynamic_ncols=True)

    # Tracking accumulators
    best_val_loss = float("inf")
    best_val_step = 0
    max_tok_s = 0.0
    total_tok_s_sum = 0.0
    peak_gpu_memory_mb = 0.0
    aux_loss_coef = config.moe.get("router_aux_loss_coef", 0.01)

    for step in pbar:
        step_start = time.perf_counter()

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

        # Compute grad_norm before optimizer step
        grad_norm = 0.0
        if hasattr(optimizer, "param_groups"):
            total_norm_sq = sum(
                p.grad.detach().norm().item() ** 2
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            grad_norm = math.sqrt(total_norm_sq)

        optimizer.step()

        step_time = time.perf_counter() - step_start

        if args.routing_trace_interval > 0 and step % args.routing_trace_interval == 0:
            rows = build_routing_trace(step, input_ids.detach().cpu(), out["router_info"])
            if rows:
                write_jsonl(output_dir / "routing_trace.jsonl", rows)

        elapsed = max(time.perf_counter() - start, 1e-6)
        tokens_seen = (step + 1) * args.batch_size * seq_len
        lm_loss_val = out["lm_loss"].item()
        aux_loss_val = out["aux_loss"].item()
        tok_sec = tokens_seen / elapsed

        # Update tracking accumulators
        max_tok_s = max(max_tok_s, tok_sec)
        total_tok_s_sum += tok_sec
        peak_gpu_memory_mb = max(peak_gpu_memory_mb, gpu_memory_mb())

        # Write training metrics every step (JSONL)
        raw_aux = (aux_loss_val / aux_loss_coef) if aux_loss_coef > 0 else 0.0
        mlog.write_metrics(
            step=step,
            tokens_seen=tokens_seen,
            loss=loss.item(),
            lm_loss=lm_loss_val,
            aux_loss=aux_loss_val,
            raw_aux_loss=raw_aux,
            ppl=perplexity(lm_loss_val),
            lr=lr,
            grad_norm=grad_norm,
            tok_s=tok_sec,
            step_time_sec=step_time,
            gpu_memory_mb=gpu_memory_mb(),
        )

        # Write router metrics every log_interval
        if step % args.log_interval == 0 and out["router_info"]:
            router_records = _build_router_records(step, tokens_seen, out["router_info"], config)
            mlog.write_router_metrics(router_records)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            lm=f"{lm_loss_val:.3f}",
            aux=f"{aux_loss_val:.3f}",
            ppl=f"{perplexity(lm_loss_val):.1f}",
            tok_s=f"{tok_sec:.0f}",
            mem=f"{gpu_memory_mb():.0f}M",
        )

        # Run evaluation on train-buffer subset every eval_interval
        if (step + 1) % args.eval_interval == 0:
            model.eval()
            eval_start = time.perf_counter()
            eval_tokens_total = 0
            eval_losses = []
            eval_lm_losses = []
            eval_steps = min(5, total_steps - step - 1)  # up to 5 mini-batches
            with torch.no_grad():
                for _ in range(eval_steps):
                    eval_ids = stream.next_batch(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        eval_out = model(eval_ids, labels=eval_ids)
                    eval_losses.append(eval_out["loss"].item())
                    eval_lm_losses.append(eval_out["lm_loss"].item())
                    eval_tokens_total += eval_ids.numel()
            eval_elapsed = max(time.perf_counter() - eval_start, 1e-6)
            val_loss = sum(eval_losses) / len(eval_losses)
            val_lm_loss = sum(eval_lm_losses) / len(eval_lm_losses)

            mlog.write_eval_metrics(
                step=step,
                tokens_seen=tokens_seen,
                val_loss=val_loss,
                val_lm_loss=val_lm_loss,
                val_ppl=perplexity(val_lm_loss),
                eval_tokens=eval_tokens_total,
                eval_tok_s=eval_tokens_total / eval_elapsed,
            )

            # Track best val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step
                # Ensure checkpoint dir exists
                best_dir = mgr.run_dir / "checkpoint"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "config": cfg_dict}, best_dir / "best.pt")

            model.train()

    # Save final checkpoint (backward compatible)
    torch.save({"model": model.state_dict(), "config": cfg_dict}, output_dir / "checkpoint.pt")

    # Compute summary stats
    router_final = {}
    for layer_name, info in out["router_info"].items():
        counts = info["expert_counts"].float()
        loads = counts / counts.sum().clamp_min(1.0)
        router_final[layer_name] = {
            "entropy": info["router_entropy"].item(),
            "gini": _gini(loads.tolist()),
        }

    summary = {
        "config_name": Path(args.config).stem,
        "dataset": args.dataset,
        "total_steps": total_steps,
        "tokens_trained": total_steps * args.batch_size * seq_len,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "final_train_lm_loss": lm_loss_val,
        "final_train_ppl": perplexity(lm_loss_val),
        "total_train_time_sec": time.perf_counter() - start,
        "peak_tok_s": max_tok_s,
        "avg_tok_s": total_tok_s_sum / max(total_steps, 1),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "total_parameters": model.total_parameters(),
        "active_parameters": model.active_parameters(),
        "active_flops_per_token": estimate_active_flops_per_token(model),
        "router_final": router_final,
    }
    mgr.finalize(summary)


def _build_router_records(step: int, tokens_seen: int, router_info: dict, config) -> list[dict]:
    """Build per-layer router records matching schema §3.3."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = []
    aux_coef = config.moe.get("router_aux_loss_coef", 0.01)
    for layer_name, info in router_info.items():
        layer_idx = int(layer_name.split("_")[-1])
        counts = info["expert_counts"].float()
        total = counts.sum().clamp_min(1.0)
        fractions = counts / total
        probs = info["expert_probs"].float()
        prob_mean = probs.mean().item()
        entropy = info["router_entropy"].item()
        num_experts = counts.numel()
        # Normalized entropy: entropy / ln(num_experts)
        normalized_entropy = entropy / max(1e-9, math.log(num_experts))
        # Gini coefficient
        sorted_fractions, _ = fractions.sort()
        gini = _gini(sorted_fractions.tolist())
        dropped = info.get("dropped_tokens", torch.tensor(0)).item()
        raw_aux = info["aux_loss"].item()
        scaled_aux = raw_aux * aux_coef

        records.append({
            "step": step,
            "tokens_seen": tokens_seen,
            "timestamp_utc": timestamp,
            "layer_idx": layer_idx,
            "expert_count": counts.long().tolist(),
            "expert_fraction": fractions.tolist(),
            "router_prob_mean": prob_mean,
            "router_entropy": entropy,
            "normalized_router_entropy": normalized_entropy,
            "expert_load_gini": gini,
            "max_expert_fraction": fractions.max().item(),
            "min_expert_fraction": fractions.min().item(),
            "dropped_tokens": dropped,
            "raw_aux_loss": raw_aux,
            "scaled_aux_loss": scaled_aux,
        })
    return records


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient of a sorted list of non-negative values."""
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    index = list(range(1, n + 1))
    total = sum(values)
    return (2 * sum(i * v for i, v in zip(index, values)) - (n + 1) * total) / (n * total)


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
