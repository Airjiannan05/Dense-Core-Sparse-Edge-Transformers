from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetricsLogger:
    """JSONL writer for training metrics, eval metrics, and router metrics.

    Three output files (created on first write):
      - metrics.jsonl      : one line per training step
      - eval_metrics.jsonl : one line per eval_interval
      - router_metrics.jsonl : one line per MoE layer per log_interval
    """

    def __init__(self, run_dir: Path) -> None:
        self._metrics_path = run_dir / "metrics.jsonl"
        self._eval_path = run_dir / "eval_metrics.jsonl"
        self._router_path = run_dir / "router_metrics.jsonl"

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def write_metrics(self, *, step: int, tokens_seen: int, loss: float,
                      lm_loss: float, aux_loss: float, raw_aux_loss: float,
                      ppl: float, lr: float, grad_norm: float,
                      tok_s: float, step_time_sec: float,
                      gpu_memory_mb: float) -> None:
        """Write one line to metrics.jsonl. All fields required per schema §3.1."""
        self._append(self._metrics_path, {
            "step": step,
            "tokens_seen": tokens_seen,
            "timestamp_utc": self._now_utc(),
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "raw_aux_loss": raw_aux_loss,
            "ppl": ppl,
            "lr": lr,
            "grad_norm": grad_norm,
            "tok_s": tok_s,
            "step_time_sec": step_time_sec,
            "gpu_memory_mb": gpu_memory_mb,
        })

    def write_eval_metrics(self, *, step: int, tokens_seen: int,
                           val_loss: float, val_lm_loss: float,
                           val_ppl: float, eval_tokens: int,
                           eval_tok_s: float) -> None:
        """Write one line to eval_metrics.jsonl. All fields required per schema §3.2."""
        self._append(self._eval_path, {
            "step": step,
            "tokens_seen": tokens_seen,
            "timestamp_utc": self._now_utc(),
            "val_loss": val_loss,
            "val_lm_loss": val_lm_loss,
            "val_ppl": val_ppl,
            "eval_tokens": eval_tokens,
            "eval_tok_s": eval_tok_s,
        })

    def write_router_metrics(self, records: list[dict[str, Any]]) -> None:
        """Write router metrics. Each dict = one MoE layer, appended as one JSONL line.

        Each record must contain all fields from schema §3.3:
          step, tokens_seen, timestamp_utc, layer_idx,
          expert_count, expert_fraction, router_prob_mean, router_entropy,
          normalized_router_entropy, expert_load_gini,
          max_expert_fraction, min_expert_fraction,
          dropped_tokens, raw_aux_loss, scaled_aux_loss
        """
        for rec in records:
            rec.setdefault("timestamp_utc", self._now_utc())
            self._append(self._router_path, rec)
