from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class RunManager:
    """Orchestrates a single training run's output directory.

    Usage:
        mgr = RunManager.init("edge_moe_12l", args, config_dict)
        mgr.write_dataset_info(dataset_info)
        mgr.write_model_info(model_info)
        # ... training loop ...
        mgr.finalize(summary_dict)
    """

    RUNS_ROOT = Path("runs")
    EXPERIMENTS_CSV = RUNS_ROOT / "experiments.csv"

    def __init__(self, run_dir: Path, run_name: str) -> None:
        self.run_dir = run_dir
        self.run_name = run_name

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def init(cls, run_name: str, args: Any, config_dict: dict[str, Any]) -> "RunManager":
        """Create run directory and write initial metadata files.

        Args:
            run_name: short name (e.g. "edge_moe_12l"); timestamp is appended.
            args: parsed argparse.Namespace from train.py.
            config_dict: merged model configuration dict.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        full_name = f"{run_name}_{timestamp}"
        run_dir = cls.RUNS_ROOT / full_name
        run_dir.mkdir(parents=True, exist_ok=True)

        mgr = cls(run_dir, full_name)

        # run_config.yaml — merged config
        _write_yaml(run_dir / "run_config.yaml", config_dict)

        # command.txt
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

        # git_info.json
        _write_json(run_dir / "git_info.json", _git_info())

        # env.json
        _write_json(run_dir / "env.json", _env_info())

        return mgr

    # ------------------------------------------------------------------
    # Metadata writers (call after model/dataset are built)
    # ------------------------------------------------------------------
    def write_dataset_info(self, dataset_path: str, vocab_size: int, **extra) -> None:
        """Write dataset_info.json."""
        info: dict[str, Any] = {
            "path": dataset_path,
            "vocab_size": vocab_size,
        }
        info.update(extra)
        # Try sha256 for local files
        dp = Path(dataset_path)
        if dp.is_file():
            info["size_bytes"] = dp.stat().st_size
            info["sha256"] = _sha256_file(dp)
        _write_json(self.run_dir / "dataset_info.json", info)

    def write_model_info(self, info: dict[str, Any]) -> None:
        """Write model_info.json. The caller computes all fields per schema §3.4."""
        _write_json(self.run_dir / "model_info.json", info)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self, summary: dict[str, Any]) -> None:
        """Write summary.json, summary.md, and append to experiments.csv.

        summary dict must contain all fields from schema §3.5.
        """
        summary.setdefault("run_name", self.run_name)
        _write_json(self.run_dir / "summary.json", summary)
        _write_md(self.run_dir / "summary.md", _render_summary_md(summary))
        _append_experiments_csv(self.EXPERIMENTS_CSV, summary)


# ======================================================================
# Internal helpers
# ======================================================================

def _write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def _write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = False
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        dirty = bool(status.strip())
    except Exception:
        pass

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def _env_info() -> dict[str, Any]:
    import torch
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    # ROCm detection
    try:
        info["rocm_version"] = torch.version.hip
    except Exception:
        pass
    return info


def _render_summary_md(s: dict[str, Any]) -> str:
    """Render summary.md from summary dict."""
    lines = [
        f"# Run Summary: {s.get('run_name', 'unknown')}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Config | {s.get('config_name', 'N/A')} |",
        f"| Dataset | {s.get('dataset', 'N/A')} |",
        f"| Total Steps | {s.get('total_steps', 'N/A')} |",
        f"| Tokens Trained | {s.get('tokens_trained', 'N/A'):,} |",
        f"| Batch Size | {s.get('batch_size', 'N/A')} |",
        f"| Sequence Length | {s.get('seq_len', 'N/A')} |",
        f"| Best Val Loss | {s.get('best_val_loss', 'N/A')} |",
        f"| Best Val Step | {s.get('best_val_step', 'N/A')} |",
        f"| Final Train LM Loss | {s.get('final_train_lm_loss', 'N/A')} |",
        f"| Final Train PPL | {s.get('final_train_ppl', 'N/A')} |",
        f"| Total Train Time (s) | {s.get('total_train_time_sec', 'N/A'):.1f} |",
        f"| Peak tok/s | {s.get('peak_tok_s', 'N/A')} |",
        f"| Avg tok/s | {s.get('avg_tok_s', 'N/A')} |",
        f"| Peak GPU Memory (MB) | {s.get('peak_gpu_memory_mb', 'N/A')} |",
        f"| Total Parameters | {s.get('total_parameters', 'N/A'):,} |",
        f"| Active Parameters | {s.get('active_parameters', 'N/A'):,} |",
        f"| Active FLOPs/Token | {s.get('active_flops_per_token', 'N/A'):,} |",
        "",
    ]
    # Router final section
    router = s.get("router_final", {})
    if router:
        lines.append("## Router Final State")
        lines.append("")
        lines.append("| Layer | Entropy | Gini |")
        lines.append("|-------|---------|------|")
        for layer_name, vals in router.items():
            lines.append(f"| {layer_name} | {vals.get('entropy', 'N/A'):.4f} | {vals.get('gini', 'N/A'):.4f} |")
        lines.append("")
    return "\n".join(lines)


_EXPERIMENTS_CSV_HEADER = (
    "run_name,config_name,dataset,total_steps,tokens_trained,batch_size,seq_len,"
    "best_val_loss,best_val_step,final_train_lm_loss,final_train_ppl,"
    "total_train_time_sec,peak_tok_s,avg_tok_s,peak_gpu_memory_mb,"
    "total_parameters,active_parameters,active_flops_per_token,"
    "git_commit,branch\n"
)


def _append_experiments_csv(csv_path: Path, summary: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    # Read git info for commit
    git_info = {}
    git_info_path = csv_path.parent / summary.get("run_name", "unknown") / "git_info.json"
    if git_info_path.exists():
        try:
            git_info = json.loads(git_info_path.read_text())
        except Exception:
            pass

    row = (
        f"{summary.get('run_name', '')},"
        f"{summary.get('config_name', '')},"
        f"{summary.get('dataset', '')},"
        f"{summary.get('total_steps', '')},"
        f"{summary.get('tokens_trained', '')},"
        f"{summary.get('batch_size', '')},"
        f"{summary.get('seq_len', '')},"
        f"{summary.get('best_val_loss', '')},"
        f"{summary.get('best_val_step', '')},"
        f"{summary.get('final_train_lm_loss', '')},"
        f"{summary.get('final_train_ppl', '')},"
        f"{summary.get('total_train_time_sec', '')},"
        f"{summary.get('peak_tok_s', '')},"
        f"{summary.get('avg_tok_s', '')},"
        f"{summary.get('peak_gpu_memory_mb', '')},"
        f"{summary.get('total_parameters', '')},"
        f"{summary.get('active_parameters', '')},"
        f"{summary.get('active_flops_per_token', '')},"
        f"{git_info.get('commit', '')},"
        f"{git_info.get('branch', '')}\n"
    )

    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(_EXPERIMENTS_CSV_HEADER)
        f.write(row)
