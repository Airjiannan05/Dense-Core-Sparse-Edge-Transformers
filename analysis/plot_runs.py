#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot training metrics from metrics.jsonl and router_metrics.jsonl.

Supports single-run plotting and multi-run comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Fallback font – avoid CJK rendering issues
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass  # fallback to default


# ======================================================================
# Helpers
# ======================================================================

def load_metrics(metrics_path: Path) -> list[dict]:
    """Read metrics.jsonl, return list of dicts."""
    records: list[dict] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_router_metrics(router_path: Path) -> list[dict]:
    """Read router_metrics.jsonl, return list of dicts."""
    records: list[dict] = []
    with open(router_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_run_name(run_dir: Path) -> str:
    """Read run_name from summary.json, fallback to directory name."""
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            return summary.get("run_name", run_dir.name)
        except Exception:
            pass
    return run_dir.name


def _resolve_dirs(compare: list[str]) -> list[Path]:
    """Resolve wildcard patterns and filter valid run dirs."""
    from glob import glob as _glob

    resolved: list[Path] = []
    for pattern in compare:
        # Expand shell-style wildcards
        matches = sorted(_glob(pattern))
        if not matches:
            # Literal path (no wildcard match)
            p = Path(pattern)
            if p.is_dir():
                matches = [str(p)]
            else:
                warnings.warn(f"No directories matched pattern: {pattern}")
                continue
        for m in matches:
            mp = Path(m)
            if mp.is_dir():
                resolved.append(mp)
    # Deduplicate while preserving order
    seen = set()
    unique: list[Path] = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _valid_run_dirs(candidates: list[Path]) -> list[tuple[Path, str]]:
    """Filter candidates to those containing metrics.jsonl, returning (dir, name)."""
    valid: list[tuple[Path, str]] = []
    for d in candidates:
        if (d / "metrics.jsonl").exists():
            valid.append((d, _read_run_name(d)))
        else:
            warnings.warn(f"Skipping '{d}' — no metrics.jsonl found")
    return valid


# ======================================================================
# Chart helpers
# ======================================================================

def _setup_ax(ax: plt.Axes, title: str, xlabel: str = "Tokens Seen",
              ylabel: str = "") -> None:
    """Common axis styling."""
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if ax.get_legend() is not None:
        ax.legend(framealpha=0.9)


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save figure to output dir with tight_layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  Saved {output_dir / filename}")


# ======================================================================
# Chart 1-5: from metrics.jsonl
# ======================================================================

def _plot_metrics_line(axes: plt.Axes, x: list, y: list, label: str,
                       **kwargs) -> None:
    """Plot a single line on axes."""
    axes.plot(x, y, label=label, **kwargs)


def _make_metrics_chart(runs: list[tuple[Path, str]], output_dir: Path,
                        key: str, filename: str, title: str,
                        ylabel: str = "",
                        figsize: tuple[float, float] = (12, 4),
                        dpi: int = 150) -> None:
    """Generic chart: one line per run for a given metrics.jsonl key."""
    fig, ax = plt.subplots(figsize=figsize)
    any_data = False
    for run_dir, run_name in runs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            warnings.warn(f"metrics.jsonl not found in {run_dir}, skipping {key}")
            continue
        records = load_metrics(metrics_path)
        if not records:
            warnings.warn(f"metrics.jsonl empty in {run_dir}, skipping {key}")
            continue
        x_vals = [r["tokens_seen"] for r in records if key in r]
        y_vals = [r[key] for r in records if key in r]
        if not y_vals:
            warnings.warn(f"Key '{key}' not found in {run_dir}/metrics.jsonl")
            continue
        _plot_metrics_line(ax, x_vals, y_vals, label=run_name)
        any_data = True

    if not any_data:
        warnings.warn(f"No data for chart '{key}', skipping {filename}")
        plt.close(fig)
        return

    _setup_ax(ax, title, ylabel=ylabel)
    _save_figure(fig, output_dir, filename)


def plot_train_lm_loss(runs: list[tuple[Path, str]], output_dir: Path,
                       figsize: tuple[float, float] = (12, 4),
                       dpi: int = 150) -> None:
    _make_metrics_chart(runs, output_dir, key="lm_loss",
                        filename="train_lm_loss.png",
                        title="Training LM Loss",
                        ylabel="lm_loss",
                        figsize=figsize, dpi=dpi)


def plot_ppl(runs: list[tuple[Path, str]], output_dir: Path,
             figsize: tuple[float, float] = (12, 4),
             dpi: int = 150) -> None:
    _make_metrics_chart(runs, output_dir, key="ppl",
                        filename="ppl.png",
                        title="Perplexity",
                        ylabel="ppl",
                        figsize=figsize, dpi=dpi)


def plot_tok_s(runs: list[tuple[Path, str]], output_dir: Path,
               figsize: tuple[float, float] = (12, 4),
               dpi: int = 150) -> None:
    _make_metrics_chart(runs, output_dir, key="tok_s",
                        filename="tok_s.png",
                        title="Training Throughput",
                        ylabel="tok/s",
                        figsize=figsize, dpi=dpi)


def plot_memory(runs: list[tuple[Path, str]], output_dir: Path,
                figsize: tuple[float, float] = (12, 4),
                dpi: int = 150) -> None:
    _make_metrics_chart(runs, output_dir, key="gpu_memory_mb",
                        filename="memory.png",
                        title="GPU Memory Usage",
                        ylabel="GPU Memory (MB)",
                        figsize=figsize, dpi=dpi)


def plot_aux_loss(runs: list[tuple[Path, str]], output_dir: Path,
                  figsize: tuple[float, float] = (12, 4),
                  dpi: int = 150) -> None:
    _make_metrics_chart(runs, output_dir, key="aux_loss",
                        filename="aux_loss.png",
                        title="Auxiliary Load-Balancing Loss",
                        ylabel="aux_loss",
                        figsize=figsize, dpi=dpi)


# ======================================================================
# Chart 6: Router Entropy by Layer (from router_metrics.jsonl)
# ======================================================================

def plot_router_entropy_by_layer(
    runs: list[tuple[Path, str]],
    output_dir: Path,
    figsize: tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> None:
    """For each run, plot one line per layer_idx showing router_entropy vs step."""
    fig, ax = plt.subplots(figsize=figsize)
    any_data = False

    for run_dir, run_name in runs:
        rpath = run_dir / "router_metrics.jsonl"
        if not rpath.exists():
            warnings.warn(f"router_metrics.jsonl not found in {run_dir}, skipping router_entropy")
            continue
        records = load_router_metrics(rpath)
        if not records:
            warnings.warn(f"router_metrics.jsonl empty in {run_dir}, skipping router_entropy")
            continue

        # Group by layer_idx
        layers: dict[int, list[tuple[int, float]]] = {}
        for rec in records:
            if "router_entropy" not in rec or "layer_idx" not in rec:
                continue
            lidx = rec["layer_idx"]
            step = rec.get("step", rec.get("tokens_seen", 0))
            layers.setdefault(lidx, []).append((step, rec["router_entropy"]))

        for lidx in sorted(layers.keys()):
            points = layers[lidx]
            points.sort(key=lambda t: t[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if len(runs) == 1:
                label = f"layer_{lidx}"
            else:
                label = f"{run_name}/layer_{lidx}"
            ax.plot(xs, ys, label=label)
            any_data = True

    if not any_data:
        warnings.warn("No router_entropy data found, skipping router_entropy_by_layer.png")
        plt.close(fig)
        return

    _setup_ax(ax, "Router Entropy by Layer", xlabel="Step", ylabel="router_entropy")
    _save_figure(fig, output_dir, "router_entropy_by_layer.png")


# ======================================================================
# Chart 7: Expert Load Heatmap (from router_metrics.jsonl)
# ======================================================================

def plot_expert_load_heatmap(
    runs: list[tuple[Path, str]],
    output_dir: Path,
    figsize: tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> None:
    """Heatmap of expert_fraction from the last recorded step.

    Comparison mode: one subplot per run (1 row, N cols).
    """
    valid_runs: list[tuple[Path, str, list[dict]]] = []
    for run_dir, run_name in runs:
        rpath = run_dir / "router_metrics.jsonl"
        if not rpath.exists():
            warnings.warn(f"router_metrics.jsonl not found in {run_dir}, skipping heatmap")
            continue
        records = load_router_metrics(rpath)
        if not records:
            warnings.warn(f"router_metrics.jsonl empty in {run_dir}, skipping heatmap")
            continue
        # Only keep records with expert_fraction
        ef_records = [r for r in records if "expert_fraction" in r and "layer_idx" in r]
        if not ef_records:
            warnings.warn(f"No expert_fraction in {run_dir}/router_metrics.jsonl")
            continue
        valid_runs.append((run_dir, run_name, ef_records))

    if not valid_runs:
        warnings.warn("No expert_fraction data, skipping expert_load_heatmap.png")
        return

    n_runs = len(valid_runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(figsize[0] * n_runs, figsize[1]),
                             squeeze=False)

    for col, (run_dir, run_name, ef_records) in enumerate(valid_runs):
        ax = axes[0, col]

        # Group by layer_idx
        by_layer: dict[int, list[float]] = {}
        last_step: dict[int, int] = {}
        for rec in ef_records:
            lidx = rec["layer_idx"]
            step = rec.get("step", 0)
            if lidx not in last_step or step > last_step[lidx]:
                last_step[lidx] = step
                by_layer[lidx] = rec["expert_fraction"]

        layers_sorted = sorted(by_layer.keys())
        if not layers_sorted:
            warnings.warn(f"No expert_fraction layers in {run_dir}")
            ax.set_visible(False)
            continue

        # Build matrix: rows=layers, cols=experts
        num_experts = len(by_layer[layers_sorted[0]])
        matrix = np.zeros((len(layers_sorted), num_experts))
        for i, lidx in enumerate(layers_sorted):
            ef = by_layer[lidx]
            matrix[i, :len(ef)] = ef[:num_experts]

        # Determine vmax
        vmax = max(0.3, np.max(matrix))

        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > vmax / 2 else "black")

        ax.set_title(f"{run_name}", fontsize=12)
        ax.set_xlabel("Expert Index", fontsize=10)
        ax.set_ylabel("Layer Index", fontsize=10)
        ax.set_yticks(range(len(layers_sorted)))
        ax.set_yticklabels([str(l) for l in layers_sorted])
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels([str(e) for e in range(num_experts)], fontsize=8)

        # Colorbar per subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("expert_fraction", fontsize=9)

    fig.suptitle("Expert Load Distribution (Final Step)", fontsize=14)
    _save_figure(fig, output_dir, "expert_load_heatmap.png")


# ======================================================================
# Main orchestration
# ======================================================================

def _make_all_charts(runs: list[tuple[Path, str]], output_dir: Path,
                     figsize: tuple[float, float], dpi: int) -> None:
    """Generate all 7 charts for the given runs."""
    print(f"Generating charts -> {output_dir.resolve()}")

    plot_train_lm_loss(runs, output_dir, figsize, dpi)
    plot_ppl(runs, output_dir, figsize, dpi)
    plot_tok_s(runs, output_dir, figsize, dpi)
    plot_memory(runs, output_dir, figsize, dpi)
    plot_aux_loss(runs, output_dir, figsize, dpi)
    plot_router_entropy_by_layer(runs, output_dir, figsize, dpi)
    plot_expert_load_heatmap(runs, output_dir, figsize, dpi)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from metrics.jsonl / router_metrics.jsonl"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, default=None,
                       help="Single run directory")
    group.add_argument("--compare", nargs="+", default=None,
                       help="Multiple run directories (supports wildcards)")
    parser.add_argument("--output", type=str, default="plots",
                        help="Output directory (default: plots/)")
    parser.add_argument("--figsize", type=str, default="12,4",
                        help="Figure size as width,height (default: 12,4)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")

    args = parser.parse_args()

    # Parse figsize
    try:
        w, h = args.figsize.split(",")
        figsize = (float(w.strip()), float(h.strip()))
    except Exception:
        warnings.warn(f"Invalid figsize '{args.figsize}', using (12,4)")
        figsize = (12.0, 4.0)

    output_dir = Path(args.output)

    if args.dir is not None:
        run_dir = Path(args.dir)
        if not (run_dir / "metrics.jsonl").exists():
            print(f"Error: metrics.jsonl not found in {run_dir}", file=sys.stderr)
            sys.exit(1)
        runs = [(run_dir, _read_run_name(run_dir))]
    else:
        candidates = _resolve_dirs(args.compare)
        runs = _valid_run_dirs(candidates)
        if not runs:
            print("Error: no valid run directories found (need metrics.jsonl)",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Comparing {len(runs)} run(s): {', '.join(name for _, name in runs)}")

    _make_all_charts(runs, output_dir, figsize, args.dpi)


if __name__ == "__main__":
    main()
