from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ===========================================================================
# Output schema
# ===========================================================================

OUTPUT_COLUMNS: list[str] = [
    "run_name",
    "config",
    "params(M)",
    "active_params(M)",
    "FLOPs/Token",
    "total_steps",
    "tokens_trained",
    "best_val_loss",
    "best_val_ppl",
    "final_train_ppl",
    "peak_tok_s",
    "avg_tok_s",
    "peak_gpu_memory_mb",
    "train_time(s)",
    "git_commit",
]

COLUMN_FORMATS: dict[str, str] = {
    "params(M)": ".1f",
    "active_params(M)": ".1f",
    "best_val_loss": ".4f",
    "best_val_ppl": ".4f",
    "final_train_ppl": ".4f",
    "peak_tok_s": ".0f",
    "avg_tok_s": ".0f",
    "peak_gpu_memory_mb": ".0f",
    "train_time(s)": ".1f",
}


# ===========================================================================
# Helpers
# ===========================================================================


def format_number(value: Any, fmt_spec: str) -> str:
    """Format a numeric value; return "N/A" on failure."""
    if value is None or value == "" or value == "N/A":
        return "N/A"
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return "N/A"
        return f"{num:{fmt_spec}}"
    except (ValueError, TypeError):
        return str(value)


def format_int_comma(value: Any) -> str:
    """Format an integer with thousands separator."""
    if value is None or value == "" or value == "N/A":
        return "N/A"
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def format_git_commit(commit: Any) -> str:
    """Truncate commit hash to first 8 characters."""
    if not commit:
        return "N/A"
    s = str(commit).strip()
    if not s:
        return "N/A"
    return s[:8]


def format_cell(col: str, value: Any) -> str:
    """Format a single table cell based on column type."""
    if value is None or value == "":
        return "N/A"

    if col == "FLOPs/Token":
        return format_int_comma(value)

    if col == "git_commit":
        return str(format_git_commit(value))

    if col in COLUMN_FORMATS:
        return format_number(value, COLUMN_FORMATS[col])

    # Generic: run_name, config, total_steps, tokens_trained
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


# ===========================================================================
# File loaders
# ===========================================================================


def resolve_run_dir(runs_dir: Path, run_name: str) -> Path | None:
    """Find the run sub-directory for *run_name*.

    Tries an exact match first, then falls back to prefix matching so that
    manually shortened names (without the timestamp suffix) still resolve.
    """
    if not runs_dir.is_dir():
        return None

    exact = runs_dir / run_name
    if exact.is_dir():
        return exact

    for entry in sorted(runs_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith(run_name):
            return entry
    return None


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on any error."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_summary_json(runs_dir: Path, run_name: str) -> dict[str, Any] | None:
    run_dir = resolve_run_dir(runs_dir, run_name)
    if run_dir is None:
        return None
    return load_json_file(run_dir / "summary.json")


def load_model_info(runs_dir: Path, run_name: str) -> dict[str, Any] | None:
    run_dir = resolve_run_dir(runs_dir, run_name)
    if run_dir is None:
        return None
    return load_json_file(run_dir / "model_info.json")


def load_git_commit(runs_dir: Path, run_name: str) -> str | None:
    run_dir = resolve_run_dir(runs_dir, run_name)
    if run_dir is None:
        return None
    info = load_json_file(run_dir / "git_info.json")
    if info is None:
        return None
    return info.get("commit")


def read_experiments_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read *experiments.csv* as a list of dicts (empty list if missing)."""
    if not csv_path.is_file():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ===========================================================================
# Data merging
# ===========================================================================


def compute_best_val_ppl(data: dict[str, Any]) -> float | None:
    """Return best validation perplexity.

    Priority: direct ``best_val_ppl`` / ``val_ppl`` field → exp(best_val_loss).
    """
    for key in ("best_val_ppl", "val_ppl"):
        val = data.get(key)
        if val is not None and val != "" and val != "N/A":
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    best_val_loss = data.get("best_val_loss")
    if best_val_loss is not None and best_val_loss != "" and best_val_loss != "N/A":
        try:
            return math.exp(float(best_val_loss))
        except (ValueError, TypeError, OverflowError):
            pass

    return None


def _get_field(*keys: str, sources: list[dict[str, Any] | None]) -> Any:
    """Return the first non-empty value from the sources for any of *keys*."""
    for d in sources:
        if d is None:
            continue
        for k in keys:
            v = d.get(k)
            if v is not None and v != "":
                return v
    return None


def build_row(
    run_name: str,
    csv_data: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    model_info: dict[str, Any] | None,
    git_commit: str | None,
) -> dict[str, Any]:
    """Merge all data sources into a single row dict keyed by ``OUTPUT_COLUMNS``."""

    sources = [summary, csv_data, model_info]

    total_params = _get_field("total_parameters", sources=sources)
    active_params = _get_field("active_parameters", sources=sources)
    flops = _get_field("active_flops_per_token", sources=sources)
    best_val_loss = _get_field("best_val_loss", sources=sources)

    # Combine all available dicts for perplexity calculation
    merged_for_ppl: dict[str, Any] = {}
    for d in sources:
        if d:
            merged_for_ppl.update(d)

    return {
        "run_name": run_name,
        "config": _get_field("config_name", sources=sources),
        "params(M)": float(total_params) / 1e6 if total_params is not None else None,
        "active_params(M)": float(active_params) / 1e6 if active_params is not None else None,
        "FLOPs/Token": int(flops) if flops is not None else None,
        "total_steps": _get_field("total_steps", sources=sources),
        "tokens_trained": _get_field("tokens_trained", sources=sources),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_val_ppl": compute_best_val_ppl(merged_for_ppl),
        "final_train_ppl": (
            float(v) if (v := _get_field("final_train_ppl", sources=sources)) is not None else None
        ),
        "peak_tok_s": _get_field("peak_tok_s", sources=sources),
        "avg_tok_s": _get_field("avg_tok_s", sources=sources),
        "peak_gpu_memory_mb": _get_field("peak_gpu_memory_mb", sources=sources),
        "train_time(s)": _get_field("total_train_time_sec", sources=sources),
        "git_commit": format_git_commit(git_commit),
    }


# ===========================================================================
# Data collection
# ===========================================================================


def collect_rows_from_csv(runs_dir: Path) -> list[dict[str, Any]]:
    """Primary path: read ``experiments.csv``, enrich with per-run JSON files."""
    csv_path = runs_dir / "experiments.csv"
    csv_rows = read_experiments_csv(csv_path)
    if not csv_rows:
        return []

    results: list[dict[str, Any]] = []
    for csv_row in csv_rows:
        run_name = csv_row.get("run_name", "").strip()
        if not run_name:
            continue

        summary = load_summary_json(runs_dir, run_name)
        if summary is None:
            continue  # skip rows whose run directory / summary.json is missing

        model_info = load_model_info(runs_dir, run_name)
        git_commit = csv_row.get("git_commit") or load_git_commit(runs_dir, run_name)

        results.append(build_row(run_name, csv_row, summary, model_info, git_commit))

    return results


def collect_rows_from_directories(runs_dir: Path) -> list[dict[str, Any]]:
    """Fallback: walk sub-directories of *runs_dir*, read ``summary.json``."""
    results: list[dict[str, Any]] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        if not summary_path.is_file():
            continue

        run_name = entry.name
        summary = load_json_file(summary_path)
        if summary is None:
            continue

        model_info = load_model_info(runs_dir, run_name)
        git_commit = load_git_commit(runs_dir, run_name)

        results.append(build_row(run_name, None, summary, model_info, git_commit))

    return results


# ===========================================================================
# Sorting
# ===========================================================================


def sort_key(row: dict[str, Any]) -> tuple[int, float]:
    """Sort rows by ``best_val_loss`` ascending; ``N/A`` values go last."""
    val = row.get("best_val_loss")
    if val is None:
        return (1, 0.0)
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, 0.0)


# ===========================================================================
# Output generators
# ===========================================================================


def generate_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the comparison table in Markdown format."""
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = [
        "# Experiment Comparison",
        "",
        f"Generated: {now}",
        "",
    ]

    # Header + separator
    header = "| " + " | ".join(OUTPUT_COLUMNS) + " |"
    separator = "|" + "|".join(" --- " for _ in OUTPUT_COLUMNS) + "|"
    lines.append(header)
    lines.append(separator)

    sorted_rows = sorted(rows, key=sort_key)
    for row in sorted_rows:
        cells = [format_cell(col, row.get(col)) for col in OUTPUT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def generate_csv_output(rows: list[dict[str, Any]]) -> str:
    """Render the comparison table in CSV format."""
    import io

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=sort_key):
        formatted = {col: format_cell(col, row.get(col)) for col in OUTPUT_COLUMNS}
        writer.writerow(formatted)
    return out.getvalue()


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize experiment runs into a comparison table."
    )
    parser.add_argument(
        "--dir",
        default="runs",
        help="Path to runs root directory (default: runs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "csv"],
        help="Output format (default: markdown)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.dir)

    # --- Collect rows ---
    csv_path = runs_dir / "experiments.csv"
    if csv_path.is_file():
        rows = collect_rows_from_csv(runs_dir)
    else:
        rows = collect_rows_from_directories(runs_dir)

    # --- Render ---
    if not rows:
        if args.format == "csv":
            output = ",".join(OUTPUT_COLUMNS) + "\n"
        else:
            output = "No experiments found.\n"
    else:
        if args.format == "csv":
            output = generate_csv_output(rows)
        else:
            output = generate_markdown(rows)

    # --- Write ---
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
