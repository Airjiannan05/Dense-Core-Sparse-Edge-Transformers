from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Token encoding
# ---------------------------------------------------------------------------

def encode_text(text: str, tokenizer: str, vocab_size: int = 32000) -> list[int]:
    """Encode *text* to token IDs matching the project tokenizer behaviour.

    ``utf8`` mirrors :class:`ByteTextTokenStream` — each UTF-8 byte modulo
    *vocab_size*.  ``gpt2`` uses the HuggingFace GPT-2 tokenizer (optional
    import).
    """
    if tokenizer == "utf8":
        return [b % vocab_size for b in text.encode("utf-8", errors="ignore")]
    elif tokenizer == "gpt2":
        from transformers import GPT2Tokenizer

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return tok.encode(text)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


# ---------------------------------------------------------------------------
# Subsequence matching (grep-style)
# ---------------------------------------------------------------------------

def find_token_indices(trace_tokens: list[int], query_tokens: list[int]) -> list[int]:
    """Return every start index in *trace_tokens* where *query_tokens* appears
    as a contiguous subsequence (naïve sliding-window)."""
    if not query_tokens:
        return []
    n, m = len(trace_tokens), len(query_tokens)
    if m > n:
        return []
    starts: list[int] = []
    for i in range(n - m + 1):
        if trace_tokens[i : i + m] == query_tokens:
            starts.append(i)
    return starts


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _flatten_experts(experts: list[list[int]]) -> list[int]:
    flat: list[int] = []
    for row in experts:
        flat.extend(row)
    return flat


def jaccard_similarity(
    experts_a: list[list[int]], experts_b: list[list[int]]
) -> float:
    """Count-aware Jaccard similarity between two expert-assignment lists.

    Flattens each list of *topk* expert slots into a multiset, then computes
    :math:`|A ∩ B| / |A ∪ B|` where intersection uses *min* counts and union
    uses *max* counts.
    """
    flat_a = _flatten_experts(experts_a)
    flat_b = _flatten_experts(experts_b)
    if not flat_a and not flat_b:
        return 0.0

    keys = set(flat_a) | set(flat_b)
    count_a = {k: flat_a.count(k) for k in keys}
    count_b = {k: flat_b.count(k) for k in keys}

    inter = sum(min(count_a[k], count_b[k]) for k in keys)
    union = sum(max(count_a[k], count_b[k]) for k in keys)
    return inter / max(1, union)


def expert_usage_distribution(
    experts: list[list[int]], num_experts: int
) -> np.ndarray:
    """Return a length-*num_experts* vector of per-expert selection ratios."""
    flat = _flatten_experts(experts)
    dist = np.zeros(num_experts, dtype=np.float64)
    if not flat:
        return dist
    for e in flat:
        dist[e] += 1.0
    dist /= dist.sum()
    return dist


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


# ---------------------------------------------------------------------------
# Trace loading & indexing
# ---------------------------------------------------------------------------

def load_trace(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_trace_by_step(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Return ``{step: [row, ...]}`` preserving insertion order."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
    return dict(by_step)


def build_layer_map(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Return ``{layer_idx: row}`` for a single step's rows."""
    return {row["layer"]: row for row in rows}


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_consistency(
    trace_rows: list[dict[str, Any]],
    paraphrase_groups: list[dict[str, Any]],
    tokenizer: str,
    vocab_size: int,
) -> dict[str, Any]:
    """Run the full route-consistency analysis.

    Returns a dict suitable for report rendering.
    """
    # ---- index trace by step ------------------------------------------------
    by_step = group_trace_by_step(trace_rows)
    all_steps = sorted(by_step.keys())

    # ---- encode all paraphrase texts ----------------------------------------
    encoded_groups: list[dict[str, Any]] = []
    for g in paraphrase_groups:
        group_id = g["id"]
        texts = g.get("texts", [])
        encoded_texts = []
        for t in texts:
            encoded_texts.append(encode_text(t, tokenizer, vocab_size))
        encoded_groups.append({"id": group_id, "texts": texts, "encoded": encoded_texts})

    # ---- collect all layers seen in trace -----------------------------------
    all_layers: set[int] = set()
    for row in trace_rows:
        all_layers.add(row["layer"])
    layers = sorted(all_layers)

    # Determine num_experts from trace data
    num_experts = 0
    for row in trace_rows:
        for experts in row.get("topk_experts", []):
            if experts:
                num_experts = max(num_experts, max(experts) + 1)
    if num_experts == 0:
        num_experts = 8  # sensible default

    # ---- per-layer per-pair results -----------------------------------------
    # layer_results[layer] = {"intra": [...], "inter": [...]}
    layer_results: dict[int, dict[str, list[float]]] = {
        layer: {"intra": [], "inter": []} for layer in layers
    }

    # group_detail[group_id] = list of pair dicts
    group_details: dict[str, list[dict[str, Any]]] = {}

    # ---- match each text in trace and extract expert assignments ------------
    # Build a lookup: (step, text_idx_within_group) -> token positions
    # Strategy: prefer same-step matches, fall back to any step

    # For each group, for each text, find matching positions per layer per step
    # match_map[group_idx][text_idx] = [(step, start_pos, {layer: experts}), ...]
    match_map: list[list[list[dict[str, Any]]]] = []

    for gi, g in enumerate(encoded_groups):
        group_matches: list[list[dict[str, Any]]] = []
        for ti, encoded in enumerate(g["encoded"]):
            if not encoded:
                group_matches.append([])
                continue
            text_matches: list[dict[str, Any]] = []
            for step in all_steps:
                step_rows = by_step[step]
                # Find any row that has tokens (all rows in same step share tokens)
                if not step_rows:
                    continue
                trace_tokens = step_rows[0]["tokens"]
                starts = find_token_indices(trace_tokens, encoded)
                for start in starts:
                    # Extract expert assignments for each layer
                    layer_experts: dict[int, list[list[int]]] = {}
                    for row in step_rows:
                        lyr = row["layer"]
                        experts = row.get("topk_experts")
                        if experts is None:
                            continue
                        # slice the experts for the matched token range
                        sliced = experts[start : start + len(encoded)]
                        layer_experts[lyr] = sliced
                    text_matches.append(
                        {
                            "step": step,
                            "start": start,
                            "layer_experts": layer_experts,
                        }
                    )
            group_matches.append(text_matches)
        match_map.append(group_matches)

    # ---- compute intra-group similarities -----------------------------------
    for gi, g in enumerate(encoded_groups):
        group_id = g["id"]
        n_texts = len(g["texts"])
        pair_details: list[dict[str, Any]] = []

        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                matches_i = match_map[gi][i]
                matches_j = match_map[gi][j]

                if not matches_i or not matches_j:
                    continue

                # Pick the best matching pair (prefer same step)
                best_i = matches_i[0]
                best_j = matches_j[0]
                for mi in matches_i:
                    for mj in matches_j:
                        if mi["step"] == mj["step"]:
                            best_i = mi
                            best_j = mj
                            break
                    if best_i["step"] == best_j["step"]:
                        break

                pair_entry = {
                    "i": i,
                    "j": j,
                    "text_a": g["texts"][i],
                    "text_b": g["texts"][j],
                    "step_a": best_i["step"],
                    "step_b": best_j["step"],
                    "layers": {},
                }

                for layer in layers:
                    experts_a = best_i["layer_experts"].get(layer)
                    experts_b = best_j["layer_experts"].get(layer)
                    if experts_a is None or experts_b is None:
                        pair_entry["layers"][layer] = None
                        continue

                    jac = jaccard_similarity(experts_a, experts_b)
                    dist_a = expert_usage_distribution(experts_a, num_experts)
                    dist_b = expert_usage_distribution(experts_b, num_experts)
                    cos = cosine_similarity(dist_a, dist_b)

                    pair_entry["layers"][layer] = {"jaccard": jac, "cosine": cos}
                    layer_results[layer]["intra"].append((jac, cos))

                pair_details.append(pair_entry)

        group_details[group_id] = pair_details

    # ---- compute inter-group similarities -----------------------------------
    # Compare texts from different groups
    n_groups = len(encoded_groups)
    for ga in range(n_groups):
        for gb in range(ga + 1, n_groups):
            for ma in match_map[ga]:
                for mb in match_map[gb]:
                    if not ma or not mb:
                        continue
                    mi = ma[0]  # first match for text in group A
                    mj = mb[0]  # first match for text in group B
                    for layer in layers:
                        experts_a = mi["layer_experts"].get(layer)
                        experts_b = mj["layer_experts"].get(layer)
                        if experts_a is None or experts_b is None:
                            continue
                        jac = jaccard_similarity(experts_a, experts_b)
                        layer_results[layer]["inter"].append(jac)

    # ---- aggregate layer stats ----------------------------------------------
    layer_agg: list[dict[str, Any]] = []
    for layer in layers:
        intra = layer_results[layer]["intra"]
        inter = layer_results[layer]["inter"]

        avg_intra_jac = float(np.mean([x[0] for x in intra])) if intra else 0.0
        avg_intra_cos = float(np.mean([x[1] for x in intra])) if intra else 0.0
        avg_inter_jac = float(np.mean(inter)) if inter else 0.0

        ratio = avg_intra_jac / avg_inter_jac if avg_inter_jac > 0 else 0.0

        layer_agg.append(
            {
                "layer": layer,
                "intra_jaccard": avg_intra_jac,
                "intra_cosine": avg_intra_cos,
                "inter_jaccard": avg_inter_jac,
                "ratio": ratio,
            }
        )

    return {
        "layers": layer_agg,
        "group_details": group_details,
        "num_groups": len(encoded_groups),
        "total_texts": sum(len(g["texts"]) for g in encoded_groups),
        "avg_texts_per_group": (
            sum(len(g["texts"]) for g in encoded_groups) / max(1, len(encoded_groups))
        ),
        "num_layers": len(layers),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_report(
    analysis: dict[str, Any],
    trace_path: str,
    paraphrase_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# Route Consistency Analysis")
    lines.append("")
    lines.append(f"**Trace**: `{trace_path}`")
    lines.append(
        f"**Paraphrase Groups**: {analysis['num_groups']} groups, "
        f"{analysis['avg_texts_per_group']:.1f} texts each (avg)"
    )
    lines.append(f"**Timestamp**: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Per-layer summary table
    lines.append("## Per-Layer Route Consistency")
    lines.append("")
    header = (
        "| Layer | Intra-Group Jaccard | Intra-Group Cosine "
        "| Inter-Group Jaccard | Intra/Inter Ratio |"
    )
    sep = (
        "|-------|---------------------|-------------------"
        "|---------------------|-------------------|"
    )
    lines.append(header)
    lines.append(sep)
    for row in analysis["layers"]:
        lines.append(
            f"| {row['layer']:<5} "
            f"| {row['intra_jaccard']:<19.3f} "
            f"| {row['intra_cosine']:<17.3f} "
            f"| {row['inter_jaccard']:<19.3f} "
            f"| {row['ratio']:<17.3f} |"
        )
    lines.append("")

    # Group-level detail
    lines.append("## Group-Level Detail")
    lines.append("")

    for group_id, pairs in analysis["group_details"].items():
        if not pairs:
            lines.append(f"### Group \"{group_id}\"")
            lines.append("")
            lines.append("_(no valid token matches found)_")
            lines.append("")
            continue

        lines.append(f"### Group \"{group_id}\"")
        lines.append("")

        # Build header dynamically from first pair's layer keys
        first_pair_layers = {
            k for p in pairs for k in (p.get("layers") or {}) if p["layers"].get(k)
        }
        sorted_layers = sorted(first_pair_layers)

        header_cols = "| Text Pair |"
        sep_cols = "|-----------|"
        for lyr in sorted_layers:
            header_cols += f" Layer {lyr} Jac | Layer {lyr} Cos |"
            sep_cols += "-------------|-------------|"
        lines.append(header_cols)
        lines.append(sep_cols)

        for p in pairs:
            row_str = f"| {p['i']}↔{p['j']} |"
            for lyr in sorted_layers:
                ld = p["layers"].get(lyr)
                if ld:
                    row_str += f" {ld['jaccard']:<11.3f} | {ld['cosine']:<11.3f} |"
                else:
                    row_str += " N/A         | N/A         |"
            lines.append(row_str)

        lines.append("")

    # Missing / skipped annotations
    lines.append("## Notes")
    lines.append("")
    missing_count = sum(
        1 for p_list in analysis["group_details"].values() for p in p_list
        if any(v is None for v in (p.get("layers") or {}).values())
    )
    if missing_count:
        lines.append(
            f"- {missing_count} text pair(s) have at least one layer with missing "
            f"routing data (skipped those layers)."
        )
    else:
        lines.append("- All layers had routing data for all matched pairs.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MoE route consistency across paraphrase groups."
    )
    parser.add_argument("--trace", required=True, help="Path to routing_trace.jsonl")
    parser.add_argument(
        "--paraphrase-groups",
        required=True,
        help="Path to paraphrases.json with 'groups' key",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path (default: print to stdout)",
    )
    parser.add_argument(
        "--tokenizer",
        default="utf8",
        choices=["utf8", "gpt2"],
        help="Token encoding method (default: utf8)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocab size for utf8 tokenizer (default: 32000)",
    )
    args = parser.parse_args()

    # Load paraphrase groups
    with open(args.paraphrase_groups, "r", encoding="utf-8") as f:
        pg_data = json.load(f)
    paraphrase_groups: list[dict[str, Any]] = pg_data.get("groups", [])
    if not paraphrase_groups:
        print("warning: no paraphrase groups found in input file.", file=sys.stderr)

    # Load trace
    trace_rows = load_trace(args.trace)
    if not trace_rows:
        print("error: trace file is empty or unreadable.", file=sys.stderr)
        sys.exit(1)

    # Analyze
    analysis = analyze_consistency(
        trace_rows, paraphrase_groups, args.tokenizer, args.vocab_size
    )

    # Render
    report = render_report(analysis, args.trace, args.paraphrase_groups)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
