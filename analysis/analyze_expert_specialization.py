from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Token encoding (shared with analyze_route_consistency.py)
# ---------------------------------------------------------------------------

def encode_text(text: str, tokenizer: str = "utf8", vocab_size: int = 32000) -> list[int]:
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
# Subsequence matching (naïve sliding-window)
# ---------------------------------------------------------------------------

def find_token_indices(trace_tokens: list[int], query_tokens: list[int]) -> list[int]:
    """Return every start index in *trace_tokens* where *query_tokens* appears
    as a contiguous subsequence."""
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
# Trace loading & indexing
# ---------------------------------------------------------------------------

def load_trace(path: str) -> list[dict[str, Any]]:
    """Load routing_trace.jsonl into a list of row dicts."""
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


# ---------------------------------------------------------------------------
# Labeled prompts loading
# ---------------------------------------------------------------------------

def load_labeled_prompts(path: str) -> list[dict[str, Any]]:
    """Load labeled_prompts.jsonl into a list of dicts with 'text' and 'label' keys."""
    prompts: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mutual_information(counts: np.ndarray) -> tuple[float, float]:
    """Compute MI and normalized MI (NMI) from co-occurrence counts.

    Parameters
    ----------
    counts : np.ndarray, shape (num_experts, num_labels)

    Returns
    -------
    (mi, nmi) : tuple[float, float]
        *mi* in bits; *nmi* normalized to [0, 1] via
        ``MI / sqrt(H_experts * H_labels)``.
    """
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0

    joint = counts / total                     # P(e, l)
    p_e = joint.sum(axis=1, keepdims=True)      # P(e)
    p_l = joint.sum(axis=0, keepdims=True)      # P(l)

    denom = p_e @ p_l                           # P(e) * P(l)
    eps = 1e-12
    mi_terms = joint * np.log2((joint + eps) / (denom + eps))
    mi = float(mi_terms.sum())

    h_experts = float(-np.sum(p_e * np.log2(p_e + eps)))
    h_labels = float(-np.sum(p_l * np.log2(p_l + eps)))

    if h_experts > 0 and h_labels > 0:
        nmi = mi / np.sqrt(h_experts * h_labels)
    else:
        nmi = 0.0

    return mi, nmi


# ---------------------------------------------------------------------------
# Expert–label preference (lift)
# ---------------------------------------------------------------------------

def expert_label_preferences(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-expert label preference metrics from co-occurrence counts.

    Parameters
    ----------
    counts : np.ndarray, shape (num_experts, num_labels)

    Returns
    -------
    p_l_given_e : np.ndarray, shape (num_experts, num_labels)
        ``P(label | expert)`` — confidence.
    lift : np.ndarray, shape (num_experts, num_labels)
        ``P(label | expert) / P(label)``.
    """
    total = counts.sum()
    if total == 0:
        num_experts, num_labels = counts.shape
        return np.zeros((num_experts, num_labels)), np.zeros((num_experts, num_labels))

    joint = counts / total
    p_e = joint.sum(axis=1, keepdims=True)      # (num_experts, 1)
    p_l = joint.sum(axis=0, keepdims=True)      # (1, num_labels)

    # P(label | expert)
    p_l_given_e = np.where(p_e > 0, joint / p_e, 0.0)

    # lift = P(label | expert) / P(label)
    lift = np.where(p_l > 0, p_l_given_e / p_l, 0.0)

    return p_l_given_e, lift


# ---------------------------------------------------------------------------
# Core analysis: build co-occurrence matrices & compute MI
# ---------------------------------------------------------------------------

def analyze_specialization(
    trace_rows: list[dict[str, Any]],
    labeled_prompts: list[dict[str, Any]],
    tokenizer: str,
    vocab_size: int,
    num_experts: int,
    min_samples: int,
) -> dict[str, Any]:
    """Run the full expert specialization analysis.

    Returns a dict suitable for report rendering.
    """
    # ---- step 0: validate inputs --------------------------------------------
    if not trace_rows:
        return {"error": "trace file is empty", "exit_code": 0}

    if not labeled_prompts:
        return {"error": "no labeled prompts found", "exit_code": 0}

    # ---- step 1: apply min-samples threshold & build label index -------------
    label_counts: dict[str, int] = defaultdict(int)
    for p in labeled_prompts:
        label_counts[p.get("label", "unknown")] += 1

    valid_labels: set[str] = set()
    has_merged = False
    for label, count in label_counts.items():
        if count >= min_samples:
            valid_labels.add(label)
        else:
            has_merged = True

    labels: list[str] = sorted(valid_labels)
    if has_merged:
        labels.append("other")

    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    # ---- step 2: encode labeled prompts -------------------------------------
    encoded_prompts: list[dict[str, Any]] = []
    for p in labeled_prompts:
        raw_label = p.get("label", "unknown")
        label = raw_label if raw_label in valid_labels else "other"
        tokens = encode_text(p["text"], tokenizer, vocab_size)
        encoded_prompts.append(
            {
                "tokens": tokens,
                "label": label,
                "text": p["text"],
            }
        )

    # ---- step 3: group trace by step & detect layers ------------------------
    by_step = group_trace_by_step(trace_rows)
    all_layers: set[int] = set()
    for row in trace_rows:
        all_layers.add(row["layer"])
    layers: list[int] = sorted(all_layers)

    # ---- step 4: infer num_experts from trace if not specified ---------------
    if num_experts <= 0:
        for row in trace_rows:
            for experts in row.get("topk_experts", []):
                if experts:
                    num_experts = max(num_experts, max(experts) + 1)
    if num_experts <= 0:
        num_experts = 8

    num_labels = len(labels)

    # ---- step 5: build co-occurrence matrices per layer ---------------------
    C: dict[int, np.ndarray] = {
        layer: np.zeros((num_experts, num_labels), dtype=np.float64)
        for layer in layers
    }

    # Track matched-token counts per layer for reporting
    matched_tokens_per_layer: dict[int, int] = defaultdict(int)

    matched_prompts = 0
    skipped_prompts = 0
    total_prompt_matches = 0

    for ep in encoded_prompts:
        query = ep["tokens"]
        if not query:
            skipped_prompts += 1
            continue

        label_idx = label_to_idx[ep["label"]]
        qlen = len(query)
        found_any = False

        for step, step_rows in by_step.items():
            if not step_rows:
                continue
            trace_tokens = step_rows[0].get("tokens")
            if trace_tokens is None:
                continue
            starts = find_token_indices(trace_tokens, query)

            for start in starts:
                found_any = True
                total_prompt_matches += 1
                for row in step_rows:
                    layer = row["layer"]
                    experts = row.get("topk_experts")
                    if experts is None:
                        continue
                    # Slice expert assignments for the matched token range
                    end = start + qlen
                    if end > len(experts):
                        end = len(experts)
                    sliced = experts[start:end]
                    for token_experts in sliced:
                        for e in token_experts:
                            if 0 <= e < num_experts:
                                C[layer][e][label_idx] += 1
                                matched_tokens_per_layer[layer] += 1

        if found_any:
            matched_prompts += 1
        else:
            skipped_prompts += 1

    # ---- step 6: check for MoE layers ---------------------------------------
    moe_layers_found = any(C[layer].sum() > 0 for layer in layers)
    if not moe_layers_found:
        return {
            "error": "no MoE layer routing data found in trace (all co-occurrence matrices are empty)",
            "exit_code": 0,
        }

    # ---- step 7: compute MI per layer ---------------------------------------
    layer_mi: list[dict[str, Any]] = []
    for layer in layers:
        counts = C[layer]
        mi, nmi = mutual_information(counts)

        # Compute p(label|expert) and lift for this layer
        p_lge, lift = expert_label_preferences(counts)

        # Find top-3 specialized experts (highest lift for any label)
        top_specialized: list[dict[str, Any]] = []
        for e in range(num_experts):
            best_label_idx = int(np.argmax(lift[e]))
            best_lift = float(lift[e][best_label_idx])
            best_conf = float(p_lge[e][best_label_idx])
            if best_lift > 0:
                top_specialized.append(
                    {
                        "expert": e,
                        "label": labels[best_label_idx],
                        "lift": best_lift,
                        "confidence": best_conf,
                    }
                )
        # Sort by lift descending, take top 3
        top_specialized.sort(key=lambda x: x["lift"], reverse=True)
        top3 = top_specialized[:3]

        layer_mi.append(
            {
                "layer": layer,
                "mi": mi,
                "nmi": nmi,
                "matched_tokens": matched_tokens_per_layer.get(layer, 0),
                "top3": top3,
                "p_l_given_e": p_lge,
                "lift": lift,
                "single_label": num_labels <= 1,
            }
        )

    return {
        "layers": layers,
        "labels": labels,
        "num_experts": num_experts,
        "layer_mi": layer_mi,
        "matched_prompts": matched_prompts,
        "skipped_prompts": skipped_prompts,
        "total_prompts": len(encoded_prompts),
        "total_prompt_matches": total_prompt_matches,
        "has_merged": has_merged,
        "min_samples": min_samples,
        "label_counts": dict(label_counts),
    }


# ---------------------------------------------------------------------------
# Markdown report rendering
# ---------------------------------------------------------------------------

def _format_top3(top3: list[dict[str, Any]]) -> str:
    """Format top-3 specialized experts as a compact string."""
    parts: list[str] = []
    for item in top3:
        parts.append(f"E{item['expert']}({item['label']}:{item['lift']:.1f}x)")
    return ", ".join(parts) if parts else "—"


def render_report(
    analysis: dict[str, Any],
    trace_path: str,
    labeled_path: str,
) -> str:
    """Render the specialization analysis to a Markdown string."""
    lines: list[str] = []

    # ---- header -------------------------------------------------------------
    lines.append("# Expert Specialization Analysis")
    lines.append("")
    lines.append(f"**Trace**: `{trace_path}`")
    lines.append(
        f"**Labeled Prompts**: {analysis['total_prompts']} prompts, "
        f"{len(analysis['labels'])} labels"
    )
    if analysis.get("has_merged"):
        lines.append(
            f"**Note**: labels with < {analysis['min_samples']} samples merged into "
            f"\"other\" (label counts: {analysis['label_counts']})"
        )
    lines.append(f"**Timestamp**: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # ---- matching summary ---------------------------------------------------
    lines.append(
        f"**Matched**: {analysis['matched_prompts']} prompts "
        f"({analysis.get('total_prompt_matches', 0)} total matches), "
        f"**Skipped**: {analysis['skipped_prompts']} prompts (token sequence not found in trace)"
    )
    lines.append("")

    # ---- per-layer MI table -------------------------------------------------
    lines.append("## Per-Layer Mutual Information")
    lines.append("")

    single_label_note = any(lm["single_label"] for lm in analysis["layer_mi"])

    header = (
        "| Layer | Mutual Info (bits) | Normalized MI | #Matched Tokens "
        "| Top-3 Specialized Experts |"
    )
    sep = (
        "|-------|-------------------|---------------|-----------------"
        "|---------------------------|"
    )
    lines.append(header)
    lines.append(sep)

    for lm in analysis["layer_mi"]:
        lines.append(
            f"| {lm['layer']:<5} "
            f"| {lm['mi']:<17.4f} "
            f"| {lm['nmi']:<13.4f} "
            f"| {lm['matched_tokens']:<15} "
            f"| {_format_top3(lm['top3']):<25} |"
        )
    lines.append("")

    if single_label_note:
        lines.append(
            "> **Note**: some layers have only one effective label; "
            "NMI is 0.0 because H_labels = 0."
        )
        lines.append("")

    # ---- per-layer heatmaps -------------------------------------------------
    lines.append("## Expert-Label Heatmaps")
    lines.append("")
    lines.append("> Values = P(label | expert). Higher = more specialized.")
    lines.append("")

    labels = analysis["labels"]
    for lm in analysis["layer_mi"]:
        layer = lm["layer"]
        p_lge = lm["p_l_given_e"]

        lines.append(f"### Layer {layer}")
        lines.append("")

        # Build header
        header_cols = "| Expert |"
        sep_cols = "|--------|"
        for lbl in labels:
            header_cols += f" {lbl} |"
            sep_cols += "---------|"
        lines.append(header_cols)
        lines.append(sep_cols)

        for e in range(analysis["num_experts"]):
            row_str = f"| E{e:<5} |"
            for li in range(len(labels)):
                row_str += f" {p_lge[e][li]:.3f}   |"
            lines.append(row_str)

        lines.append("")

    # ---- top specializations ------------------------------------------------
    lines.append("## Top Specializations")
    lines.append("")

    for lm in analysis["layer_mi"]:
        layer = lm["layer"]
        top3 = lm["top3"]

        lines.append(f"### Layer {layer}")
        lines.append("")

        if not top3:
            lines.append("_(no significant specializations detected)_")
            lines.append("")
            continue

        for item in top3:
            lines.append(
                f"- **Expert {item['expert']}** → {item['label']} "
                f"(lift={item['lift']:.1f}x, confidence={item['confidence']:.3f})"
            )
        lines.append("")

    # ---- notes --------------------------------------------------------------
    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- Mutual information computed from token-level expert–label "
        f"co-occurrence across {analysis['num_experts']} experts × "
        f"{len(analysis['labels'])} labels."
    )
    if analysis.get("has_merged"):
        lines.append(
            f"- Labels with fewer than {analysis['min_samples']} samples "
            f"were merged into \"other\"."
        )
    lines.append(
        f"- NMI normalization: ``MI / sqrt(H_experts * H_labels)``."
    )
    lines.append(
        "- Lift = P(label | expert) / P(label); >1 means the expert is "
        "over-represented for that label relative to chance."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MoE expert specialization via mutual information."
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to routing_trace.jsonl",
    )
    parser.add_argument(
        "--labeled-prompts",
        required=True,
        help="Path to labeled prompts JSONL file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path (default: print to stdout)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts per MoE layer (default: 8; 0 = infer from trace)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per label; labels below threshold merged into 'other' (default: 10)",
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

    # ---- load trace ---------------------------------------------------------
    trace_rows = load_trace(args.trace)
    if not trace_rows:
        print("error: trace file is empty or unreadable.", file=sys.stderr)
        sys.exit(1)

    # ---- load labeled prompts -----------------------------------------------
    labeled_prompts = load_labeled_prompts(args.labeled_prompts)
    if not labeled_prompts:
        print("error: labeled prompts file is empty or unreadable.", file=sys.stderr)
        sys.exit(1)

    # ---- validate labeled prompts format ------------------------------------
    for i, p in enumerate(labeled_prompts):
        if "text" not in p or "label" not in p:
            print(
                f"error: labeled prompt at line {i + 1} missing 'text' or 'label' key.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ---- analyze ------------------------------------------------------------
    analysis = analyze_specialization(
        trace_rows=trace_rows,
        labeled_prompts=labeled_prompts,
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        num_experts=args.num_experts,
        min_samples=args.min_samples,
    )

    # ---- handle early-exit conditions ---------------------------------------
    if "error" in analysis:
        print(f"info: {analysis['error']}", file=sys.stderr)
        sys.exit(analysis.get("exit_code", 0))

    # ---- render report ------------------------------------------------------
    report = render_report(analysis, args.trace, args.labeled_prompts)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
