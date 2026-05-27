from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()

    counts = defaultdict(Counter)
    probs = defaultdict(float)
    n = defaultdict(int)
    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            layer = row["layer"]
            for experts, expert_probs in zip(row["topk_experts"], row["topk_probs"]):
                for expert, prob in zip(experts, expert_probs):
                    counts[layer][expert] += 1
                    probs[(layer, expert)] += prob
                    n[(layer, expert)] += 1

    for layer in sorted(counts):
        total = sum(counts[layer].values())
        print(f"layer={layer}")
        for expert, count in sorted(counts[layer].items()):
            mean_prob = probs[(layer, expert)] / max(1, n[(layer, expert)])
            print(f"  expert={expert} load={count / max(1, total):.4f} mean_topk_prob={mean_prob:.4f}")


if __name__ == "__main__":
    main()
