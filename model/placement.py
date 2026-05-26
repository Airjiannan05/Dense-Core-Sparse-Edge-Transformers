from __future__ import annotations

import random


def is_moe_layer(
    layer_idx: int,
    num_layers: int,
    front_moe_layers: int,
    back_moe_layers: int,
) -> bool:
    if layer_idx < front_moe_layers:
        return True
    if layer_idx >= num_layers - back_moe_layers:
        return True
    return False


def get_moe_layers(
    strategy: str,
    num_layers: int,
    n_moe_layers: int | None = None,
    seed: int = 0,
    front_moe_layers: int | None = None,
    back_moe_layers: int | None = None,
) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    strategy = strategy.lower()
    if strategy == "dense":
        return []
    if strategy == "full_moe":
        return list(range(num_layers))

    if strategy == "edge_moe" and (front_moe_layers is not None or back_moe_layers is not None):
        front = front_moe_layers or 0
        back = back_moe_layers or 0
        _validate_count(front + back, num_layers)
        return list(range(front)) + list(range(num_layers - back, num_layers))

    if n_moe_layers is None:
        raise ValueError(f"n_moe_layers is required for placement '{strategy}'")
    _validate_count(n_moe_layers, num_layers)

    if strategy == "early_moe":
        return list(range(n_moe_layers))
    if strategy == "late_moe":
        return list(range(num_layers - n_moe_layers, num_layers))
    if strategy == "middle_moe":
        start = (num_layers - n_moe_layers) // 2
        return list(range(start, start + n_moe_layers))
    if strategy == "edge_moe":
        front = n_moe_layers // 2
        back = n_moe_layers - front
        return list(range(front)) + list(range(num_layers - back, num_layers))
    if strategy == "random_moe":
        rng = random.Random(seed)
        return sorted(rng.sample(range(num_layers), n_moe_layers))

    raise ValueError(f"unknown MoE placement strategy: {strategy}")


def _validate_count(n_moe_layers: int, num_layers: int) -> None:
    if n_moe_layers < 0:
        raise ValueError("n_moe_layers must be non-negative")
    if n_moe_layers > num_layers:
        raise ValueError("n_moe_layers cannot exceed num_layers")
