from __future__ import annotations

import argparse

from model import DenseCoreSparseEdgeConfig, DenseCoreSparseEdgeTransformer
from model.metrics import estimate_active_flops_per_token
from utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = DenseCoreSparseEdgeConfig.from_dict(load_yaml(args.config)["model"])
    model = DenseCoreSparseEdgeTransformer(cfg)
    print(f"total_parameters={model.total_parameters()}")
    print(f"active_parameters={model.active_parameters()}")
    print(f"active_flops_per_token_estimate={estimate_active_flops_per_token(model)}")
    print(f"moe_layers={sorted(model.moe_layers)}")


if __name__ == "__main__":
    main()
