from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def _get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


class CheckpointManager:
    """Manages checkpoint save/load with rotating cleanup and full recovery metadata.

    Usage::

        ckpt_mgr = CheckpointManager(run_dir / "checkpoint", keep_last_n=3)

        # Save during training
        ckpt_mgr.save(step=1000, tokens_seen=4096000, model=model,
                      optimizer=optimizer, config_dict=cfg_dict)

        # Save best checkpoint
        ckpt_mgr.save(step=5000, tokens_seen=20480000, model=model,
                      optimizer=optimizer, config_dict=cfg_dict, is_best=True)

        # Load
        info = CheckpointManager.load(Path(".../step_001000.pt"),
                                      model, optimizer=optimizer)
    """

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Path to the checkpoint directory (e.g. runs/{name}/checkpoint/).
            keep_last_n: Number of recent regular checkpoints to retain. best.pt is
                never pruned.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self._saved_steps: list[int] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        tokens_seen: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config_dict: dict[str, Any],
        rng_state: dict[str, Any] | None = None,
        is_best: bool = False,
        dataset_hash: str | None = None,
    ) -> Path:
        """Save a full checkpoint and return its file path.

        Args:
            step: Current training step.
            tokens_seen: Total tokens processed so far.
            model: The model whose state_dict will be saved.
            optimizer: The optimizer whose state_dict will be saved.
            config_dict: Model / run configuration dictionary.
            rng_state: Optional extra RNG state to merge into the built-in RNG block.
            is_best: If True, save as best.pt (never pruned by rotation).
            dataset_hash: Optional SHA-256 hash of the dataset for reproducibility.

        Returns:
            The Path to the saved checkpoint file.
        """
        # --- Build RNG state block ---
        full_rng: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if rng_state:
            full_rng.update(rng_state)

        # --- Build checkpoint dictionary ---
        checkpoint: dict[str, Any] = {
            "step": step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_dict,
            "git_commit": _get_git_commit(),
            "dataset_hash": dataset_hash,
            "rng_state": full_rng,
        }

        # --- Determine file path ---
        if is_best:
            filepath = self.checkpoint_dir / "best.pt"
        else:
            filepath = self.checkpoint_dir / f"step_{step:06d}.pt"
            self._saved_steps.append(step)

        # --- Atomic write via temporary file ---
        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(filepath)

        # --- Rotating cleanup of regular checkpoints ---
        if not is_best:
            self._rotating_cleanup()

        return filepath

    @staticmethod
    def load(
        path: Path,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint and restore model (and optionally optimizer) state.

        Args:
            path: Path to the .pt checkpoint file.
            model: Model instance to load state_dict into.
            optimizer: Optional optimizer instance to load state_dict into.
            map_location: Device string passed to torch.load.

        Returns:
            A dictionary containing the checkpoint metadata (step, tokens_seen,
            config, rng_state, etc.).
        """
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=map_location, weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "step": checkpoint.get("step"),
            "tokens_seen": checkpoint.get("tokens_seen"),
            "config": checkpoint.get("config"),
            "git_commit": checkpoint.get("git_commit"),
            "dataset_hash": checkpoint.get("dataset_hash"),
            "rng_state": checkpoint.get("rng_state"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotating_cleanup(self) -> None:
        """Remove oldest regular checkpoints when exceeding keep_last_n."""
        while len(self._saved_steps) > self.keep_last_n:
            oldest_step = self._saved_steps.pop(0)
            oldest_path = self.checkpoint_dir / f"step_{oldest_step:06d}.pt"
            if oldest_path.exists():
                oldest_path.unlink()
