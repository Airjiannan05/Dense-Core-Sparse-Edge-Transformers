from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_token_count(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().upper()
    suffixes = {"K": 10**3, "M": 10**6, "B": 10**9}
    if text[-1:] in suffixes:
        return int(float(text[:-1]) * suffixes[text[-1]])
    return int(text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RandomTokenStream:
    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def next_batch(self, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.batch_size, self.seq_len),
            generator=self.generator,
            device=device,
        )


class ByteTextTokenStream:
    def __init__(
        self,
        texts,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        seed: int,
    ):
        self.texts = iter(texts)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eos_id = min(vocab_size - 1, 256)
        self.buffer: list[int] = []
        random.seed(seed)

    def next_batch(self, device: torch.device) -> torch.Tensor:
        needed = self.batch_size * self.seq_len
        while len(self.buffer) < needed:
            try:
                text = next(self.texts)
            except StopIteration:
                self.texts = iter([""])
                text = ""
            encoded = [b % self.vocab_size for b in text.encode("utf-8", errors="ignore")]
            self.buffer.extend(encoded + [self.eos_id])
            if not encoded:
                self.buffer.extend([self.eos_id] * needed)

        chunk = self.buffer[:needed]
        del self.buffer[:needed]
        return torch.tensor(chunk, dtype=torch.long, device=device).view(self.batch_size, self.seq_len)


def build_token_stream(
    dataset: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    seed: int,
):
    if dataset == "random":
        return RandomTokenStream(vocab_size, seq_len, batch_size, seed)

    path = Path(dataset)
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ByteTextTokenStream([text], vocab_size, seq_len, batch_size, seed)

    if dataset == "fineweb_sample":
        try:
            from datasets import load_dataset

            ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
            return ByteTextTokenStream((row.get("text", "") for row in ds), vocab_size, seq_len, batch_size, seed)
        except Exception as exc:
            print(f"warning: could not initialize fineweb_sample ({exc}); falling back to random tokens")
            return RandomTokenStream(vocab_size, seq_len, batch_size, seed)

    raise ValueError("dataset must be 'random', 'fineweb_sample', or a local text file path")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
