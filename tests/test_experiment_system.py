from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from infrastructure.checkpoint_manager import CheckpointManager
from infrastructure.metrics_logger import MetricsLogger
from infrastructure.run_manager import RunManager

from analysis import summarize_runs


# ===========================================================================
# Dummy data helpers
# ===========================================================================


def _make_dummy_summary(run_name: str, best_val_loss: float = 2.45) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "config_name": "dense_12l_125m",
        "dataset": "data/wikitext2_train.txt",
        "total_steps": 100,
        "tokens_trained": 204800,
        "batch_size": 2,
        "seq_len": 1024,
        "best_val_loss": best_val_loss,
        "best_val_step": 90,
        "final_train_lm_loss": 2.33,
        "final_train_ppl": 10.3,
        "total_train_time_sec": 45.2,
        "peak_tok_s": 24500,
        "avg_tok_s": 21500,
        "peak_gpu_memory_mb": 15000,
        "total_parameters": 125000000,
        "active_parameters": 120000000,
        "active_flops_per_token": 118000000,
        "router_final": {"layer_0": {"entropy": 1.8, "gini": 0.03}},
    }


def _make_dummy_model_info() -> dict[str, Any]:
    return {
        "total_parameters": 125000000,
        "active_parameters": 120000000,
        "active_flops_per_token_estimate": 118000000,
        "moe_layer_indices": [0, 1, 10, 11],
        "dense_layer_indices": [2, 3, 4, 5, 6, 7, 8, 9],
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "num_experts": 8,
        "top_k": 2,
        "shared_expert": True,
        "expert_intermediate_size": 1024,
    }


def _make_dummy_router_records(
    num_experts: int = 8, num_layers: int = 4
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(num_layers):
        counts = [random.randint(50, 200) for _ in range(num_experts)]
        total = sum(counts)
        fractions = [c / total for c in counts]
        records.append({
            "step": 10,
            "tokens_seen": 20480,
            "layer_idx": i,
            "expert_count": counts,
            "expert_fraction": fractions,
            "router_prob_mean": 0.125,
            "router_entropy": 1.8,
            "normalized_router_entropy": 0.86,
            "expert_load_gini": 0.04,
            "max_expert_fraction": max(fractions),
            "min_expert_fraction": min(fractions),
            "dropped_tokens": 0,
            "raw_aux_loss": 4.0,
            "scaled_aux_loss": 0.04,
        })
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ===========================================================================
# TestMetricsLogger — JSONL 格式正确性
# ===========================================================================


class TestMetricsLogger:

    METRICS_FIELDS: dict[str, type] = {
        "step": int,
        "tokens_seen": int,
        "timestamp_utc": str,
        "loss": float,
        "lm_loss": float,
        "aux_loss": float,
        "raw_aux_loss": float,
        "ppl": float,
        "lr": float,
        "grad_norm": float,
        "tok_s": float,
        "step_time_sec": float,
        "gpu_memory_mb": float,
    }

    EVAL_FIELDS: dict[str, type] = {
        "step": int,
        "tokens_seen": int,
        "timestamp_utc": str,
        "val_loss": float,
        "val_lm_loss": float,
        "val_ppl": float,
        "eval_tokens": int,
        "eval_tok_s": float,
    }

    ROUTER_FIELDS: set[str] = {
        "step",
        "tokens_seen",
        "timestamp_utc",
        "layer_idx",
        "expert_count",
        "expert_fraction",
        "router_prob_mean",
        "router_entropy",
        "normalized_router_entropy",
        "expert_load_gini",
        "max_expert_fraction",
        "min_expert_fraction",
        "dropped_tokens",
        "raw_aux_loss",
        "scaled_aux_loss",
    }

    # ------------------------------------------------------------------
    # metrics.jsonl
    # ------------------------------------------------------------------

    def test_write_metrics_creates_valid_jsonl(self, tmp_path: Path) -> None:
        """验证 metrics.jsonl 每行可解析，包含所有必需字段且类型正确."""
        logger = MetricsLogger(tmp_path)

        for step in range(3):
            logger.write_metrics(
                step=step,
                tokens_seen=step * 1024,
                loss=2.5 - step * 0.1,
                lm_loss=2.4 - step * 0.1,
                aux_loss=0.05,
                raw_aux_loss=5.0,
                ppl=12.0 - step,
                lr=1e-4,
                grad_norm=1.0,
                tok_s=20000.0,
                step_time_sec=0.05,
                gpu_memory_mb=15000.0,
            )

        metrics_path = tmp_path / "metrics.jsonl"
        assert metrics_path.is_file()

        lines = metrics_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            for field, expected_type in self.METRICS_FIELDS.items():
                assert field in record, f"Missing field '{field}' in metrics line {i}"
                assert isinstance(record[field], expected_type), (
                    f"Field '{field}' expected {expected_type.__name__}, "
                    f"got {type(record[field]).__name__} in metrics line {i}"
                )
            assert record["step"] == i

    # ------------------------------------------------------------------
    # eval_metrics.jsonl
    # ------------------------------------------------------------------

    def test_write_eval_metrics_creates_valid_jsonl(self, tmp_path: Path) -> None:
        """验证 eval_metrics.jsonl 每行可解析，包含所有必需字段且类型正确."""
        logger = MetricsLogger(tmp_path)

        for step in (50, 100, 150):
            logger.write_eval_metrics(
                step=step,
                tokens_seen=step * 1024,
                val_loss=2.3,
                val_lm_loss=2.2,
                val_ppl=10.0,
                eval_tokens=2048,
                eval_tok_s=18000.0,
            )

        eval_path = tmp_path / "eval_metrics.jsonl"
        assert eval_path.is_file()

        lines = eval_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            for field, expected_type in self.EVAL_FIELDS.items():
                assert field in record, f"Missing field '{field}' in eval line"
                assert isinstance(record[field], expected_type), (
                    f"Field '{field}' expected {expected_type.__name__}, "
                    f"got {type(record[field]).__name__}"
                )

    # ------------------------------------------------------------------
    # router_metrics.jsonl
    # ------------------------------------------------------------------

    def test_write_router_metrics_creates_valid_jsonl(self, tmp_path: Path) -> None:
        """验证 router_metrics.jsonl — 每层一行的非嵌套结构."""
        logger = MetricsLogger(tmp_path)
        records = _make_dummy_router_records(num_experts=8, num_layers=4)
        logger.write_router_metrics(records)

        router_path = tmp_path / "router_metrics.jsonl"
        assert router_path.is_file()

        lines = router_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4

        for i, line in enumerate(lines):
            record = json.loads(line)
            for field in self.ROUTER_FIELDS:
                assert field in record, f"Missing field '{field}' in router line {i}"
            assert record["layer_idx"] == i
            # expert_count 是 list[int]
            assert isinstance(record["expert_count"], list)
            assert all(isinstance(c, int) for c in record["expert_count"])
            # expert_fraction 是 list[float]
            assert isinstance(record["expert_fraction"], list)
            assert all(isinstance(f, (int, float)) for f in record["expert_fraction"])

    # ------------------------------------------------------------------
    # 维度匹配
    # ------------------------------------------------------------------

    def test_router_metrics_dimensions_match_num_experts(self, tmp_path: Path) -> None:
        """验证 expert_count / expert_fraction 长度 == num_experts."""
        num_experts = 8
        logger = MetricsLogger(tmp_path)
        records = _make_dummy_router_records(num_experts=num_experts, num_layers=4)
        logger.write_router_metrics(records)

        router_path = tmp_path / "router_metrics.jsonl"
        lines = router_path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            record = json.loads(line)
            assert len(record["expert_count"]) == num_experts, (
                f"expert_count length {len(record['expert_count'])} != {num_experts}"
            )
            assert len(record["expert_fraction"]) == num_experts, (
                f"expert_fraction length {len(record['expert_fraction'])} != {num_experts}"
            )


# ===========================================================================
# TestRunManager — 运行目录和元数据管理
# ===========================================================================


class TestRunManager:

    @staticmethod
    def _patch_paths(tmp_path: Path, monkeypatch) -> Path:
        """Monkeypatch RunManager 的 RUNS_ROOT 和 EXPERIMENTS_CSV 到临时目录."""
        runs_root = tmp_path / "runs"
        monkeypatch.setattr(RunManager, "RUNS_ROOT", runs_root)
        monkeypatch.setattr(RunManager, "EXPERIMENTS_CSV", runs_root / "experiments.csv")
        return runs_root

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def test_init_creates_run_directory(self, tmp_path: Path, monkeypatch) -> None:
        """验证 init() 创建正确命名的目录和元数据文件."""
        import argparse

        runs_root = self._patch_paths(tmp_path, monkeypatch)

        mgr = RunManager.init(
            run_name="test_run",
            args=argparse.Namespace(),
            config_dict={"model": "dense", "hidden_size": 768},
        )

        # 目录存在且命名格式 {name}_{timestamp}
        assert mgr.run_dir.exists()
        assert mgr.run_dir.parent == runs_root
        assert mgr.run_name.startswith("test_run_")
        # 时间戳部分 (YYYYMMDD_HHMM) 至少 13 个字符
        assert len(mgr.run_name) >= len("test_run_") + 11

        # 元数据文件存在
        assert (mgr.run_dir / "run_config.yaml").is_file()
        assert (mgr.run_dir / "command.txt").is_file()
        assert (mgr.run_dir / "git_info.json").is_file()
        assert (mgr.run_dir / "env.json").is_file()

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def test_finalize_writes_summary_and_csv(self, tmp_path: Path, monkeypatch) -> None:
        """验证 finalize 生成 summary.json, summary.md, experiments.csv."""
        import argparse

        runs_root = self._patch_paths(tmp_path, monkeypatch)

        mgr = RunManager.init(
            run_name="finalize_test",
            args=argparse.Namespace(),
            config_dict={"model": "dense"},
        )
        mgr.write_dataset_info("data/test.txt", vocab_size=32000)
        mgr.write_model_info(_make_dummy_model_info())

        summary = _make_dummy_summary(mgr.run_name)
        mgr.finalize(summary)

        # summary.json
        summary_path = mgr.run_dir / "summary.json"
        assert summary_path.is_file()
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        assert loaded["run_name"] == mgr.run_name
        assert loaded["best_val_loss"] == 2.45
        assert loaded["total_parameters"] == 125000000

        # summary.md
        md_path = mgr.run_dir / "summary.md"
        assert md_path.is_file()
        md_content = md_path.read_text(encoding="utf-8")
        assert mgr.run_name in md_content
        assert "Best Val Loss" in md_content

        # experiments.csv
        csv_path = runs_root / "experiments.csv"
        assert csv_path.is_file()
        csv_lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(csv_lines) >= 2  # header + ≥1 data row
        assert csv_lines[0].startswith("run_name")

    # ------------------------------------------------------------------
    # experiments.csv 追加
    # ------------------------------------------------------------------

    def test_experiments_csv_appends(self, tmp_path: Path, monkeypatch) -> None:
        """验证多次 finalize 追加而非覆盖 experiments.csv."""
        import argparse

        runs_root = self._patch_paths(tmp_path, monkeypatch)

        # Run 1
        mgr1 = RunManager.init(
            run_name="append_test_1",
            args=argparse.Namespace(),
            config_dict={"model": "dense"},
        )
        mgr1.write_dataset_info("data/test.txt", 32000)
        mgr1.write_model_info(_make_dummy_model_info())
        mgr1.finalize(_make_dummy_summary(mgr1.run_name, best_val_loss=2.5))

        # Run 2
        mgr2 = RunManager.init(
            run_name="append_test_2",
            args=argparse.Namespace(),
            config_dict={"model": "moe"},
        )
        mgr2.write_dataset_info("data/test.txt", 32000)
        mgr2.write_model_info(_make_dummy_model_info())
        mgr2.finalize(_make_dummy_summary(mgr2.run_name, best_val_loss=2.3))

        csv_path = runs_root / "experiments.csv"
        csv_lines = csv_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(csv_lines) == 3, (
            f"Expected 3 lines (header + 2 data), got {len(csv_lines)}"
        )
        assert "append_test_1" in csv_lines[1]
        assert "append_test_2" in csv_lines[2]


# ===========================================================================
# TestSummarizeRuns — 汇总脚本
# ===========================================================================


class TestSummarizeRuns:

    def test_on_two_dummy_runs(self, tmp_path: Path, monkeypatch) -> None:
        """在两个 dummy run 上验证 summarize_runs 内部逻辑."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Run 1
        run1_dir = runs_dir / "dense_test_20260527_1200"
        run1_dir.mkdir()
        _write_json(
            run1_dir / "summary.json",
            _make_dummy_summary("dense_test_20260527_1200", best_val_loss=2.45),
        )
        _write_json(run1_dir / "model_info.json", _make_dummy_model_info())

        # Run 2
        run2_dir = runs_dir / "edge_moe_test_20260527_1300"
        run2_dir.mkdir()
        _write_json(
            run2_dir / "summary.json",
            _make_dummy_summary("edge_moe_test_20260527_1300", best_val_loss=2.15),
        )
        _write_json(run2_dir / "model_info.json", _make_dummy_model_info())

        rows = summarize_runs.collect_rows_from_directories(runs_dir)
        assert len(rows) == 2

        # 验证 best_val_loss 数值与 summary.json 一致
        run_names = {r["run_name"] for r in rows}
        assert "dense_test_20260527_1200" in run_names
        assert "edge_moe_test_20260527_1300" in run_names

        for row in rows:
            if row["run_name"] == "dense_test_20260527_1200":
                assert row["best_val_loss"] == 2.45
            elif row["run_name"] == "edge_moe_test_20260527_1300":
                assert row["best_val_loss"] == 2.15

        # 生成 markdown 并验证包含两个 run
        md = summarize_runs.generate_markdown(rows)
        assert "dense_test_20260527_1200" in md
        assert "edge_moe_test_20260527_1300" in md

    def test_markdown_table_contains_expected_columns(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """验证 markdown 表格包含所有 OUTPUT_COLUMNS 列."""
        row = summarize_runs.build_row(
            run_name="test_run",
            csv_data=None,
            summary=_make_dummy_summary("test_run"),
            model_info=_make_dummy_model_info(),
            git_commit="abc12345",
        )
        md = summarize_runs.generate_markdown([row])

        for col in summarize_runs.OUTPUT_COLUMNS:
            assert col in md, f"Column '{col}' missing from markdown output"

    def test_missing_summary_skipped(self, tmp_path: Path, monkeypatch) -> None:
        """验证 summary.json 缺失的目录被跳过而非崩溃."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # 有 summary 的目录
        good_dir = runs_dir / "good_run"
        good_dir.mkdir()
        _write_json(good_dir / "summary.json", _make_dummy_summary("good_run"))

        # 没有 summary 的目录
        bad_dir = runs_dir / "bad_run"
        bad_dir.mkdir()
        (bad_dir / "some_other_file.txt").write_text("hello", encoding="utf-8")

        rows = summarize_runs.collect_rows_from_directories(runs_dir)
        assert len(rows) == 1
        assert rows[0]["run_name"] == "good_run"


# ===========================================================================
# TestCheckpointManager — checkpoint 保存/加载
# ===========================================================================


class TestCheckpointManager:

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """验证 save 后 load 恢复完整 state_dict 和元数据."""
        ckpt_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(ckpt_dir, keep_last_n=3)

        model = nn.Linear(16, 4)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config_dict = {"hidden_size": 16, "output_size": 4}

        # 保存
        saved_path = mgr.save(
            step=100,
            tokens_seen=409600,
            model=model,
            optimizer=optimizer,
            config_dict=config_dict,
        )
        assert saved_path.exists()

        # 加载到新模型
        new_model = nn.Linear(16, 4)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        metadata = CheckpointManager.load(
            saved_path, new_model, optimizer=new_optimizer
        )

        # state_dict 一致
        orig_sd = model.state_dict()
        loaded_sd = new_model.state_dict()
        for key in orig_sd:
            assert torch.equal(orig_sd[key], loaded_sd[key]), f"Mismatch in {key}"

        # 元数据完整
        assert metadata["step"] == 100
        assert metadata["tokens_seen"] == 409600
        assert metadata["config"] == config_dict
        assert "rng_state" in metadata
        assert "python" in metadata["rng_state"]
        assert "torch" in metadata["rng_state"]
        assert "cuda" in metadata["rng_state"]  # key 存在（值可能为 None）

    def test_save_rotates_old_checkpoints(self, tmp_path: Path) -> None:
        """验证 keep_last_n=2 时旧 checkpoint 被删除."""
        ckpt_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(ckpt_dir, keep_last_n=2)

        model = nn.Linear(16, 4)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        for step in range(5):
            mgr.save(
                step=step,
                tokens_seen=step * 1024,
                model=model,
                optimizer=optimizer,
                config_dict={},
            )

        step_files = sorted(ckpt_dir.glob("step_*.pt"))
        assert len(step_files) == 2, (
            f"Expected 2 step files, got {len(step_files)}: {[f.name for f in step_files]}"
        )
        steps = [int(f.stem.split("_")[1]) for f in step_files]
        assert steps == [3, 4], f"Expected steps [3, 4], got {steps}"

    def test_best_checkpoint_not_rotated(self, tmp_path: Path) -> None:
        """验证 best.pt 不受 keep_last_n 旋转影响."""
        ckpt_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(ckpt_dir, keep_last_n=2)

        model = nn.Linear(16, 4)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 保存 best checkpoint
        mgr.save(
            step=10,
            tokens_seen=10240,
            model=model,
            optimizer=optimizer,
            config_dict={},
            is_best=True,
        )

        # 保存 3 个普通 checkpoint（超过 keep_last_n=2）
        for step in range(3):
            mgr.save(
                step=step,
                tokens_seen=step * 1024,
                model=model,
                optimizer=optimizer,
                config_dict={},
            )

        # best.pt 仍存在
        best_path = ckpt_dir / "best.pt"
        assert best_path.exists(), "best.pt should not be rotated"

        # 普通 checkpoint 保留 2 个
        step_files = sorted(ckpt_dir.glob("step_*.pt"))
        assert len(step_files) == 2

        # 总共 3 个文件
        all_pt = sorted(ckpt_dir.glob("*.pt"))
        assert len(all_pt) == 3


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
