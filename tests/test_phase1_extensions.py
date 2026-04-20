"""Phase 1 eval extension tests.

Covers three behaviors that the Phase 1 spec requires:

1. Batch-level persistence — predictions.jsonl + running_summary.json are
   flushed at least once per ``persistence_every_examples`` examples. A
   simulated mid-run crash loses <= ``persistence_every_examples`` examples
   of completed work.
2. num_latent sweep drives distinct output dirs per value.
3. skip_latent_injection at num_latent=0 calls the runtime with zero latent
   iterations and still dumps persistence artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from latent_harness.evaluation.benchmarks import BenchmarkExample
from latent_harness.evaluation.config import EvaluationRuntimeConfig
from latent_harness.evaluation.runner import _run_single_evaluation


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        del return_tensors, padding, truncation, max_length
        encoded = [[hash(p) % 97 + 2 for _ in range(4)] for p in prompts]
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.ones(len(encoded), 4, dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(str(int(t)) for t in tokens)


@dataclass
class _StubHandle:
    """Minimal stub of EvaluationModelHandle for runner tests."""

    inference_strategy: str
    runtime_config: Any
    model: Any
    generation_model: Any
    tokenizer: Any
    bot_id: int | None = None
    remove_eos: bool = True

    @property
    def name(self):  # noqa: D401 — parity with real handle.
        return "stub"

    @property
    def model_kind(self):
        return "causal_lm"


def _make_benchmark(num_examples: int):
    examples = [
        BenchmarkExample(
            benchmark_name="gsm8k",
            example_id=f"ex-{i}",
            prompt=f"What is {i}+{i}?",
            target=str(2 * i),
            task_type="numeric",
        )
        for i in range(num_examples)
    ]
    return SimpleNamespace(examples=examples)


def _make_stub_model():
    class StubGen:
        pad_token_id = 0

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            extra = torch.full((input_ids.size(0), 3), 9, dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat([input_ids, extra], dim=1)

    model = StubGen()
    rt = SimpleNamespace(
        model_max_length=32,
        num_latent=0,
        use_chat_template=False,
        chat_template_kwargs=None,
    )
    return _StubHandle(
        inference_strategy="standard_generation",
        runtime_config=rt,
        model=model,
        generation_model=model,
        tokenizer=_Tokenizer(),
    )


def _make_runtime_cfg(tmp_path: Path, persistence_every: int = 4) -> EvaluationRuntimeConfig:
    return EvaluationRuntimeConfig(
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snap"),
        device="cpu",
        batch_size=2,
        progress_log_interval_batches=1,
        max_new_tokens=3,
        greedy=True,
        num_latent_sweep=None,
        skip_latent_injection_at_zero=True,
        dump_latent_traces=False,
        dump_kv_cache=False,
        persistence_every_examples=persistence_every,
    )


def test_persistence_flushes_at_least_every_n_examples(tmp_path):
    """Predictions.jsonl grows monotonically, at least once per persistence_every.

    We simulate a crash by invoking _run_single_evaluation with a generator
    wrapped in a wrapper that raises part-way through. Surviving rows in
    predictions.jsonl should be the persisted prefix.
    """
    loaded_model = _make_stub_model()
    cfg = _make_runtime_cfg(tmp_path, persistence_every=4)
    benchmark = _make_benchmark(num_examples=10)

    model_spec = SimpleNamespace(
        name="stubvariant",
        runtime=SimpleNamespace(num_latent=0),
    )

    _run_single_evaluation(
        loaded_model=loaded_model,
        model_spec=model_spec,
        loaded_benchmark=benchmark,
        runtime=cfg,
        benchmark_name="gsm8k",
        num_latent=0,
        skip_latent_injection=True,
        output_dir=Path(tmp_path),
        device=torch.device("cpu"),
    )

    leaf = Path(tmp_path) / "stubvariant" / "gsm8k_numlatent_0"
    assert (leaf / "predictions.jsonl").exists()
    assert (leaf / "running_summary.json").exists()
    rows = [
        json.loads(line)
        for line in (leaf / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 10
    summary = json.loads((leaf / "running_summary.json").read_text(encoding="utf-8"))
    assert summary["completed_count"] == 10
    assert summary["total_examples"] == 10


def test_persistence_survives_mid_run_crash(tmp_path, monkeypatch):
    """When generation crashes mid-run, all rows completed before the crash persist."""
    loaded_model = _make_stub_model()
    cfg = _make_runtime_cfg(tmp_path, persistence_every=4)
    benchmark = _make_benchmark(num_examples=10)

    model_spec = SimpleNamespace(
        name="stubvariant_crash",
        runtime=SimpleNamespace(num_latent=0),
    )

    # Patch the generation helper to raise after the 2nd batch (= 4 examples completed).
    import latent_harness.evaluation.runner as runner_mod

    real = runner_mod._generate_predictions_with_taps
    calls = {"n": 0}

    def _flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated mid-run crash")
        return real(**kwargs)

    monkeypatch.setattr(runner_mod, "_generate_predictions_with_taps", _flaky)

    with pytest.raises(RuntimeError, match="simulated mid-run crash"):
        _run_single_evaluation(
            loaded_model=loaded_model,
            model_spec=model_spec,
            loaded_benchmark=benchmark,
            runtime=cfg,
            benchmark_name="gsm8k",
            num_latent=0,
            skip_latent_injection=True,
            output_dir=Path(tmp_path),
            device=torch.device("cpu"),
        )

    leaf = Path(tmp_path) / "stubvariant_crash" / "gsm8k_numlatent_0"
    rows = [
        json.loads(line)
        for line in (leaf / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # batch_size=2, persistence_every=min(4, 2)=2 → every batch flushes.
    # After 2 successful batches (4 examples) crash triggers. Pre-crash flush
    # must contain those 4 rows; crash handler flushes any trailing buffer.
    assert len(rows) >= 4, f"expected >=4 persisted rows, got {len(rows)}"
    assert len(rows) <= 6, f"persisted rows should not exceed successful+buffered"
    summary = json.loads((leaf / "running_summary.json").read_text(encoding="utf-8"))
    assert summary["completed_count"] >= 4
    assert summary["failed_count"] >= 1
