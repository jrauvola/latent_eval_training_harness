from __future__ import annotations

from types import SimpleNamespace

import torch

from latent_harness.evaluation.benchmarks import BenchmarkExample
from latent_harness.evaluation.reporting import append_jsonl
from latent_harness.evaluation.config import EvaluationConfig
from latent_harness.evaluation.runner import (
    _generate_baseline,
    _num_batches,
    _pretokenize_examples,
    _slice_prepared_batch,
    _should_log_batch_progress,
)


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        del return_tensors, padding, truncation, max_length
        encoded = []
        for index, prompt in enumerate(prompts, start=1):
            encoded.append([index] * min(len(prompt), 3))
        width = max(len(row) for row in encoded)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        attention = [[1 if token != self.pad_token_id else 0 for token in row] for row in padded]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(int(token)) for token in tokens.tolist())


class DummyGenerateModel:
    def __init__(self) -> None:
        self.runtime_config = SimpleNamespace(model_max_length=32)
        self.calls: list[dict] = []
        self.generate = self._generate

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        extra = torch.full((input_ids.size(0), 2), 9, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


def test_pretokenize_examples_preserves_original_prompts() -> None:
    examples = [
        BenchmarkExample("gsm8k", "ex-1", "What is 2+2?", "4", "numeric"),
        BenchmarkExample("gsm8k", "ex-2", "What is 3+3?", "6", "numeric"),
    ]
    prepared = _pretokenize_examples(
        SimpleNamespace(model_max_length=32),
        DummyTokenizer(),
        examples,
    )

    assert prepared["prompts"][0] == "What is 2+2?"
    assert prepared["input_ids"].shape[0] == 2
    assert prepared["attention_mask"].shape == prepared["input_ids"].shape


def test_slice_prepared_batch_returns_requested_range() -> None:
    prepared = {
        "input_ids": torch.tensor([[1, 1], [2, 2], [3, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.long),
    }

    batch = _slice_prepared_batch(prepared, 1, 3, device=torch.device("cpu"))

    assert torch.equal(batch["input_ids"], torch.tensor([[2, 2], [3, 0]], dtype=torch.long))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]], dtype=torch.long))


def test_generate_baseline_omits_sampling_args_when_greedy() -> None:
    model = DummyGenerateModel()
    tokenizer = DummyTokenizer()
    prepared_batch = {
        "input_ids": torch.tensor([[1, 1, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }

    predictions = _generate_baseline(
        model,
        tokenizer,
        prepared_batch,
        max_new_tokens=2,
        greedy=True,
        temperature=0.2,
        top_k=40,
        top_p=0.9,
    )

    call = model.calls[-1]
    assert call["do_sample"] is False
    assert "temperature" not in call
    assert "top_k" not in call
    assert "top_p" not in call
    assert predictions == ["9,9"]


def test_num_batches_rounds_up() -> None:
    assert _num_batches(0, 32) == 0
    assert _num_batches(1, 32) == 1
    assert _num_batches(33, 32) == 2


def test_should_log_batch_progress_logs_first_interval_and_last() -> None:
    assert _should_log_batch_progress(1, 20, 10) is True
    assert _should_log_batch_progress(10, 20, 10) is True
    assert _should_log_batch_progress(20, 20, 10) is True
    assert _should_log_batch_progress(9, 20, 10) is False
    assert _should_log_batch_progress(5, 20, 0) is False


def test_evaluation_config_accepts_progress_log_interval() -> None:
    config = EvaluationConfig.from_dict(
        {
            "runtime": {
                "batch_size": 32,
                "progress_log_interval_batches": 7,
            },
            "benchmarks": [],
            "models": [],
        }
    )

    assert config.runtime.batch_size == 32
    assert config.runtime.progress_log_interval_batches == 7


def test_append_jsonl_appends_rows(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, [{"example_id": "a"}])
    append_jsonl(path, [{"example_id": "b"}])

    assert path.read_text(encoding="utf-8").splitlines() == [
        '{"example_id": "a"}',
        '{"example_id": "b"}',
    ]
