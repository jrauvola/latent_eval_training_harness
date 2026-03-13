from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from codi_reimplementation.benchmarks import get_benchmark_spec, load_benchmark
from codi_reimplementation.benchmarks.types import BenchmarkExample
from codi_reimplementation.config import dump_yamlable, ensure_dir, load_yaml_config
from codi_reimplementation.eval.metrics import score_prediction
from codi_reimplementation.eval.reporting import append_jsonl, write_csv, write_markdown_summary
from codi_reimplementation.models import EvalModelSpec, load_eval_model


@dataclass(slots=True)
class EvalRuntimeConfig:
    output_dir: str = "artifacts/eval/default"
    cache_dir: str = ".cache/huggingface"
    snapshot_dir: str = "artifacts/datasets"
    device: str = "auto"
    batch_size: int = 8
    max_examples_per_benchmark: int | None = None
    comparison_modes: list[str] = field(default_factory=lambda: ["latent_cot"])
    max_new_tokens: int = 128
    greedy: bool = True
    temperature: float = 0.1
    top_k: int = 40
    top_p: float = 0.95
    num_passes: int = 1


@dataclass(slots=True)
class EvalConfig:
    runtime: EvalRuntimeConfig
    benchmarks: list[str]
    models: list[EvalModelSpec]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalConfig":
        runtime = EvalRuntimeConfig(**payload.get("runtime", {}))
        benchmarks = payload.get("benchmarks", [])
        model_specs = [EvalModelSpec(**model_payload) for model_payload in payload.get("models", [])]
        return cls(runtime=runtime, benchmarks=benchmarks, models=model_specs)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _chunked(items: list[BenchmarkExample], batch_size: int) -> list[list[BenchmarkExample]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _generate_baseline(
    model,
    tokenizer,
    prompts: list[str],
    *,
    device: torch.device,
    mode: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[str]:
    prompt_suffix = ""
    if mode == "direct_answer":
        prompt_suffix = "\nAnswer with only the final answer."
    elif mode == "verbal_cot":
        prompt_suffix = "\nLet's think step by step."
    prepared_prompts = [prompt + prompt_suffix for prompt in prompts]
    batch = tokenizer(
        prepared_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.runtime_config.model_max_length,
    ).to(device)

    with torch.no_grad():
        generated = model.codi.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    predictions: list[str] = []
    for index, sequence in enumerate(generated):
        new_tokens = sequence[batch["input_ids"].size(1) :]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return predictions


def _generate_predictions(
    *,
    model,
    tokenizer,
    batch_examples: list[BenchmarkExample],
    mode: str,
    device: torch.device,
    config: EvalRuntimeConfig,
) -> list[str]:
    prompts = [example.prompt for example in batch_examples]
    if mode == "latent_cot":
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.runtime_config.model_max_length,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        if model.runtime_config.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long, device=device).expand(input_ids.size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long, device=device).expand(input_ids.size(0), 2)
        input_ids = torch.cat([input_ids, bot_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)

        with torch.no_grad():
            return model.generate_from_latent(
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inf_latent_iterations=model.runtime_config.num_latent,
                max_new_tokens=config.max_new_tokens,
                greedy=config.greedy,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )

    return _generate_baseline(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        mode=mode,
        max_new_tokens=config.max_new_tokens,
        greedy=config.greedy,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )


def run_evaluation_from_config(config_path: str) -> None:
    payload = load_yaml_config(config_path)
    config = EvalConfig.from_dict(payload)
    device = _resolve_device(config.runtime.device)
    output_dir = ensure_dir(config.runtime.output_dir)
    ensure_dir(config.runtime.snapshot_dir)

    summary_rows: list[dict[str, Any]] = []

    run_metadata_path = output_dir / "resolved_config.json"
    run_metadata_path.write_text(json.dumps(dump_yamlable(payload), indent=2), encoding="utf-8")

    predictions_path = output_dir / "predictions.jsonl"
    # Start fresh so a re-run overwrites; incremental appends happen below
    if predictions_path.exists():
        predictions_path.unlink()

    for model_spec in config.models:
        model = load_eval_model(model_spec, device=device)
        tokenizer = model.build_tokenizer()

        for benchmark_name in config.benchmarks:
            loaded = load_benchmark(
                get_benchmark_spec(benchmark_name),
                cache_dir=config.runtime.cache_dir,
                snapshot_dir=config.runtime.snapshot_dir,
                max_examples=config.runtime.max_examples_per_benchmark,
            )

            for mode in config.runtime.comparison_modes:
                scored_rows: list[dict[str, Any]] = []
                latencies: list[float] = []
                char_counts: list[int] = []

                for batch_examples in _chunked(loaded.examples, config.runtime.batch_size):
                    started = time.perf_counter()
                    predictions = _generate_predictions(
                        model=model,
                        tokenizer=tokenizer,
                        batch_examples=batch_examples,
                        mode=mode,
                        device=device,
                        config=config.runtime,
                    )
                    elapsed = time.perf_counter() - started
                    per_example_latency = elapsed / max(len(batch_examples), 1)

                    for example, prediction in zip(batch_examples, predictions):
                        score = score_prediction(example, prediction)
                        latencies.append(per_example_latency)
                        char_counts.append(len(prediction))
                        row = {
                            "model": model_spec.name,
                            "benchmark": benchmark_name,
                            "mode": mode,
                            "example_id": example.example_id,
                            "task_type": example.task_type,
                            "target": example.target,
                            "prediction_text": prediction,
                            "parsed_prediction": score.parsed_prediction.value,
                            "prediction_valid": score.parsed_prediction.is_valid,
                            "correct": score.correct,
                            "latency_s": per_example_latency,
                        }
                        scored_rows.append(row)

                accuracy = sum(1 for row in scored_rows if row["correct"]) / max(len(scored_rows), 1)
                summary_row = {
                    "model": model_spec.name,
                    "benchmark": benchmark_name,
                    "mode": mode,
                    "accuracy": accuracy,
                    "num_examples": len(scored_rows),
                    "avg_latency_s": sum(latencies) / max(len(latencies), 1),
                    "avg_prediction_chars": sum(char_counts) / max(len(char_counts), 1),
                }
                summary_rows.append(summary_row)

                # Checkpoint: write after each (model, benchmark, mode) so a crash loses at most one run
                prediction_rows_for_mode = [row | {"prompt": example.prompt} for row, example in zip(scored_rows, loaded.examples[: len(scored_rows)])]
                append_jsonl(predictions_path, prediction_rows_for_mode)
                write_csv(output_dir / "summary.csv", summary_rows)
                write_markdown_summary(output_dir / "summary.md", summary_rows)
