from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from latent_harness.core.io import dump_yamlable, ensure_dir, load_yaml_config
from latent_harness.evaluation.benchmarks import BenchmarkExample, get_benchmark_spec, load_benchmark
from latent_harness.evaluation.config import EvaluationConfig, EvaluationRuntimeConfig
from latent_harness.evaluation.metrics import score_prediction
from latent_harness.evaluation.models import load_evaluation_model
from latent_harness.evaluation.reporting import append_jsonl, write_csv, write_markdown_summary


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _chunked(items: list[BenchmarkExample], batch_size: int) -> list[list[BenchmarkExample]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _log_event(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[eval {timestamp}] {message}", flush=True)


def _num_batches(num_examples: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return (num_examples + batch_size - 1) // batch_size


def _should_log_batch_progress(
    batch_index: int,
    total_batches: int,
    interval_batches: int,
) -> bool:
    if batch_index <= 1 or batch_index >= total_batches:
        return True
    return interval_batches > 0 and batch_index % interval_batches == 0


def _pretokenize_examples(runtime_config, tokenizer, examples: list[BenchmarkExample]) -> dict[str, Any]:
    prompts = [example.prompt for example in examples]
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=runtime_config.model_max_length,
    )
    return {
        "prompts": prompts,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def _slice_prepared_batch(
    prepared_inputs: dict[str, Any],
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": prepared_inputs["input_ids"][start:stop].to(device),
        "attention_mask": prepared_inputs["attention_mask"][start:stop].to(device),
    }


def _generate_baseline(
    model,
    tokenizer,
    prepared_batch: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[str]:
    generation_kwargs = {
        "input_ids": prepared_batch["input_ids"],
        "attention_mask": prepared_batch["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not greedy:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_k"] = top_k
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(**generation_kwargs)

    predictions: list[str] = []
    for sequence in generated:
        new_tokens = sequence[prepared_batch["input_ids"].size(1) :]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return predictions


def _generate_predictions(
    *,
    loaded_model,
    prepared_batch: dict[str, torch.Tensor],
    config: EvaluationRuntimeConfig,
) -> list[str]:
    if loaded_model.inference_strategy == "latent_cot":
        latent_model = loaded_model.model
        input_ids = prepared_batch["input_ids"]
        attention_mask = prepared_batch["attention_mask"]
        device = input_ids.device

        if loaded_model.remove_eos:
            bot_tensor = torch.tensor([loaded_model.bot_id], dtype=torch.long, device=device).expand(input_ids.size(0), 1)
        else:
            bot_tensor = torch.tensor(
                [loaded_model.tokenizer.eos_token_id, loaded_model.bot_id],
                dtype=torch.long,
                device=device,
            ).expand(
                input_ids.size(0), 2
            )
        input_ids = torch.cat([input_ids, bot_tensor], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)

        with torch.no_grad():
            return latent_model.generate_from_latent(
                tokenizer=loaded_model.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inf_latent_iterations=loaded_model.runtime_config.num_latent,
                max_new_tokens=config.max_new_tokens,
                greedy=config.greedy,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )

    return _generate_baseline(
        model=loaded_model.generation_model,
        tokenizer=loaded_model.tokenizer,
        prepared_batch=prepared_batch,
        max_new_tokens=config.max_new_tokens,
        greedy=config.greedy,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )


def run_evaluation_from_config(config_path: str) -> None:
    payload = load_yaml_config(config_path)
    config = EvaluationConfig.from_dict(payload)
    device = _resolve_device(config.runtime.device)
    output_dir = ensure_dir(config.runtime.output_dir)
    ensure_dir(config.runtime.snapshot_dir)
    _log_event(
        "starting evaluation "
        f"config={config_path} device={device} models={len(config.models)} "
        f"benchmarks={len(config.benchmarks)} batch_size={config.runtime.batch_size}"
    )

    summary_rows: list[dict[str, Any]] = []

    run_metadata_path = Path(output_dir) / "resolved_config.json"
    run_metadata_path.write_text(json.dumps(dump_yamlable(payload), indent=2), encoding="utf-8")
    predictions_path = Path(output_dir) / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    for model_spec in config.models:
        _log_event(f"loading model name={model_spec.name} source={model_spec.checkpoint_source}")
        loaded_model = load_evaluation_model(model_spec, device=device)
        _log_event(
            f"loaded model name={model_spec.name} "
            f"base_model={model_spec.model.base_model_name_or_path} "
            f"inference_strategy={model_spec.inference_strategy} "
            f"num_latent={model_spec.runtime.num_latent}"
        )

        for benchmark_name in config.benchmarks:
            benchmark_started = time.perf_counter()
            loaded = load_benchmark(
                get_benchmark_spec(benchmark_name),
                cache_dir=config.runtime.cache_dir,
                snapshot_dir=config.runtime.snapshot_dir,
                max_examples=config.runtime.max_examples_per_benchmark,
            )
            _log_event(
                f"loaded benchmark model={model_spec.name} benchmark={benchmark_name} "
                f"examples={len(loaded.examples)}"
            )

            mode_started = time.perf_counter()
            prepared_inputs = _pretokenize_examples(loaded_model.runtime_config, loaded_model.tokenizer, loaded.examples)
            scored_rows: list[dict[str, Any]] = []
            prediction_rows_for_mode: list[dict[str, Any]] = []
            latencies: list[float] = []
            char_counts: list[int] = []
            total_batches = _num_batches(len(loaded.examples), config.runtime.batch_size)
            _log_event(
                f"starting evaluation model={model_spec.name} benchmark={benchmark_name} "
                f"inference_strategy={model_spec.inference_strategy} "
                f"examples={len(loaded.examples)} batches={total_batches}"
            )

            for batch_index, start in enumerate(range(0, len(loaded.examples), config.runtime.batch_size), start=1):
                stop = min(start + config.runtime.batch_size, len(loaded.examples))
                batch_examples = loaded.examples[start:stop]
                prepared_batch = _slice_prepared_batch(prepared_inputs, start, stop, device=device)
                batch_started = time.perf_counter()
                predictions = _generate_predictions(
                    loaded_model=loaded_model,
                    prepared_batch=prepared_batch,
                    config=config.runtime,
                )
                elapsed = time.perf_counter() - batch_started
                per_example_latency = elapsed / max(len(batch_examples), 1)

                for example, prediction in zip(batch_examples, predictions):
                    score = score_prediction(example, prediction)
                    latencies.append(per_example_latency)
                    char_counts.append(len(prediction))
                    row = {
                        "model": model_spec.name,
                        "benchmark": benchmark_name,
                        "inference_strategy": model_spec.inference_strategy,
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
                    prediction_rows_for_mode.append(row | {"prompt": example.prompt})

                if _should_log_batch_progress(
                    batch_index,
                    total_batches,
                    config.runtime.progress_log_interval_batches,
                ):
                    elapsed_mode_s = time.perf_counter() - mode_started
                    _log_event(
                        f"progress model={model_spec.name} benchmark={benchmark_name} "
                        f"inference_strategy={model_spec.inference_strategy} "
                        f"batch={batch_index}/{total_batches} "
                        f"examples={stop}/{len(loaded.examples)} "
                        f"batch_elapsed_s={elapsed:.2f} "
                        f"mode_elapsed_s={elapsed_mode_s:.1f}"
                    )

            accuracy = sum(1 for row in scored_rows if row["correct"]) / max(len(scored_rows), 1)
            summary_row = {
                "model": model_spec.name,
                "benchmark": benchmark_name,
                "inference_strategy": model_spec.inference_strategy,
                "accuracy": accuracy,
                "num_examples": len(scored_rows),
                "avg_latency_s": sum(latencies) / max(len(latencies), 1),
                "avg_prediction_chars": sum(char_counts) / max(len(char_counts), 1),
            }
            summary_rows.append(summary_row)
            _log_event(
                f"finished evaluation model={model_spec.name} benchmark={benchmark_name} "
                f"inference_strategy={model_spec.inference_strategy} "
                f"accuracy={summary_row['accuracy']:.4f} "
                f"avg_latency_s={summary_row['avg_latency_s']:.4f} "
                f"avg_chars={summary_row['avg_prediction_chars']:.1f} "
                f"elapsed_s={time.perf_counter() - mode_started:.1f}"
            )

            append_jsonl(predictions_path, prediction_rows_for_mode)
            write_csv(output_dir / "summary.csv", summary_rows)
            write_markdown_summary(output_dir / "summary.md", summary_rows)
            _log_event(
                f"finished benchmark model={model_spec.name} benchmark={benchmark_name} "
                f"elapsed_s={time.perf_counter() - benchmark_started:.1f}"
            )
    _log_event(
        f"finished evaluation output_dir={output_dir} "
        f"predictions={predictions_path} summary_csv={output_dir / 'summary.csv'}"
    )
