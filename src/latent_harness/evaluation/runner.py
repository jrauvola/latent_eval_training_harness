from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_harness.core.io import dump_yamlable, ensure_dir, load_yaml_config
from latent_harness.evaluation.benchmarks import BenchmarkExample, get_benchmark_spec, load_benchmark
from latent_harness.evaluation.config import EvaluationConfig, EvaluationRuntimeConfig
from latent_harness.evaluation.latent_tap import (
    extract_kv_at_latent_positions,
    generate_from_latent_with_taps,
    project_hidden_to_topk,
)
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
    use_chat = getattr(runtime_config, "use_chat_template", False)
    chat_kw = getattr(runtime_config, "chat_template_kwargs", None) or {}
    prompts: list[str] = []
    for example in examples:
        prompt = example.prompt
        if use_chat and getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_kw,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        prompts.append(prompt)
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
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
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


def _generate_predictions_with_taps(
    *,
    loaded_model,
    prepared_batch: dict[str, torch.Tensor],
    config: EvaluationRuntimeConfig,
    num_latent: int,
    skip_latent_injection: bool,
    capture_hidden: bool,
):
    """Latent-CoT generation that also returns hidden-state/KV taps.

    Returns ``(predictions, taps_or_none)``. ``taps_or_none`` is None for the
    baseline (non-latent) path.
    """
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
            taps = generate_from_latent_with_taps(
                latent_model,
                tokenizer=loaded_model.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inf_latent_iterations=num_latent,
                max_new_tokens=config.max_new_tokens,
                greedy=config.greedy,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                skip_latent_injection=skip_latent_injection,
                capture_latent_hidden=capture_hidden,
            )
        return taps.predictions, taps

    predictions = _generate_baseline(
        model=loaded_model.generation_model,
        tokenizer=loaded_model.tokenizer,
        prepared_batch=prepared_batch,
        max_new_tokens=config.max_new_tokens,
        greedy=config.greedy,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    return predictions, None


def _variant_label(model_spec) -> str:
    """Slug used for per-variant artifact paths."""
    return model_spec.name


def _dump_latent_traces(
    *,
    loaded_model,
    taps,
    batch_global_indices: list[int],
    traces_root: Path,
    variant: str,
    benchmark: str,
    num_latent: int,
    top_k: int,
) -> None:
    """Project each captured latent hidden state through lm_head and dump top-K."""

    if not taps.latent_traces:
        return

    dest_dir = traces_root / variant / benchmark / f"numlatent_{num_latent}"
    ensure_dir(dest_dir)

    # Project per-step for all examples in batch simultaneously.
    per_step_topk: list[list[list[dict[str, Any]]]] = []
    for step in taps.latent_traces:
        topk_batch = project_hidden_to_topk(
            loaded_model.model,
            step.hidden_state,
            loaded_model.tokenizer,
            top_k=top_k,
        )
        per_step_topk.append(topk_batch)

    # Fan out per example into the jsonl file.
    for batch_offset, example_idx in enumerate(batch_global_indices):
        dest = dest_dir / f"trace_{example_idx}.jsonl"
        with dest.open("a", encoding="utf-8") as handle:
            for latent_step_index, topk_batch in enumerate(per_step_topk):
                row_topk = topk_batch[batch_offset]
                row = {
                    "example_idx": example_idx,
                    "latent_step": latent_step_index,
                    "top10_token_ids": [entry["token_id"] for entry in row_topk],
                    "top10_probs": [entry["prob"] for entry in row_topk],
                    "top10_token_strs": [entry["token_str"] for entry in row_topk],
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _dump_kv_cache(
    *,
    taps,
    batch_global_indices: list[int],
    kv_root: Path,
    variant: str,
    benchmark: str,
    num_latent: int,
    max_examples: int,
    final_layer_only: bool,
    already_dumped: set[int],
) -> None:
    """Dump per-example KV arrays at latent positions up to ``max_examples``."""

    if num_latent == 0:
        return

    dest_dir = kv_root / variant / benchmark / f"numlatent_{num_latent}"
    ensure_dir(dest_dir)

    kv_array = extract_kv_at_latent_positions(
        taps.final_cache,
        encoder_prefix_length=taps.encoder_prefix_length,
        num_latent=taps.num_latent_iterated,
        final_layer_only=final_layer_only,
    )
    for batch_offset, example_idx in enumerate(batch_global_indices):
        if len(already_dumped) >= max_examples:
            return
        if example_idx in already_dumped:
            continue
        dest = dest_dir / f"kv_example_{example_idx}.npy"
        np.save(str(dest), kv_array[batch_offset])
        already_dumped.add(example_idx)


def _write_running_summary(
    path: Path,
    *,
    variant: str,
    benchmark: str,
    num_latent: int,
    completed_count: int,
    correct_count: int,
    failed_count: int,
    total_examples: int,
    started_at: float,
) -> None:
    summary = {
        "variant": variant,
        "benchmark": benchmark,
        "num_latent": num_latent,
        "completed_count": completed_count,
        "correct_count": correct_count,
        "failed_count": failed_count,
        "total_examples": total_examples,
        "accuracy_so_far": (correct_count / completed_count) if completed_count else 0.0,
        "wall_time_s": time.perf_counter() - started_at,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_single_evaluation(
    *,
    loaded_model,
    model_spec,
    loaded_benchmark,
    runtime: EvaluationRuntimeConfig,
    benchmark_name: str,
    num_latent: int,
    skip_latent_injection: bool,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run a single (variant, benchmark, num_latent) eval with full instrumentation."""

    variant = _variant_label(model_spec)

    # Per-(variant, benchmark, num_latent) directory for predictions + summary.
    eval_leaf = output_dir / variant / f"{benchmark_name}_numlatent_{num_latent}"
    ensure_dir(eval_leaf)
    predictions_path = eval_leaf / "predictions.jsonl"
    summary_path = eval_leaf / "running_summary.json"
    # Reset predictions for this leaf so reruns don't accrete duplicate rows.
    if predictions_path.exists():
        predictions_path.unlink()

    mode_started = time.perf_counter()
    examples = loaded_benchmark.examples
    prepared_inputs = _pretokenize_examples(loaded_model.runtime_config, loaded_model.tokenizer, examples)

    total_batches = _num_batches(len(examples), runtime.batch_size)
    latencies: list[float] = []
    char_counts: list[int] = []
    completed_count = 0
    correct_count = 0
    failed_count = 0
    persistence_every = max(1, min(runtime.persistence_every_examples, runtime.batch_size))
    rows_since_flush: list[dict[str, Any]] = []
    kv_already_dumped: set[int] = set()

    traces_root = Path(runtime.latent_traces_dir).expanduser().resolve() if runtime.dump_latent_traces else None
    kv_root = Path(runtime.kv_dump_dir).expanduser().resolve() if runtime.dump_kv_cache else None
    if traces_root is not None:
        ensure_dir(traces_root)
    if kv_root is not None:
        ensure_dir(kv_root)

    _log_event(
        f"starting (variant={variant}, benchmark={benchmark_name}, num_latent={num_latent}) "
        f"examples={len(examples)} batches={total_batches} "
        f"persistence_every={persistence_every} "
        f"skip_latent_injection={skip_latent_injection}"
    )

    for batch_index, start in enumerate(range(0, len(examples), runtime.batch_size), start=1):
        stop = min(start + runtime.batch_size, len(examples))
        batch_examples = examples[start:stop]
        batch_global_indices = list(range(start, stop))
        prepared_batch = _slice_prepared_batch(prepared_inputs, start, stop, device=device)
        batch_started = time.perf_counter()
        try:
            want_kv_dump = bool(kv_root is not None and len(kv_already_dumped) < runtime.kv_dump_max_examples)
            want_trace_dump = traces_root is not None
            capture_hidden = want_trace_dump
            predictions, taps = _generate_predictions_with_taps(
                loaded_model=loaded_model,
                prepared_batch=prepared_batch,
                config=runtime,
                num_latent=num_latent,
                skip_latent_injection=skip_latent_injection,
                capture_hidden=capture_hidden,
            )
        except Exception as exc:  # noqa: BLE001 — broad on purpose: persist partial, re-raise.
            failed_count += len(batch_examples)
            _log_event(
                f"ERROR batch variant={variant} benchmark={benchmark_name} num_latent={num_latent} "
                f"batch={batch_index}/{total_batches} error={type(exc).__name__}: {exc}"
            )
            if rows_since_flush:
                append_jsonl(predictions_path, rows_since_flush)
                rows_since_flush = []
            _write_running_summary(
                summary_path,
                variant=variant,
                benchmark=benchmark_name,
                num_latent=num_latent,
                completed_count=completed_count,
                correct_count=correct_count,
                failed_count=failed_count,
                total_examples=len(examples),
                started_at=mode_started,
            )
            raise

        elapsed = time.perf_counter() - batch_started
        per_example_latency = elapsed / max(len(batch_examples), 1)

        for example, prediction in zip(batch_examples, predictions):
            score = score_prediction(example, prediction)
            latencies.append(per_example_latency)
            char_counts.append(len(prediction))
            row = {
                "model": variant,
                "benchmark": benchmark_name,
                "inference_strategy": loaded_model.inference_strategy,
                "num_latent": num_latent,
                "example_id": example.example_id,
                "task_type": example.task_type,
                "target": example.target,
                "prompt": example.prompt,
                "prediction_text": prediction,
                "parsed_prediction": score.parsed_prediction.value,
                "prediction_valid": score.parsed_prediction.is_valid,
                "correct": score.correct,
                "latency_s": per_example_latency,
            }
            rows_since_flush.append(row)
            completed_count += 1
            if score.correct:
                correct_count += 1

        # Dump interp artifacts for this batch if enabled.
        if taps is not None:
            if traces_root is not None:
                _dump_latent_traces(
                    loaded_model=loaded_model,
                    taps=taps,
                    batch_global_indices=batch_global_indices,
                    traces_root=traces_root,
                    variant=variant,
                    benchmark=benchmark_name,
                    num_latent=num_latent,
                    top_k=runtime.latent_trace_topk,
                )
            if kv_root is not None:
                _dump_kv_cache(
                    taps=taps,
                    batch_global_indices=batch_global_indices,
                    kv_root=kv_root,
                    variant=variant,
                    benchmark=benchmark_name,
                    num_latent=num_latent,
                    max_examples=runtime.kv_dump_max_examples,
                    final_layer_only=runtime.kv_dump_final_layer_only,
                    already_dumped=kv_already_dumped,
                )

        # Batch-level persistence: flush every `persistence_every` examples.
        if len(rows_since_flush) >= persistence_every or completed_count == len(examples):
            append_jsonl(predictions_path, rows_since_flush)
            rows_since_flush = []
            _write_running_summary(
                summary_path,
                variant=variant,
                benchmark=benchmark_name,
                num_latent=num_latent,
                completed_count=completed_count,
                correct_count=correct_count,
                failed_count=failed_count,
                total_examples=len(examples),
                started_at=mode_started,
            )

        if _should_log_batch_progress(batch_index, total_batches, runtime.progress_log_interval_batches):
            _log_event(
                f"progress variant={variant} benchmark={benchmark_name} num_latent={num_latent} "
                f"batch={batch_index}/{total_batches} examples={completed_count}/{len(examples)} "
                f"accuracy_so_far={(correct_count / max(completed_count, 1)):.4f} "
                f"batch_elapsed_s={elapsed:.2f}"
            )

    # Final flush (in case anything is left).
    if rows_since_flush:
        append_jsonl(predictions_path, rows_since_flush)
        rows_since_flush = []

    accuracy = correct_count / max(completed_count, 1)
    summary_row = {
        "model": variant,
        "benchmark": benchmark_name,
        "inference_strategy": loaded_model.inference_strategy,
        "num_latent": num_latent,
        "accuracy": accuracy,
        "num_examples": completed_count,
        "avg_latency_s": sum(latencies) / max(len(latencies), 1),
        "avg_prediction_chars": sum(char_counts) / max(len(char_counts), 1),
    }
    _write_running_summary(
        summary_path,
        variant=variant,
        benchmark=benchmark_name,
        num_latent=num_latent,
        completed_count=completed_count,
        correct_count=correct_count,
        failed_count=failed_count,
        total_examples=len(examples),
        started_at=mode_started,
    )
    _log_event(
        f"finished variant={variant} benchmark={benchmark_name} num_latent={num_latent} "
        f"accuracy={accuracy:.4f} examples={completed_count} "
        f"elapsed_s={time.perf_counter() - mode_started:.1f}"
    )
    return summary_row


def run_evaluation_from_config(config_path: str) -> None:
    payload = load_yaml_config(config_path)
    config = EvaluationConfig.from_dict(payload)
    device = _resolve_device(config.runtime.device)
    output_dir = ensure_dir(config.runtime.output_dir)
    ensure_dir(config.runtime.snapshot_dir)
    _log_event(
        "starting evaluation "
        f"config={config_path} device={device} models={len(config.models)} "
        f"benchmarks={len(config.benchmarks)} batch_size={config.runtime.batch_size} "
        f"num_latent_sweep={config.runtime.num_latent_sweep}"
    )

    summary_rows: list[dict[str, Any]] = []
    run_metadata_path = Path(output_dir) / "resolved_config.json"
    run_metadata_path.write_text(json.dumps(dump_yamlable(payload), indent=2), encoding="utf-8")

    for model_spec in config.models:
        _log_event(f"loading model name={model_spec.name} source={model_spec.checkpoint_source}")
        loaded_model = load_evaluation_model(model_spec, device=device)
        _log_event(
            f"loaded model name={model_spec.name} "
            f"base_model={model_spec.model.base_model_name_or_path} "
            f"inference_strategy={model_spec.inference_strategy} "
            f"spec_num_latent={model_spec.runtime.num_latent}"
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

            if config.runtime.num_latent_sweep:
                nl_values = list(config.runtime.num_latent_sweep)
            else:
                nl_values = [model_spec.runtime.num_latent]

            for num_latent in nl_values:
                # CRITICAL: mutate the runtime config in-place so the runtime honors the
                # current sweep value. This is safe because the runtime consults
                # ``runtime_config.num_latent`` only at generate time, not at load.
                loaded_model.runtime_config.num_latent = num_latent
                skip_latent_injection = (num_latent == 0) and config.runtime.skip_latent_injection_at_zero

                summary_row = _run_single_evaluation(
                    loaded_model=loaded_model,
                    model_spec=model_spec,
                    loaded_benchmark=loaded,
                    runtime=config.runtime,
                    benchmark_name=benchmark_name,
                    num_latent=num_latent,
                    skip_latent_injection=skip_latent_injection,
                    output_dir=Path(output_dir),
                    device=device,
                )
                summary_rows.append(summary_row)
                # Global summary keeps getting refreshed so it's resumable on crash.
                write_csv(Path(output_dir) / "summary.csv", summary_rows)
                write_markdown_summary(Path(output_dir) / "summary.md", summary_rows)

            _log_event(
                f"finished benchmark model={model_spec.name} benchmark={benchmark_name} "
                f"elapsed_s={time.perf_counter() - benchmark_started:.1f}"
            )

    _log_event(
        f"finished evaluation output_dir={output_dir} "
        f"summary_csv={output_dir / 'summary.csv'}"
    )
