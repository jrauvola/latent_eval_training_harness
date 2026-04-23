"""Standalone runner for F4 / F5 / F6 inert-latent hypothesis tests.

Loads V2 bf16 once and runs one or more of:

* F4 — ablate latent-position KV from the cache before the answer-generation loop.
* F5 — cross-example latent KV swap (inject example A's latent KV into example B).
* F6 — Gaussian noise on latent-position KV (sigma sweep).

Baseline (no intervention) is also run as a sanity check so we can verify the
numbers match Phase 1.

All per-batch predictions are flushed to disk immediately (batch-level eval
persistence mandate).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from latent_harness.core.io import load_yaml_config
from latent_harness.evaluation.benchmarks import get_benchmark_spec, load_benchmark
from latent_harness.evaluation.config import EvaluationConfig
from latent_harness.evaluation.latent_tap import (
    capture_latent_kv_slice,
    generate_from_latent_with_taps,
)
from latent_harness.evaluation.metrics import score_prediction
from latent_harness.evaluation.models import load_evaluation_model
from latent_harness.evaluation.runner import _log_event, _pretokenize_examples, _slice_prepared_batch


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _preencode_batch(loaded_model, prepared_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Append BOT (+ optionally EOS) to match the standard latent-cot entry."""
    input_ids = prepared_batch["input_ids"]
    attention_mask = prepared_batch["attention_mask"]
    device = input_ids.device
    if loaded_model.remove_eos:
        bot_tensor = torch.tensor([loaded_model.bot_id], dtype=torch.long, device=device).expand(
            input_ids.size(0), 1
        )
    else:
        bot_tensor = torch.tensor(
            [loaded_model.tokenizer.eos_token_id, loaded_model.bot_id],
            dtype=torch.long,
            device=device,
        ).expand(input_ids.size(0), 2)
    input_ids = torch.cat([input_ids, bot_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(bot_tensor)], dim=1)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _score_row(row_base: dict, example, prediction: str) -> dict:
    score = score_prediction(example, prediction)
    return {
        **row_base,
        "example_id": example.example_id,
        "task_type": example.task_type,
        "target": example.target,
        "prompt": example.prompt,
        "prediction_text": prediction,
        "parsed_prediction": score.parsed_prediction.value,
        "prediction_valid": score.parsed_prediction.is_valid,
        "correct": score.correct,
    }


def _run_intervention(
    *,
    loaded_model,
    prepared_inputs,
    benchmark_examples,
    device,
    batch_size,
    num_latent,
    max_new_tokens,
    greedy,
    temperature,
    top_k_sampling,
    top_p,
    out_path: Path,
    summary_path: Path,
    variant_label: str,
    benchmark_name: str,
    intervention_kwargs: dict[str, Any],
    intervention_label: str,
    row_meta: dict,
    examples_cap: int | None = None,
    persistence_every: int = 32,
) -> dict:
    """Loop over batches, call generate_from_latent_with_taps with intervention, persist."""
    total = len(benchmark_examples) if examples_cap is None else min(examples_cap, len(benchmark_examples))
    if out_path.exists():
        out_path.unlink()
    rows: list[dict] = []
    correct = 0
    completed = 0
    started = time.perf_counter()
    _log_event(
        f"starting intervention='{intervention_label}' variant={variant_label} "
        f"benchmark={benchmark_name} num_latent={num_latent} n={total}"
    )

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        batch_examples = benchmark_examples[start:stop]
        batch_prepared = _slice_prepared_batch(prepared_inputs, start, stop, device=device)
        batch = _preencode_batch(loaded_model, batch_prepared)

        with torch.no_grad():
            taps = generate_from_latent_with_taps(
                loaded_model.model,
                tokenizer=loaded_model.tokenizer,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                inf_latent_iterations=num_latent,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p,
                skip_latent_injection=(num_latent == 0),
                capture_latent_hidden=False,
                **intervention_kwargs,
            )

        for example, prediction in zip(batch_examples, taps.predictions):
            row = _score_row(
                {
                    **row_meta,
                    "intervention": intervention_label,
                    "num_latent": num_latent,
                },
                example,
                prediction,
            )
            rows.append(row)
            completed += 1
            if row["correct"]:
                correct += 1

        if len(rows) >= persistence_every or completed == total:
            _append_jsonl(out_path, rows)
            rows = []
            elapsed = time.perf_counter() - started
            _write_summary(
                summary_path,
                {
                    "variant": variant_label,
                    "benchmark": benchmark_name,
                    "intervention": intervention_label,
                    "num_latent": num_latent,
                    "completed": completed,
                    "correct": correct,
                    "total": total,
                    "accuracy_so_far": correct / max(completed, 1),
                    "wall_time_s": elapsed,
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            _log_event(
                f"progress intervention='{intervention_label}' benchmark={benchmark_name} "
                f"done={completed}/{total} acc={correct / max(completed, 1):.4f}"
            )
    # Final flush if anything left
    if rows:
        _append_jsonl(out_path, rows)
        rows = []

    final = {
        "variant": variant_label,
        "benchmark": benchmark_name,
        "intervention": intervention_label,
        "num_latent": num_latent,
        "completed": completed,
        "correct": correct,
        "total": total,
        "accuracy": correct / max(completed, 1),
        "wall_time_s": time.perf_counter() - started,
    }
    _write_summary(summary_path, final)
    _log_event(f"finished intervention='{intervention_label}' accuracy={final['accuracy']:.4f}")
    return final


def _run_f5(
    *,
    loaded_model,
    prepared_inputs,
    benchmark_examples,
    device,
    num_latent,
    max_new_tokens,
    greedy,
    temperature,
    top_k_sampling,
    top_p,
    out_path: Path,
    summary_path: Path,
    variant_label: str,
    benchmark_name: str,
    n_pairs: int,
    seed: int,
    row_meta: dict,
    persistence_every: int = 10,
) -> dict:
    """F5: capture latent KV from example A, inject into example B's forward.

    For each of n_pairs pairs, we run TWO generations:
      1. Baseline for B (no swap) — for comparison
      2. Swapped for B (A's latent KV injected)
    so we can measure the Δ directly. Assumes A != B. Uses greedy for reproducibility.
    """
    random.seed(seed)
    total_examples = len(benchmark_examples)
    idx_pool = list(range(total_examples))
    random.shuffle(idx_pool)
    pairs = [(idx_pool[i], idx_pool[i + 1]) for i in range(0, 2 * n_pairs, 2)]
    pairs = pairs[:n_pairs]

    if out_path.exists():
        out_path.unlink()
    rows: list[dict] = []
    started = time.perf_counter()
    n_b_correct_baseline = 0
    n_b_correct_swapped = 0
    n_b_changed = 0

    _log_event(f"starting F5 swap benchmark={benchmark_name} pairs={len(pairs)}")

    for pair_idx, (idx_a, idx_b) in enumerate(pairs):
        ex_a = benchmark_examples[idx_a]
        ex_b = benchmark_examples[idx_b]

        # Build single-example batches for A and B
        def _single_batch(idx):
            sliced = _slice_prepared_batch(prepared_inputs, idx, idx + 1, device=device)
            return _preencode_batch(loaded_model, sliced)

        batch_a = _single_batch(idx_a)
        batch_b = _single_batch(idx_b)

        # --- Step 1: forward A, capture latent KV slice ---
        with torch.no_grad():
            taps_a = generate_from_latent_with_taps(
                loaded_model.model,
                tokenizer=loaded_model.tokenizer,
                input_ids=batch_a["input_ids"],
                attention_mask=batch_a["attention_mask"],
                inf_latent_iterations=num_latent,
                max_new_tokens=1,  # we only need to reach the end of the latent rollout
                greedy=True,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p,
                skip_latent_injection=False,
                capture_latent_hidden=False,
            )
        # Capture A's latent KV slice (from A's final cache, which has latents appended)
        slices_a = capture_latent_kv_slice(
            taps_a.final_cache,
            encoder_prefix_length=taps_a.encoder_prefix_length,
            num_latent=num_latent,
        )

        # --- Step 2: forward B baseline (no swap) ---
        with torch.no_grad():
            taps_b_base = generate_from_latent_with_taps(
                loaded_model.model,
                tokenizer=loaded_model.tokenizer,
                input_ids=batch_b["input_ids"],
                attention_mask=batch_b["attention_mask"],
                inf_latent_iterations=num_latent,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p,
                skip_latent_injection=False,
                capture_latent_hidden=False,
            )
        pred_b_base = taps_b_base.predictions[0]
        score_b_base = score_prediction(ex_b, pred_b_base)

        # --- Step 3: forward B with A's slice injected ---
        # Shape compatibility check: slices_a tensors must match B's cache shape.
        # Since both are single-example (batch=1) and use the same model, they should align.
        with torch.no_grad():
            taps_b_swap = generate_from_latent_with_taps(
                loaded_model.model,
                tokenizer=loaded_model.tokenizer,
                input_ids=batch_b["input_ids"],
                attention_mask=batch_b["attention_mask"],
                inf_latent_iterations=num_latent,
                max_new_tokens=max_new_tokens,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k_sampling,
                top_p=top_p,
                skip_latent_injection=False,
                capture_latent_hidden=False,
                inject_latent_kv_slices=slices_a,
            )
        pred_b_swap = taps_b_swap.predictions[0]
        score_b_swap = score_prediction(ex_b, pred_b_swap)

        if score_b_base.correct:
            n_b_correct_baseline += 1
        if score_b_swap.correct:
            n_b_correct_swapped += 1
        if score_b_base.parsed_prediction.value != score_b_swap.parsed_prediction.value:
            n_b_changed += 1

        rows.append(
            {
                **row_meta,
                "intervention": "f5_swap",
                "num_latent": num_latent,
                "pair_idx": pair_idx,
                "ex_a_id": ex_a.example_id,
                "ex_b_id": ex_b.example_id,
                "target_b": ex_b.target,
                "pred_b_baseline": pred_b_base,
                "pred_b_swapped": pred_b_swap,
                "parsed_b_baseline": score_b_base.parsed_prediction.value,
                "parsed_b_swapped": score_b_swap.parsed_prediction.value,
                "correct_b_baseline": score_b_base.correct,
                "correct_b_swapped": score_b_swap.correct,
                "b_changed": score_b_base.parsed_prediction.value != score_b_swap.parsed_prediction.value,
            }
        )

        if len(rows) >= persistence_every or pair_idx == len(pairs) - 1:
            _append_jsonl(out_path, rows)
            rows = []
            completed = pair_idx + 1
            _write_summary(
                summary_path,
                {
                    "variant": variant_label,
                    "benchmark": benchmark_name,
                    "intervention": "f5_swap",
                    "num_latent": num_latent,
                    "pairs_done": completed,
                    "pairs_total": len(pairs),
                    "b_correct_baseline": n_b_correct_baseline,
                    "b_correct_swapped": n_b_correct_swapped,
                    "b_changed": n_b_changed,
                    "acc_b_baseline": n_b_correct_baseline / max(completed, 1),
                    "acc_b_swapped": n_b_correct_swapped / max(completed, 1),
                    "frac_b_changed": n_b_changed / max(completed, 1),
                    "wall_time_s": time.perf_counter() - started,
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            _log_event(
                f"F5 progress pair {pair_idx + 1}/{len(pairs)} "
                f"acc_base={n_b_correct_baseline/max(pair_idx+1,1):.3f} "
                f"acc_swap={n_b_correct_swapped/max(pair_idx+1,1):.3f} "
                f"frac_changed={n_b_changed/max(pair_idx+1,1):.3f}"
            )
    # Final flush
    if rows:
        _append_jsonl(out_path, rows)

    final = {
        "variant": variant_label,
        "benchmark": benchmark_name,
        "intervention": "f5_swap",
        "num_latent": num_latent,
        "pairs_total": len(pairs),
        "b_correct_baseline": n_b_correct_baseline,
        "b_correct_swapped": n_b_correct_swapped,
        "b_changed": n_b_changed,
        "acc_b_baseline": n_b_correct_baseline / max(len(pairs), 1),
        "acc_b_swapped": n_b_correct_swapped / max(len(pairs), 1),
        "frac_b_changed": n_b_changed / max(len(pairs), 1),
        "wall_time_s": time.perf_counter() - started,
    }
    _write_summary(summary_path, final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tests", type=str, default="baseline,f4,f6,f5",
                        help="Comma-separated list of: baseline,f4,f6,f5")
    parser.add_argument("--f6_sigmas", type=str, default="0.1,0.5,1.0")
    parser.add_argument("--f5_pairs", type=int, default=100)
    parser.add_argument("--f6_examples", type=int, default=200)
    parser.add_argument("--output_root", type=str, default="artifacts/eval/qwen3_4b_codi_f_tests")
    args = parser.parse_args()

    tests = {t.strip() for t in args.tests.split(",") if t.strip()}
    payload = load_yaml_config(args.config)
    config = EvaluationConfig.from_dict(payload)
    device = _resolve_device(config.runtime.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _log_event(f"config={args.config} device={device} tests={sorted(tests)}")

    # We expect exactly one model spec (V2 bf16). Load it once.
    assert len(config.models) == 1, "F-tests config should have exactly one model"
    model_spec = config.models[0]
    loaded_model = load_evaluation_model(model_spec, device=device)
    _log_event(f"loaded model={model_spec.name}")

    # Use num_latent=8 consistently (peak for V2 bf16)
    num_latent = 8
    # Override runtime's num_latent for symmetry
    loaded_model.runtime_config.num_latent = num_latent

    aggregate: dict[str, Any] = {}

    for benchmark_name in config.benchmarks:
        loaded = load_benchmark(
            get_benchmark_spec(benchmark_name),
            cache_dir=config.runtime.cache_dir,
            snapshot_dir=config.runtime.snapshot_dir,
            max_examples=config.runtime.max_examples_per_benchmark,
        )
        examples = loaded.examples
        _log_event(f"loaded benchmark={benchmark_name} examples={len(examples)}")
        prepared_inputs = _pretokenize_examples(
            loaded_model.runtime_config,
            loaded_model.tokenizer,
            examples,
            prompt_suffix="",
        )
        row_meta = {
            "model": model_spec.name,
            "benchmark": benchmark_name,
            "inference_strategy": "latent_cot",
        }

        # ---- Baseline (sanity: should match Phase 1 V2 bf16 numbers at n=8) ----
        if "baseline" in tests:
            out_path = output_root / model_spec.name / f"{benchmark_name}_baseline" / "predictions.jsonl"
            summary_path = output_root / model_spec.name / f"{benchmark_name}_baseline" / "running_summary.json"
            res = _run_intervention(
                loaded_model=loaded_model,
                prepared_inputs=prepared_inputs,
                benchmark_examples=examples,
                device=device,
                batch_size=config.runtime.batch_size,
                num_latent=num_latent,
                max_new_tokens=config.runtime.max_new_tokens,
                greedy=config.runtime.greedy,
                temperature=config.runtime.temperature,
                top_k_sampling=config.runtime.top_k,
                top_p=config.runtime.top_p,
                out_path=out_path,
                summary_path=summary_path,
                variant_label=model_spec.name,
                benchmark_name=benchmark_name,
                intervention_kwargs={},
                intervention_label="baseline",
                row_meta=row_meta,
                persistence_every=config.runtime.persistence_every_examples,
            )
            aggregate.setdefault(benchmark_name, {})["baseline"] = res

        # ---- F4: ablate latent KV before answer ----
        if "f4" in tests:
            out_path = output_root / model_spec.name / f"{benchmark_name}_f4_ablate" / "predictions.jsonl"
            summary_path = output_root / model_spec.name / f"{benchmark_name}_f4_ablate" / "running_summary.json"
            res = _run_intervention(
                loaded_model=loaded_model,
                prepared_inputs=prepared_inputs,
                benchmark_examples=examples,
                device=device,
                batch_size=config.runtime.batch_size,
                num_latent=num_latent,
                max_new_tokens=config.runtime.max_new_tokens,
                greedy=config.runtime.greedy,
                temperature=config.runtime.temperature,
                top_k_sampling=config.runtime.top_k,
                top_p=config.runtime.top_p,
                out_path=out_path,
                summary_path=summary_path,
                variant_label=model_spec.name,
                benchmark_name=benchmark_name,
                intervention_kwargs={"ablate_latent_kv_before_answer": True},
                intervention_label="f4_ablate",
                row_meta=row_meta,
                persistence_every=config.runtime.persistence_every_examples,
            )
            aggregate.setdefault(benchmark_name, {})["f4_ablate"] = res

        # ---- F6: Gaussian noise sweep on latent KV ----
        if "f6" in tests:
            sigmas = [float(s) for s in args.f6_sigmas.split(",") if s.strip()]
            cap = args.f6_examples
            for sigma in sigmas:
                label = f"f6_sigma_{sigma}"
                out_path = output_root / model_spec.name / f"{benchmark_name}_{label}" / "predictions.jsonl"
                summary_path = output_root / model_spec.name / f"{benchmark_name}_{label}" / "running_summary.json"
                res = _run_intervention(
                    loaded_model=loaded_model,
                    prepared_inputs=prepared_inputs,
                    benchmark_examples=examples,
                    device=device,
                    batch_size=config.runtime.batch_size,
                    num_latent=num_latent,
                    max_new_tokens=config.runtime.max_new_tokens,
                    greedy=config.runtime.greedy,
                    temperature=config.runtime.temperature,
                    top_k_sampling=config.runtime.top_k,
                    top_p=config.runtime.top_p,
                    out_path=out_path,
                    summary_path=summary_path,
                    variant_label=model_spec.name,
                    benchmark_name=benchmark_name,
                    intervention_kwargs={
                        "perturb_latent_kv_sigma_mult": sigma,
                        "perturb_seed": 1234,
                    },
                    intervention_label=label,
                    row_meta=row_meta,
                    examples_cap=cap,
                    persistence_every=config.runtime.persistence_every_examples,
                )
                aggregate.setdefault(benchmark_name, {})[label] = res

        # ---- F5: cross-example latent KV swap ----
        if "f5" in tests:
            # Only run F5 on gsm8k (too slow on both benchmarks; 100 pairs ~ 200 forwards per benchmark)
            if benchmark_name != "gsm8k":
                continue
            out_path = output_root / model_spec.name / f"{benchmark_name}_f5_swap" / "predictions.jsonl"
            summary_path = output_root / model_spec.name / f"{benchmark_name}_f5_swap" / "running_summary.json"
            res = _run_f5(
                loaded_model=loaded_model,
                prepared_inputs=prepared_inputs,
                benchmark_examples=examples,
                device=device,
                num_latent=num_latent,
                max_new_tokens=config.runtime.max_new_tokens,
                greedy=True,
                temperature=config.runtime.temperature,
                top_k_sampling=config.runtime.top_k,
                top_p=config.runtime.top_p,
                out_path=out_path,
                summary_path=summary_path,
                variant_label=model_spec.name,
                benchmark_name=benchmark_name,
                n_pairs=args.f5_pairs,
                seed=11,
                row_meta=row_meta,
            )
            aggregate.setdefault(benchmark_name, {})["f5_swap"] = res

    out_json = output_root / "summary.json"
    out_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _log_event(f"final summary -> {out_json}")


if __name__ == "__main__":
    main()
