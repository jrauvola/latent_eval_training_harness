"""F5 — Cross-example latent-KV swap experiment on V2 bf16.

Design
------
For 100 GSM8k test examples, we:

  1. Run V2 bf16 through its full latent rollout (num_latent=8), capturing
     each example's **final KV cache slice at the latent positions**.
     We also record the own-KV answer prediction + correctness.
  2. Build a random permutation ``shuffle`` of the 100 indices. For each
     example B, the "swap partner" A = shuffle[B]. (Sanity: we ensure no
     fixed points — B != A.)
  3. For each B, re-run its encoder + latent rollout, then **overwrite the
     latent-position KV slice** in B's cache with A's captured latent-KV.
     Finally generate B's answer from that modified cache.
  4. Compare own-KV vs swapped-KV accuracy across all 100 examples.

Interpretation
--------------
If swapped-KV accuracy matches own-KV → latents are near-identical across
examples; they're template-only. If accuracy degrades → latents encode
per-example information that matters.

Runtime notes
-------------
- Runs in a single process so the model stays resident. Uses batch_size=1
  for simplicity: we need to pair exactly example A with example B, and
  cross-example swaps inside a batch would require index gymnastics.
- Writes per-example rows to a JSONL file incrementally; a running summary
  JSON is updated after every example so a crash doesn't lose completed
  work (per project policy).

Output layout (under ``runtime.output_dir``):
  - predictions.jsonl : per-example rows with own + swapped predictions.
  - running_summary.json : accuracy-so-far for own / swapped / noise.
  - resolved_config.json : the resolved config for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Path bootstrap — mirror run_evaluation_entrypoint.py so this script can be
# invoked directly without ``pip install -e .``.
ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
blocked_paths = {"", str(ROOT), str(ROOT / "src" / "latent_harness" / "evaluation")}
sys.path = [SRC] + [p for p in sys.path if p not in blocked_paths]

import torch  # noqa: E402

from latent_harness.core.io import dump_yamlable, ensure_dir, load_yaml_config  # noqa: E402
from latent_harness.evaluation.benchmarks import get_benchmark_spec, load_benchmark  # noqa: E402
from latent_harness.evaluation.config import EvaluationConfig  # noqa: E402
from latent_harness.evaluation.latent_tap import (  # noqa: E402
    extract_latent_kv_all_layers,
    generate_from_latent_with_taps,
    generate_with_injected_latent_kv,
)
from latent_harness.evaluation.metrics import score_prediction  # noqa: E402
from latent_harness.evaluation.models import load_evaluation_model  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[f5 {ts}] {msg}", flush=True)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_prepared_inputs(loaded_model, runtime_config, examples):
    """Tokenize each example's prompt individually (batch_size=1)."""
    use_chat = getattr(runtime_config, "use_chat_template", False)
    chat_kw = getattr(runtime_config, "chat_template_kwargs", None) or {}
    tokenizer = loaded_model.tokenizer
    prepared = []
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
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=runtime_config.model_max_length,
        )
        prepared.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        )
    return prepared


def _prepend_bot_token(loaded_model, input_ids, attention_mask):
    """Append BOT (and optionally EOS) to the encoder input, matching stock path."""
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
    return input_ids, attention_mask


def _run_own_kv_and_capture(
    loaded_model,
    prepared,
    runtime,
    num_latent,
    device,
):
    """Run own-KV generation on every example; capture per-example latent-KV."""
    captured_kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    own_predictions: list[str] = []
    tokenizer = loaded_model.tokenizer
    for idx, prep in enumerate(prepared):
        iids = prep["input_ids"].to(device)
        amask = prep["attention_mask"].to(device)
        iids, amask = _prepend_bot_token(loaded_model, iids, amask)
        with torch.no_grad():
            taps = generate_from_latent_with_taps(
                loaded_model.model,
                tokenizer=tokenizer,
                input_ids=iids,
                attention_mask=amask,
                inf_latent_iterations=num_latent,
                max_new_tokens=runtime.max_new_tokens,
                greedy=runtime.greedy,
                temperature=runtime.temperature,
                top_k=runtime.top_k,
                top_p=runtime.top_p,
                skip_latent_injection=False,
                capture_latent_hidden=False,
            )
        own_predictions.append(taps.predictions[0])
        # Extract the latent-position KV on CPU to free GPU memory.
        kv_per_layer = extract_latent_kv_all_layers(
            taps.final_cache,
            encoder_prefix_length=taps.encoder_prefix_length,
            num_latent=taps.num_latent_iterated,
        )
        cpu_kv = [(k.cpu(), v.cpu()) for k, v in kv_per_layer]
        captured_kv.append(cpu_kv)
        del taps
        if (idx + 1) % 10 == 0:
            _log(f"own-KV pass: {idx + 1}/{len(prepared)} done")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return own_predictions, captured_kv


def _run_swapped_kv(
    loaded_model,
    prepared,
    captured_kv,
    pairing,
    runtime,
    num_latent,
    device,
    *,
    noise_only: bool = False,
):
    """For each B, run generation with A = pairing[B]'s latent-KV injected."""
    tokenizer = loaded_model.tokenizer
    predictions: list[str] = []
    for b_idx, prep in enumerate(prepared):
        a_idx = pairing[b_idx]
        injected = captured_kv[a_idx]
        # Move to device matching the model.
        injected_dev = [(k.to(device), v.to(device)) for k, v in injected]
        iids = prep["input_ids"].to(device)
        amask = prep["attention_mask"].to(device)
        iids, amask = _prepend_bot_token(loaded_model, iids, amask)
        with torch.no_grad():
            taps = generate_with_injected_latent_kv(
                loaded_model.model,
                tokenizer=tokenizer,
                input_ids=iids,
                attention_mask=amask,
                inf_latent_iterations=num_latent,
                max_new_tokens=runtime.max_new_tokens,
                greedy=runtime.greedy,
                temperature=runtime.temperature,
                top_k=runtime.top_k,
                top_p=runtime.top_p,
                injected_kv_per_layer=injected_dev,
                noise_only=noise_only,
            )
        predictions.append(taps.predictions[0])
        del taps, injected_dev
        if (b_idx + 1) % 10 == 0:
            mode = "noise" if noise_only else "swap"
            _log(f"{mode}-KV pass: {b_idx + 1}/{len(prepared)} done")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return predictions


def _score_all(examples, predictions):
    scores = [score_prediction(ex, p) for ex, p in zip(examples, predictions)]
    correct = sum(1 for s in scores if s.correct)
    return scores, correct


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_pairing(n: int, seed: int = 11) -> list[int]:
    """Build a derangement-like pairing: shuffle indices and ensure no fixed points."""
    rng = random.Random(seed)
    indices = list(range(n))
    # Try a simple rotation first (guarantees no fixed points for n>=2).
    shift = n // 2
    pairing = [(i + shift) % n for i in range(n)]
    # Shuffle via a derangement: swap positions until random + no self-pair.
    # The rotation by n/2 already ensures i != pairing[i] for n >= 2.
    # Add random permutation on top.
    order = list(range(n))
    rng.shuffle(order)
    # Apply: new_pairing[i] = pairing[order[i]] if it doesn't hit i.
    # Simpler: shuffle pairing list under a derangement constraint.
    # Build by swap-walk: keep shuffling until no fixed points.
    for _ in range(100):
        rng.shuffle(pairing)
        if all(pairing[i] != i for i in range(n)):
            return pairing
    # Fallback: if we can't find one by chance, force with rotation.
    return [(i + shift) % n for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="F5 — cross-example latent-KV swap on V2 bf16")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "evaluation" / "qwen3_4b_codi_v2_f5_kv_swap.yaml"),
        help="Path to the F5 evaluation config YAML.",
    )
    parser.add_argument(
        "--noise-also",
        action="store_true",
        help="Run a third pass that overwrites latent KV with random Gaussian noise "
        "of matching magnitude — the weaker fallback test.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override runtime.max_examples_per_benchmark at the CLI.",
    )
    args = parser.parse_args()

    payload = load_yaml_config(args.config)
    config = EvaluationConfig.from_dict(payload)
    runtime = config.runtime
    if args.num_examples is not None:
        runtime.max_examples_per_benchmark = args.num_examples

    # F5 is single-variant, single-benchmark, single num_latent.
    if len(config.models) != 1:
        raise ValueError(f"F5 expects exactly 1 model, got {len(config.models)}")
    if len(config.benchmarks) != 1:
        raise ValueError(f"F5 expects exactly 1 benchmark, got {len(config.benchmarks)}")
    num_latent = (runtime.num_latent_sweep or [config.models[0].runtime.num_latent])[0]
    benchmark_name = config.benchmarks[0]

    device = _resolve_device(runtime.device)
    output_dir = ensure_dir(runtime.output_dir)
    ensure_dir(runtime.snapshot_dir)
    _log(
        f"starting F5 kv-swap experiment config={args.config} device={device} "
        f"num_latent={num_latent} benchmark={benchmark_name}"
    )

    (Path(output_dir) / "resolved_config.json").write_text(
        json.dumps(dump_yamlable(payload), indent=2), encoding="utf-8"
    )

    model_spec = config.models[0]
    loaded_model = load_evaluation_model(model_spec, device=device)
    loaded_model.runtime_config.num_latent = num_latent
    _log(f"loaded model name={model_spec.name}")

    loaded_benchmark = load_benchmark(
        get_benchmark_spec(benchmark_name),
        cache_dir=runtime.cache_dir,
        snapshot_dir=runtime.snapshot_dir,
        max_examples=runtime.max_examples_per_benchmark,
    )
    examples = loaded_benchmark.examples
    n = len(examples)
    _log(f"loaded benchmark={benchmark_name} examples={n}")

    prepared = _build_prepared_inputs(loaded_model, loaded_model.runtime_config, examples)

    predictions_path = Path(output_dir) / "predictions.jsonl"
    summary_path = Path(output_dir) / "running_summary.json"
    if predictions_path.exists():
        predictions_path.unlink()

    started = time.perf_counter()

    # --- PASS 1: own-KV predictions, capture each example's latent-KV ---
    _log("pass 1 starting: own-KV generation + latent-KV capture")
    own_preds, captured_kv = _run_own_kv_and_capture(
        loaded_model, prepared, runtime, num_latent, device
    )
    own_scores, own_correct = _score_all(examples, own_preds)
    _log(f"pass 1 complete: own-KV accuracy = {own_correct}/{n} = {own_correct / n:.4f}")

    # --- Build pairing ---
    pairing = _make_pairing(n, seed=runtime.seed)
    n_self = sum(1 for i in range(n) if pairing[i] == i)
    _log(f"pairing built: {n} examples, {n_self} self-pairs (should be 0), method=random-derangement")

    # --- PASS 2: swapped-KV ---
    _log("pass 2 starting: swapped-KV generation")
    swap_preds = _run_swapped_kv(
        loaded_model, prepared, captured_kv, pairing, runtime, num_latent, device, noise_only=False
    )
    swap_scores, swap_correct = _score_all(examples, swap_preds)
    _log(f"pass 2 complete: swapped-KV accuracy = {swap_correct}/{n} = {swap_correct / n:.4f}")

    # --- (optional) PASS 3: noise-only ---
    noise_preds = []
    noise_correct = 0
    if args.noise_also:
        _log("pass 3 starting: noise-KV generation (Gaussian random, matching magnitude)")
        noise_preds = _run_swapped_kv(
            loaded_model, prepared, captured_kv, pairing, runtime, num_latent, device, noise_only=True
        )
        _, noise_correct = _score_all(examples, noise_preds)
        _log(f"pass 3 complete: noise-KV accuracy = {noise_correct}/{n} = {noise_correct / n:.4f}")

    # --- Write per-example rows ---
    rows = []
    for i, ex in enumerate(examples):
        row = {
            "example_id": ex.example_id,
            "example_index": i,
            "pairing_a_index": pairing[i],
            "target": ex.target,
            "prompt_excerpt": (ex.prompt or "")[:200],
            "own_prediction": own_preds[i],
            "own_parsed": own_scores[i].parsed_prediction.value,
            "own_correct": own_scores[i].correct,
            "swap_prediction": swap_preds[i],
            "swap_parsed": swap_scores[i].parsed_prediction.value,
            "swap_correct": swap_scores[i].correct,
        }
        if args.noise_also:
            row["noise_prediction"] = noise_preds[i]
            noise_score = score_prediction(ex, noise_preds[i])
            row["noise_parsed"] = noise_score.parsed_prediction.value
            row["noise_correct"] = noise_score.correct
        rows.append(row)

    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "experiment": "F5_kv_swap",
        "model": model_spec.name,
        "benchmark": benchmark_name,
        "num_latent": num_latent,
        "n_examples": n,
        "pairing_method": "random_derangement_seed_{}".format(runtime.seed),
        "n_self_pairs": n_self,
        "own_kv_accuracy": own_correct / n,
        "swapped_kv_accuracy": swap_correct / n,
        "own_kv_correct": own_correct,
        "swapped_kv_correct": swap_correct,
        "agreement_own_vs_swap": sum(
            1 for i in range(n) if own_scores[i].parsed_prediction.value == swap_scores[i].parsed_prediction.value
        ) / n,
        "wall_time_s": time.perf_counter() - started,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.noise_also:
        summary["noise_kv_accuracy"] = noise_correct / n
        summary["noise_kv_correct"] = noise_correct

    _write_summary(summary_path, summary)
    _log(f"wrote summary: {summary_path}")
    _log(f"wrote predictions: {predictions_path}")
    _log(
        f"FINAL: own={summary['own_kv_accuracy']:.4f} swap={summary['swapped_kv_accuracy']:.4f} "
        f"agreement={summary['agreement_own_vs_swap']:.4f}"
    )


if __name__ == "__main__":
    main()
