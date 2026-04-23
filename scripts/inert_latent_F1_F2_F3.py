"""Inert-latent hypothesis tests F1/F2/F3 — pure local analyses.

F1: unique-correct analysis — problems V2 bf16 gets right that zero-shot gets WRONG.
F2: question-specific loop content — do V2 bf16's loop substrings contain question digits?
F3: per-latent-position trace variance — entropy of top-1 tokens across examples per latent step.

Reads predictions.jsonl from the phase 1 eval artifacts (V2 bf16, P0, V2', V3', zero-shot).
Reads latent traces from research_findings/latent_traces/.../gsm8k/numlatent_8/.
Writes results JSON + human-friendly print output usable for the integrated deliverable.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path("/Users/jrauvola/Desktop/Latent_Reasoning_Project")
PHASE1_EVAL = REPO_ROOT / "latent_eval_training_harness/artifacts/eval/qwen3_4b_codi_phase1"
ZEROSHOT_EVAL = REPO_ROOT / "research_findings/eval/qwen3_4b_zero_shot_cot/qwen3_4b_instruct_2507_zero_shot_cot"
TRACES_ROOT = REPO_ROOT / "research_findings/latent_traces"

V2 = "qwen3_4b_codi_bf16_kv_latent_detach_last_2"

BENCHMARKS = ["gsm8k", "gsm-hard", "svamp"]

OUT_JSON = REPO_ROOT / "research_findings/inert_latent_F1_F2_F3.json"


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def detect_loop_substring(text: str, min_len: int = 15, min_repeats: int = 3) -> str | None:
    """Find the longest substring of length >= min_len that occurs >= min_repeats times."""
    if not text:
        return None
    n = len(text)
    # Greedy-ish: try windows centered on common anchors, or scan a range.
    # To keep this tractable, sample candidate lengths and find repeats.
    best: str | None = None
    # Search from long to short; stop when we find the longest repeating substring.
    # Limit to lengths up to 120 chars (loops usually ~40-100 chars per repeat).
    max_len = min(n // min_repeats, 160)
    for L in range(max_len, min_len - 1, -1):
        # Scan start positions
        seen: dict[str, int] = {}
        for start in range(0, n - L + 1, max(1, L // 4)):  # step fraction of L
            sub = text[start:start + L]
            seen[sub] = seen.get(sub, 0) + 1
            if seen[sub] >= min_repeats:
                best = sub
                break
        if best is not None:
            return best
    # Also try standard ordered scan at a few common lengths as fallback
    for L in (60, 40, 25, min_len):
        if L > n // min_repeats:
            continue
        seen2: Counter[str] = Counter()
        for start in range(n - L + 1):
            sub = text[start:start + L]
            seen2[sub] += 1
            if seen2[sub] >= min_repeats:
                return sub
    return None


# ---------- F1: unique-correct analysis ----------

def f1_unique_correct() -> dict:
    """Problems V2 bf16 (n=8) gets right that zero-shot gets wrong.

    For each benchmark: build V2_correct set and zshot_correct set keyed by example_id.
    Report size, magnitude stats, sample questions, and the set per benchmark.
    """
    results = {}
    for bench in BENCHMARKS:
        # V2 bf16 uses num_latent=8 (where its accuracy peaks).
        v2_path = PHASE1_EVAL / V2 / f"{bench}_numlatent_8" / "predictions.jsonl"
        zs_path = ZEROSHOT_EVAL / f"{bench}_numlatent_0" / "predictions.jsonl"
        v2_rows = load_predictions(v2_path)
        zs_rows = load_predictions(zs_path)
        v2_by_id = {r["example_id"]: r for r in v2_rows}
        zs_by_id = {r["example_id"]: r for r in zs_rows}
        shared = sorted(set(v2_by_id) & set(zs_by_id))

        v2_correct = {ex for ex in shared if v2_by_id[ex].get("correct")}
        zs_correct = {ex for ex in shared if zs_by_id[ex].get("correct")}
        unique_v2 = sorted(v2_correct - zs_correct)
        # Also compute direction: zs_correct - v2_correct (regression)
        unique_zs = sorted(zs_correct - v2_correct)

        # Target magnitudes in each set
        def magnitudes(ids):
            mags = []
            for ex in ids:
                t = v2_by_id[ex].get("target")
                try:
                    mags.append(abs(float(t)))
                except Exception:
                    pass
            return mags

        v2_mags = magnitudes(unique_v2)
        zs_mags = magnitudes(unique_zs)

        def median(xs):
            if not xs:
                return None
            s = sorted(xs)
            n = len(s)
            return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

        # Sample questions
        sample_questions = []
        for ex in unique_v2[:10]:
            r = v2_by_id[ex]
            sample_questions.append({
                "example_id": ex,
                "target": r.get("target"),
                "v2_parsed": r.get("parsed_prediction"),
                "v2_pred_text_start": (r.get("prediction_text") or "")[:180],
                "question_start": (r.get("prompt") or "")[:180],
            })

        results[bench] = {
            "n_shared": len(shared),
            "n_v2_correct": len(v2_correct),
            "n_zshot_correct": len(zs_correct),
            "n_unique_to_v2": len(unique_v2),
            "n_unique_to_zshot": len(unique_zs),
            "unique_to_v2_ids": unique_v2,
            "v2_unique_target_median_abs": median(v2_mags),
            "zshot_unique_target_median_abs": median(zs_mags),
            "v2_unique_target_max_abs": max(v2_mags) if v2_mags else None,
            "zshot_unique_target_max_abs": max(zs_mags) if zs_mags else None,
            "sample_unique_v2_questions": sample_questions,
        }
    return results


# ---------- F2: question-specific loop content ----------

def f2_loop_question_digits(n_sample: int = 100, seed: int = 0) -> dict:
    """For a random sample of V2 bf16 looped predictions, check if the loop
    substring contains any digit that appears in the question.
    """
    random.seed(seed)
    results = {}
    for bench in BENCHMARKS:
        v2_path = PHASE1_EVAL / V2 / f"{bench}_numlatent_8" / "predictions.jsonl"
        rows = load_predictions(v2_path)
        # Find looping predictions
        looped = []
        for r in rows:
            text = r.get("prediction_text", "") or ""
            loop_sub = detect_loop_substring(text)
            if loop_sub is not None:
                looped.append((r, loop_sub))
        if not looped:
            results[bench] = {"error": "no loops detected"}
            continue
        if len(looped) > n_sample:
            sample = random.sample(looped, n_sample)
        else:
            sample = looped

        total = 0
        loop_has_question_digit = 0
        loop_digits_any = 0
        examples = []
        for r, loop_sub in sample:
            q = r.get("prompt", "") or ""
            q_digits = set(extract_digits(q))
            loop_digits = set(extract_digits(loop_sub))
            # Target is likely in loop (format-prior emission)
            total += 1
            if loop_digits:
                loop_digits_any += 1
            shared = q_digits & loop_digits
            if shared:
                loop_has_question_digit += 1
            if len(examples) < 5:
                examples.append({
                    "example_id": r.get("example_id"),
                    "target": r.get("target"),
                    "q_digits": sorted(q_digits)[:10],
                    "loop_digits": sorted(loop_digits)[:10],
                    "shared": sorted(shared),
                    "loop_substring": loop_sub[:160],
                })
        # Also compute: fraction where loop digit EQUALS target
        target_in_loop = 0
        for r, loop_sub in sample:
            target = str(r.get("target", ""))
            # Normalise target: extract numeric
            tnums = extract_digits(target)
            if tnums and tnums[0] in extract_digits(loop_sub):
                target_in_loop += 1

        results[bench] = {
            "n_total_predictions": len(rows),
            "n_looped": len(looped),
            "fraction_looped": len(looped) / len(rows),
            "n_sample": total,
            "fraction_loop_has_any_digit": loop_digits_any / max(total, 1),
            "fraction_loop_has_question_digit": loop_has_question_digit / max(total, 1),
            "fraction_loop_has_target": target_in_loop / max(total, 1),
            "sample_examples": examples,
        }
    return results


# ---------- F3: per-latent-position trace variance ----------

def f3_trace_variance(num_latent: int = 8) -> dict:
    """For each latent_step 0..N-1, compute entropy of top-1 token IDs across examples.

    Uses trace files at research_findings/latent_traces/{V2}/gsm8k/numlatent_8/trace_*.jsonl.
    Each file contains N rows (one per latent step), each with top10_token_ids etc.
    """
    traces_dir = TRACES_ROOT / V2 / "gsm8k" / f"numlatent_{num_latent}"
    trace_files = sorted(traces_dir.glob("trace_*.jsonl"))
    # Per-step: list of top-1 token_id across all examples
    per_step_top1: dict[int, list[int]] = {k: [] for k in range(num_latent)}
    per_step_top1_str: dict[int, list[str]] = {k: [] for k in range(num_latent)}
    per_step_top1_prob: dict[int, list[float]] = {k: [] for k in range(num_latent)}
    for tf in trace_files:
        with tf.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = row.get("latent_step")
                if step is None or step < 0 or step >= num_latent:
                    continue
                ids = row.get("top10_token_ids") or []
                probs = row.get("top10_probs") or []
                strs = row.get("top10_token_strs") or []
                if ids:
                    per_step_top1[step].append(int(ids[0]))
                    per_step_top1_str[step].append(str(strs[0]) if strs else "")
                    per_step_top1_prob[step].append(float(probs[0]) if probs else 0.0)

    results = {}
    for step in range(num_latent):
        ids = per_step_top1[step]
        if not ids:
            results[step] = {"error": "no rows"}
            continue
        c = Counter(ids)
        total = len(ids)
        entropy = 0.0
        for _, cnt in c.items():
            p = cnt / total
            entropy -= p * math.log2(p)
        top5_ids = c.most_common(5)
        # Map most-common id to token string (use the mode's string from parallel list)
        id_to_str = dict(zip(ids, per_step_top1_str[step]))
        top5 = [
            {
                "token_id": tid,
                "count": cnt,
                "fraction": cnt / total,
                "token_str": id_to_str.get(tid, ""),
            }
            for tid, cnt in top5_ids
        ]
        avg_top1_prob = sum(per_step_top1_prob[step]) / total
        results[step] = {
            "n_examples": total,
            "unique_top1_tokens": len(c),
            "entropy_bits": entropy,
            "top5_tokens": top5,
            "dominant_token_fraction": top5_ids[0][1] / total,
            "avg_top1_prob": avg_top1_prob,
        }
    return {"variant": V2, "benchmark": "gsm8k", "num_latent": num_latent, "per_step": results}


def main():
    print("Running F1 — unique-correct analysis")
    f1 = f1_unique_correct()
    for bench, stats in f1.items():
        print(
            f"  [{bench}] shared={stats['n_shared']} V2-correct={stats['n_v2_correct']} "
            f"zshot-correct={stats['n_zshot_correct']} "
            f"unique-V2={stats['n_unique_to_v2']} unique-zshot={stats['n_unique_to_zshot']} "
            f"V2-unique median |target|={stats['v2_unique_target_median_abs']}"
        )

    print("\nRunning F2 — question-specific loop content")
    f2 = f2_loop_question_digits(n_sample=100, seed=0)
    for bench, stats in f2.items():
        if "error" in stats:
            print(f"  [{bench}] {stats}")
            continue
        print(
            f"  [{bench}] looped={stats['n_looped']}/{stats['n_total_predictions']} "
            f"({stats['fraction_looped']:.2%}) | "
            f"loop-has-question-digit={stats['fraction_loop_has_question_digit']:.2%} | "
            f"loop-has-target={stats['fraction_loop_has_target']:.2%}"
        )

    print("\nRunning F3 — per-latent-position trace variance")
    f3 = f3_trace_variance(num_latent=8)
    for step, stats in f3["per_step"].items():
        if "error" in stats:
            print(f"  step {step}: {stats}")
            continue
        print(
            f"  step {step}: n={stats['n_examples']} unique_top1={stats['unique_top1_tokens']} "
            f"entropy={stats['entropy_bits']:.3f} bits | dominant={stats['top5_tokens'][0]['token_str']!r} "
            f"({stats['dominant_token_fraction']:.2%}) avg_p1={stats['avg_top1_prob']:.3f}"
        )

    out = {"F1": f1, "F2": f2, "F3": f3}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
