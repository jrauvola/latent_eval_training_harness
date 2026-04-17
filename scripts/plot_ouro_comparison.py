#!/usr/bin/env python3
"""Plot Ouro-2.6B-Base vs harness comparison models on GSM8K + commonsense_qa.

Reads summary.csv from several eval artifact directories, computes Wilson
95% binomial CI from accuracy + num_examples, and emits one PNG per
benchmark to artifacts/eval/<ouro_run>/plots/.

Usage:
    python scripts/plot_ouro_comparison.py [--ouro-dir artifacts/eval/ouro_2.6b_base_sanity]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARKS = ("gsm8k", "commonsense_qa")

# Paper numbers from Table 7/8 of arXiv:2510.25741v4 (Ouro-2.6B row and peers).
PAPER_REFERENCES = {
    "gsm8k": {
        "Ouro-2.6B R=4 (paper)": 81.58,
        "Qwen3-4B (paper)": 72.86,
        "Qwen3-8B (paper)": 83.09,
        "Gemma3-4B (paper)": 68.69,
        "Gemma3-12B (paper)": 77.18,
    },
    # Paper does not report CSQA for Ouro; keep empty so no paper markers appear.
    "commonsense_qa": {},
}

# Which artifact summaries contribute comparison models. Each entry:
#   (summary_csv_path_relative_to_harness, set_of_model_names_to_include)
# If the set is None, all models in that summary are included.
COMPARISON_SOURCES = [
    (
        "artifacts/eval/paper_faithful_broader_suite_gh200_combined/summary.csv",
        {
            "base_gpt2",
            "base_llama32_1b_instruct",
            "codi_gpt2_official",
            "codi_llama32_1b_official",
        },
    ),
    (
        "artifacts/eval/gemma3_4b_codi_gh200_full_detach_full_epoch_eval/summary.csv",
        {"gemma3_4b_codi_full_detach_full_epoch"},
    ),
]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95% CI by default). Well-behaved at small N / extreme p."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def read_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                row["accuracy"] = float(row["accuracy"])
                row["num_examples"] = int(row["num_examples"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def collect_rows(harness_root: Path, ouro_dir: Path) -> dict[str, list[dict]]:
    """Return {benchmark: [row, ...]} where each row is (model_label, accuracy, n, source)."""
    out: dict[str, list[dict]] = {b: [] for b in BENCHMARKS}

    # Ouro run first (will render as the leftmost bar after sorting).
    ouro_rows = read_summary(ouro_dir / "summary.csv")
    for r in ouro_rows:
        bm = r["benchmark"]
        if bm in BENCHMARKS:
            out[bm].append(
                {
                    "label": f"{r['model']} (this run, n={r['num_examples']})",
                    "accuracy": r["accuracy"],
                    "n": r["num_examples"],
                    "source": str(ouro_dir),
                    "is_ouro": True,
                }
            )

    # Comparison runs.
    for rel_path, allowed_models in COMPARISON_SOURCES:
        rows = read_summary(harness_root / rel_path)
        for r in rows:
            if allowed_models is not None and r["model"] not in allowed_models:
                continue
            bm = r["benchmark"]
            if bm not in BENCHMARKS:
                continue
            out[bm].append(
                {
                    "label": f"{r['model']} (n={r['num_examples']})",
                    "accuracy": r["accuracy"],
                    "n": r["num_examples"],
                    "source": rel_path,
                    "is_ouro": False,
                }
            )
    return out


def plot_benchmark(benchmark: str, entries: list[dict], out_path: Path) -> None:
    if not entries:
        print(f"no entries for {benchmark}; skipping")
        return

    # Keep Ouro first, then sort comparisons by accuracy descending.
    ouro = [e for e in entries if e["is_ouro"]]
    others = sorted((e for e in entries if not e["is_ouro"]), key=lambda e: -e["accuracy"])
    entries_sorted = ouro + others

    labels = [e["label"] for e in entries_sorted]
    accuracies = np.array([e["accuracy"] for e in entries_sorted]) * 100
    ns = [e["n"] for e in entries_sorted]

    # Error bars: Wilson 95% CI in percent.
    lowers, uppers = [], []
    for e in entries_sorted:
        k = round(e["accuracy"] * e["n"])
        lo, hi = wilson_interval(k, e["n"])
        lowers.append((e["accuracy"] - lo) * 100)
        uppers.append((hi - e["accuracy"]) * 100)
    yerr = np.array([lowers, uppers])

    colors = ["#d62728" if e["is_ouro"] else "#4c72b0" for e in entries_sorted]

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(entries_sorted) + 3), 5.5))
    x = np.arange(len(entries_sorted))
    bars = ax.bar(x, accuracies, yerr=yerr, capsize=4, color=colors, edgecolor="black")

    # Paper reference horizontal dashed lines.
    refs = PAPER_REFERENCES.get(benchmark, {})
    for i, (name, val) in enumerate(refs.items()):
        ax.axhline(val, linestyle="--", linewidth=1.0, alpha=0.6, color=f"C{i+2}")
        ax.text(
            len(entries_sorted) - 0.5,
            val + 0.5,
            f"{name}: {val:.1f}",
            ha="right",
            va="bottom",
            fontsize=8,
            color=f"C{i+2}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(100, accuracies.max() + max(uppers) + 10))
    ax.set_title(f"{benchmark} — Ouro-2.6B-Base vs harness baselines (95% Wilson CI)")
    ax.grid(axis="y", alpha=0.25)
    # Value labels on bars.
    for rect, acc, n in zip(bars, accuracies, ns):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            f"{acc:.1f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ouro-dir",
        type=Path,
        default=Path("artifacts/eval/ouro_2.6b_base_sanity"),
        help="Path to the Ouro eval artifact dir containing summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination for plots; defaults to <ouro-dir>/plots/.",
    )
    args = parser.parse_args()

    harness_root = Path(__file__).resolve().parent.parent
    ouro_dir = args.ouro_dir
    if not ouro_dir.is_absolute():
        ouro_dir = harness_root / ouro_dir
    out_dir = args.out_dir or (ouro_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(harness_root, ouro_dir)
    for bm in BENCHMARKS:
        out_path = out_dir / f"{bm}_comparison.png"
        plot_benchmark(bm, rows[bm], out_path)

    print(f"all plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
