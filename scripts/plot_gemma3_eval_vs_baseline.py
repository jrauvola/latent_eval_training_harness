#!/usr/bin/env python3
"""Bar chart: Gemma-3-4B-it baseline (standard gen) vs CODI checkpoints (latent_cot).

Data sources (checked into artifacts):
  - Baseline: artifacts/eval/paper_core_remaining_gh200_v4/summary.csv
    model base_gemma3_4b_it, n=32 per benchmark, inference_strategy=standard_generation
  - CODI: artifacts/eval/gemma3_4b_codi_gh200_lora16_detach_accum1_compare/summary.csv
    n=64 per benchmark, inference_strategy=latent_cot

⚠️ Sample sizes differ (32 vs 64); treat small deltas as noisy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BENCHMARK_ORDER = ["gsm8k", "svamp", "gsm-hard", "multi-arith", "commonsense_qa"]


def default_paths(harness_root: Path) -> tuple[Path, Path, Path]:
    ev = harness_root / "artifacts" / "eval"
    return (
        ev / "paper_core_remaining_gh200_v4" / "summary.csv",
        ev / "gemma3_4b_codi_gh200_lora16_detach_accum1_compare" / "summary.csv",
        harness_root / "artifacts" / "reports" / "gemma3_codi_stability_mpl",
    )


def load_series(csv_path: Path, model_name: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    sub = df[df["model"] == model_name].set_index("benchmark")["accuracy"]
    return sub.reindex(BENCHMARK_ORDER)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for eval_vs_baseline.png (default: artifacts/reports/gemma3_codi_stability_mpl)",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline_csv, codi_csv, default_mpl = default_paths(root)
    out_dir = args.out_dir or default_mpl
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_series(baseline_csv, "base_gemma3_4b_it")
    probe = load_series(codi_csv, "gemma3_4b_codi_lora16_detach_accum1_probe_seed42")
    full = load_series(codi_csv, "gemma3_4b_codi_lora16_detach_accum1_full_ckpt200")

    x = np.arange(len(BENCHMARK_ORDER))
    w = 0.25

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, base.values, width=w, label="Baseline: Gemma-3-4B-it (standard gen, n=32)", color="#6b7280")
    ax.bar(x, probe.values, width=w, label="CODI probe seed42 (latent_cot, n=64)", color="#ff4757")
    ax.bar(x + w, full.values, width=w, label="CODI full ckpt-200 (latent_cot, n=64)", color="#0be881")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARK_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Paper-core benchmarks: frozen base vs CODI LoRA (note mixed n)")
    fig.text(
        0.5,
        0.02,
        "Baseline run: paper_core_remaining_gh200_v4 (32 ex/benchmark). CODI: gemma3_4b_codi_gh200_lora16_detach_accum1_compare (64 ex).",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = out_dir / "eval_accuracy_vs_baseline.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
