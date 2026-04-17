#!/usr/bin/env python3
"""Recreate gemma3_codi_stability_dashboard.html charts with matplotlib.

Data matches the embedded arrays in artifacts/reports/gemma3_codi_stability_dashboard.html.
Default output: artifacts/reports/gemma3_codi_stability_mpl/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Same series as dashboard (Chart.js) ---
fail_steps_18 = list(range(1, 19))
fail_loss_18 = [
    5.17, 4.96, 4.27, 4.77, 7.02, 4.35, 3.88, 4.44, 4.16, 6.06, 4.87, 3.97, 4.71,
    4.85, 3.62, 5.52, 4.77, 5.28,
]
fail_grad_18 = [
    10.87, 9.39, 5.48, 6.49, 14.27, 11.75, 10.67, 8.89, 8.81, 15.25, 17.02, 2.76,
    5.61, 8.88, 6.79, 15.49, 9.88, 101.33,
]

fail_steps_55 = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
fail_loss_55 = [5.12, 6.18, 7.39, 6.23, 4.31, 3.89, 3.48, 4.36, 4.35, 2.49, 1.99, 2.91]
fail_grad_55 = [7.05, 8.18, 5.95, 5.27, 2.51, 1.94, 2.67, 5.25, 5.22, 1.99, 11.81, 2.74]

fail_steps_223 = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 223]
fail_loss_223 = [2.81, 4.31, 2.53, 4.41, 2.18, 4.32, 4.04, 12.46, 13.46, 5.35, 2.35, 4.56, 3.84]
fail_grad_223 = [3.78, 62.5, 54.9, 20.35, 6.19, 10.8, 14.37, 104.81, 88.15, 37.67, 3.85, 13.87, 6.65]

success_steps = [
    1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000,
    7500, 8000, 8500, 9000, 9500, 10000, 10500, 10603,
]
success_loss = [
    2.81, 2.65, 1.41, 1.06, 1.35, 1.97, 0.24, 0.42, 1.22, 1.04, 0.65, 1.57, 0.69, 0.65,
    0.19, 0.28, 0.53, 0.18, 0.80, 0.59, 0.43, 1.02, 0.88,
]
success_grad = [
    2.64, 41.6, 3.8, 3.33, 4.95, 5.71, 4.70, 3.21, 3.28, 2.93, 6.04, 2.38, 1.97, 7.60,
    1.61, 2.15, 4.0, 12.16, 3.54, 2.87, 3.80, 2.83, 3.13,
]

COL_FAIL_18 = "#ff4757"
COL_FAIL_55 = "#ff9f43"
COL_FAIL_223 = "#feca57"
COL_SUCCESS = "#0be881"

EXPLODE_GRAD_THRESHOLD = 50.0


def _peak_grad_annotation(steps: list[int], grads: list[float]) -> tuple[int, float, str]:
    """Return (step, grad, label) for the gradient spike to highlight on loss plots."""
    g = np.array(grads, dtype=float)
    s = np.array(steps, dtype=int)
    i = int(np.argmax(g))
    peak_s, peak_g = int(s[i]), float(g[i])
    if peak_g >= EXPLODE_GRAD_THRESHOLD:
        tag = "Gradient explosion"
    else:
        tag = "Peak grad norm"
    label = f"{tag}\nstep {peak_s}\n‖g‖={peak_g:.1f}"
    return peak_s, peak_g, label


def _extra_explosion_vlines(
    steps: list[int], grads: list[float], threshold: float
) -> list[tuple[int, float]]:
    """Additional spike points (e.g. second peak on long fail run)."""
    out = [(int(s), float(g)) for s, g in zip(steps, grads) if g >= threshold]
    return out


def _top_spike_steps(steps: list[int], grads: list[float], top_k: int = 2) -> list[tuple[int, float]]:
    """Return the top-K grad spikes by magnitude, sorted by step."""
    ranked = sorted(zip(steps, grads, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted((int(s), float(g)) for s, g in ranked)


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "gemma3_codi_stability_mpl"


def plot_loss_failed_with_explosion(out: Path, dpi: int) -> None:
    """Loss for failed runs only, with vertical callouts where grad norms explode."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 5))

    runs = [
        (fail_steps_18, fail_loss_18, fail_grad_18, COL_FAIL_18, "Fail @ step 18 (no detach)"),
        (fail_steps_55, fail_loss_55, fail_grad_55, COL_FAIL_55, "Fail @ step 55 (latent detach)"),
        (fail_steps_223, fail_loss_223, fail_grad_223, COL_FAIL_223, "Fail @ step 223 (accum=1)"),
    ]

    for steps, loss, grad, color, label in runs:
        ax.plot(steps, loss, "o-", color=color, lw=2, ms=4, label=label, zorder=2)

    # Highlight the late catastrophic region around the two largest spikes.
    spikes_223 = _top_spike_steps(fail_steps_223, fail_grad_223, top_k=2)
    if len(spikes_223) >= 2:
        x0 = min(s for s, _ in spikes_223)
        x1 = max(s for s, _ in spikes_223)
        ax.axvspan(x0 - 8, x1 + 8, alpha=0.12, color=COL_FAIL_223, zorder=0)
        ax.text(
            (x0 + x1) / 2,
            14.2,
            "Largest late spikes\n(steps 140 & 160, ‖g‖≈105 & 88)",
            ha="center",
            va="top",
            fontsize=9,
            color="#8b5900",
        )

    for steps, loss, grad, color, _label in runs:
        peak_s, peak_g, ann = _peak_grad_annotation(steps, grad)
        # loss value at peak step (interpolate from nearest logged point)
        step_to_loss = dict(zip(steps, loss, strict=True))
        ly = step_to_loss.get(peak_s)
        if ly is None:
            ly = float(np.interp(peak_s, steps, loss))
        ax.axvline(peak_s, color=color, ls="--", lw=1.6, alpha=0.85, zorder=1)
        ax.scatter([peak_s], [ly], s=110, c=color, marker="D", edgecolors="white", linewidths=1, zorder=4)
        ax.annotate(
            ann,
            xy=(peak_s, ly),
            xytext=(12, 18 if color != COL_FAIL_223 else -28),
            textcoords="offset points",
            fontsize=8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, alpha=0.92),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=4),
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 16)
    ax.set_title("Failed runs: training loss with gradient-explosion markers (‖g‖ from same run)")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_comparison(out: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fail_steps_18, fail_loss_18, "o-", color=COL_FAIL_18, lw=1.8, ms=3, label="Fail @ step 18 (no detach)")
    ax.plot(fail_steps_55, fail_loss_55, "o-", color=COL_FAIL_55, lw=1.8, ms=3, label="Fail @ step 55 (latent detach)")
    ax.plot(fail_steps_223, fail_loss_223, "o-", color=COL_FAIL_223, lw=1.8, ms=3, label="Fail @ step 223 (accum=1)")
    ax.plot(
        success_steps,
        success_loss,
        "o-",
        color=COL_SUCCESS,
        lw=2.2,
        ms=3,
        label="SUCCESS: Full Detach (10,603 steps)",
    )
    ax.fill_between(success_steps, success_loss, alpha=0.12, color=COL_SUCCESS)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 16)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Training loss: all runs compared")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grad_failed(out: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(fail_steps_18, fail_grad_18, "o-", color=COL_FAIL_18, lw=1.5, ms=3, label="Fail @ 18")
    ax.plot(fail_steps_55, fail_grad_55, "o-", color=COL_FAIL_55, lw=1.5, ms=3, label="Fail @ 55")
    ax.plot(fail_steps_223, fail_grad_223, "o-", color=COL_FAIL_223, lw=1.5, ms=3, label="Fail @ 223")
    ax.axhline(EXPLODE_GRAD_THRESHOLD, color="#888", ls=":", lw=1, alpha=0.8, label=f"‖g‖ = {EXPLODE_GRAD_THRESHOLD:.0f} (spike threshold)")

    for steps, grad, color, name in (
        (fail_steps_18, fail_grad_18, COL_FAIL_18, "18"),
        (fail_steps_55, fail_grad_55, COL_FAIL_55, "55"),
        (fail_steps_223, fail_grad_223, COL_FAIL_223, "223"),
    ):
        peak_s, peak_g, _ = _peak_grad_annotation(steps, grad)
        ax.axvline(peak_s, color=color, ls="--", lw=1.2, alpha=0.7)
        ax.scatter([peak_s], [peak_g], s=85, c=color, marker="D", edgecolors="white", linewidths=0.8, zorder=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Grad norm")
    ax.set_ylim(0, 300)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gradient norm: failed runs (diamond = peak grad norm, dashed line = that step)")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grad_success(out: Path, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(success_steps, success_grad, "o-", color=COL_SUCCESS, lw=2, ms=3, label="Full Detach (stable)")
    ax.fill_between(success_steps, success_grad, alpha=0.12, color=COL_SUCCESS)
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad norm")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right")
    ax.set_title("Gradient norm: successful full epoch")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Matplotlib versions of CODI stability dashboard charts.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {default_out_dir()})",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    out_dir = args.out_dir or default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_failed_with_explosion(out_dir / "loss_failed_runs_explosion.png", args.dpi)
    plot_loss_comparison(out_dir / "loss_all_runs.png", args.dpi)
    plot_grad_failed(out_dir / "grad_norm_failed.png", args.dpi)
    plot_grad_success(out_dir / "grad_norm_success.png", args.dpi)

    print(
        f"Wrote:\n  {out_dir / 'loss_failed_runs_explosion.png'}\n  {out_dir / 'loss_all_runs.png'}\n"
        f"  {out_dir / 'grad_norm_failed.png'}\n  {out_dir / 'grad_norm_success.png'}"
    )


if __name__ == "__main__":
    main()
