#!/usr/bin/env python3
"""Read dgrad_per_layer.csv, emit summary JSON + heatmap PNG.

Usage: python scripts/analyze_dgrad_probe.py <probe_dir>
Inputs: <probe_dir>/dgrad_per_layer.csv
Outputs:
  <probe_dir>/dgrad_heatmap.png (steps x layers, log10 scale)
  <probe_dir>/dgrad_summary.json (per-layer percentiles across all steps)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    args = parser.parse_args()

    csv_path = args.probe_dir / "dgrad_per_layer.csv"
    df = pd.read_csv(csv_path)
    df["layer_idx"] = df["layer"].str.extract(r"layer_(\d+)").astype(int)
    pivot = df.pivot_table(index="step", columns="layer_idx", values="max_abs_dgrad")

    # Heatmap (steps on x, layers on y; log10 because values span orders of magnitude)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = np.log10(pivot.values + 1e-30)
    im = ax.imshow(data.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("step")
    ax.set_ylabel("layer index")
    ax.set_title(f"log10 max|dgrad| per layer  ({args.probe_dir.name})")
    fig.colorbar(im, ax=ax, label="log10 max|dgrad|")
    out_png = args.probe_dir / "dgrad_heatmap.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Summary JSON: per-layer p99 and median across all recorded steps
    summary = {
        "num_steps": int(len(pivot)),
        "num_layers": int(pivot.shape[1]),
        "per_layer_percentile_99": {
            int(col): float(np.nanpercentile(pivot[col].values, 99))
            for col in pivot.columns
        },
        "per_layer_median": {
            int(col): float(np.nanmedian(pivot[col].values))
            for col in pivot.columns
        },
    }
    out_json = args.probe_dir / "dgrad_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
