from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from codi_reimplementation.config import ensure_dir


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return destination


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    ensure_dir(destination.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return destination


def write_markdown_summary(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    ensure_dir(destination.parent)
    lines = [
        "# Evaluation Summary",
        "",
        "| model | benchmark | mode | accuracy | examples | avg_latency_s | avg_chars |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {benchmark} | {mode} | {accuracy:.4f} | {num_examples} | {avg_latency_s:.4f} | {avg_prediction_chars:.1f} |".format(
                model=row["model"],
                benchmark=row["benchmark"],
                mode=row["mode"],
                accuracy=row["accuracy"],
                num_examples=row["num_examples"],
                avg_latency_s=row["avg_latency_s"],
                avg_prediction_chars=row["avg_prediction_chars"],
            )
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
