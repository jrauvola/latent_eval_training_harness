from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import yaml


class DatasetFallback(TypedDict, total=False):
    path: str
    subset: str | None
    smoke_split: str


class DatasetEntry(TypedDict, total=False):
    id: str
    category: str
    path: str
    subset: str | None
    smoke_split: str
    fallbacks: list[DatasetFallback]


def default_registry_path() -> Path:
    """Path to ``configs/datasets/full_registry.yaml`` under the harness package root."""
    return Path(__file__).resolve().parents[3] / "configs" / "datasets" / "full_registry.yaml"


def load_dataset_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_file = Path(path) if path is not None else default_registry_path()
    with registry_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
