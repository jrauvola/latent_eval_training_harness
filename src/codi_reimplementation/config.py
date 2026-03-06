from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping at {config_path}, got {type(payload)!r}")
    payload["_config_path"] = str(config_path)
    return payload


def dump_yamlable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: dump_yamlable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: dump_yamlable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump_yamlable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_from_config(config_path: str | Path, maybe_relative: str | Path) -> Path:
    config_root = Path(config_path).expanduser().resolve().parent
    candidate = Path(maybe_relative).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_root / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
