"""Load one row from every registry entry to verify Hub access and config names."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from datasets import load_dataset

from latent_harness.data.dataset_registry import default_registry_path, load_dataset_registry


def _try_load(path: str, subset: str | None, smoke_split: str) -> dict[str, Any]:
    """Prefer streaming so huge Hub datasets do not materialize full splits."""
    cache = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_HOME")
    base_kw: dict[str, Any] = {"split": smoke_split}
    if cache:
        base_kw["cache_dir"] = cache

    def _call(streaming: bool) -> Any:
        kw = {**base_kw, "streaming": streaming}
        if subset is None:
            return load_dataset(path, **kw)
        return load_dataset(path, subset, **kw)

    try:
        ds = _call(streaming=True)
        return next(iter(ds))
    except Exception:
        ds = _call(streaming=False)
        return ds[0]


def _attempt_entry(entry: dict[str, Any]) -> tuple[bool, str, str]:
    entry_id = entry.get("id", "?")
    attempts: list[tuple[str, str | None, str]] = [
        (entry["path"], entry.get("subset"), entry["smoke_split"]),
    ]
    for fb in entry.get("fallbacks") or []:
        attempts.append((fb["path"], fb.get("subset"), fb["smoke_split"]))

    last_err = ""
    for path, subset, split in attempts:
        try:
            row = _try_load(path, subset, split)
            keys = ", ".join(sorted(row.keys())[:12])
            if len(row.keys()) > 12:
                keys += ", …"
            used = f"{path}" + (f" (subset={subset})" if subset is not None else "")
            return True, used, keys
        except Exception as exc:  # noqa: BLE001 — surface all load failures
            last_err = f"{exc}"
    return False, entry.get("path", ""), last_err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Hugging Face datasets from full_registry.yaml")
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to full_registry.yaml (default: harness configs/datasets/full_registry.yaml)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only run entries with this category (e.g. eval_final)",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include entries marked heavy_smoke (large multi-shard downloads, e.g. OpenThoughts3, FineWeb-Edu).",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Include Hub gated datasets (e.g. yale-nlp/FOLIO) after you accepted access with this account.",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print(
            "Warning: HF_TOKEN / HUGGINGFACE_HUB_TOKEN not set — gated datasets may fail. "
            "Use project .env or `export HF_TOKEN=...`.",
            file=sys.stderr,
        )

    data = load_dataset_registry(args.registry)
    entries = data.get("entries") or []
    ok = 0
    fail = 0
    skipped = 0
    for entry in entries:
        if args.category and entry.get("category") != args.category:
            continue
        if entry.get("heavy_smoke") and not args.include_heavy:
            skipped += 1
            eid = entry.get("id", "?")
            cat = entry.get("category", "?")
            print(f"SKIP [{cat}] {eid}  (heavy_smoke; use --include-heavy)")
            continue
        if entry.get("gated") and not args.include_gated:
            skipped += 1
            eid = entry.get("id", "?")
            cat = entry.get("category", "?")
            print(f"SKIP [{cat}] {eid}  (gated; use --include-gated after Hub access)")
            continue
        success, detail, info = _attempt_entry(entry)
        eid = entry.get("id", "?")
        cat = entry.get("category", "?")
        if success:
            ok += 1
            print(f"OK   [{cat}] {eid}  ->  {detail}  keys: {info}")
        else:
            fail += 1
            print(f"FAIL [{cat}] {eid}  path={detail}  error={info}")

    print(f"\nSummary: {ok} ok, {fail} failed, {skipped} skipped (total tried: {ok + fail})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
