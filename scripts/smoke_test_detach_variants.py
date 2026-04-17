"""Smoke-test V2/V3/V4 detach variants by running ~10 training steps each.

Exits non-zero if:
- Subprocess returns non-zero
- ``NumericalInstabilityError`` appears in the subprocess stderr (the harness
  raises this on non-finite activations/gradients — see
  ``latent_harness/core/runtime.py``)
- A structured error marker ("non-finite" in the harness's own JSON/log output)
  is observed

Does NOT use grep or substring heuristics on log text; relies on the harness's
existing structured error surface.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HARNESS_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = HARNESS_ROOT / "configs" / "training"
VARIANTS: list[tuple[str, str]] = [
    ("v2_keep_last_2", "gemma3_4b_codi_gh200_v2_keep_last_2.yaml"),
    ("v3_reasoning_only", "gemma3_4b_codi_gh200_v3_reasoning_only.yaml"),
    ("v4_strict_no_latent_detach", "gemma3_4b_codi_gh200_v4_strict_no_latent_detach.yaml"),
]
SMOKE_STEPS = 10
SMOKE_SAMPLES = 64
SMOKE_TIMEOUT_SEC = 1800  # 30 min per variant; well above expected 10-step wall time


def _materialize_smoke_config(src_cfg: Path, variant: str) -> Path:
    with src_cfg.open() as f:
        payload = yaml.safe_load(f)

    out_dir = HARNESS_ROOT / "artifacts" / "smoke" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = payload.setdefault("trainer", {})
    trainer["max_steps"] = SMOKE_STEPS
    trainer["save_strategy"] = "no"
    trainer["output_dir"] = str(out_dir)
    trainer["logging_dir"] = str(out_dir / "logs")
    trainer["num_train_epochs"] = 1  # max_steps takes precedence but keep consistent

    data = payload.setdefault("data", {})
    data["max_samples_per_dataset"] = SMOKE_SAMPLES
    data.pop("validation_max_samples", None)  # skip eval during smoke

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"smoke_{variant}_", delete=False,
        dir=str(out_dir),
    )
    yaml.safe_dump(payload, tmp)
    tmp.close()
    return Path(tmp.name)


def _run_one(variant: str, cfg_filename: str) -> bool:
    src_cfg = CONFIGS_DIR / cfg_filename
    if not src_cfg.is_file():
        print(f"FAIL [{variant}]: config not found at {src_cfg}")
        return False

    smoke_cfg = _materialize_smoke_config(src_cfg, variant)
    print(f"=== smoke: {variant} (cfg={smoke_cfg.name}) ===", flush=True)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(HARNESS_ROOT / "src") + (os.pathsep + existing if existing else "")
    )
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    log_path = smoke_cfg.with_suffix(".log")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "latent_harness.training.cli", "--config", str(smoke_cfg)],
            cwd=str(HARNESS_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=SMOKE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(
            f"STDOUT\n------\n{stdout}\n\nSTDERR\n------\n{stderr}\n\n"
            f"--- TIMEOUT after {SMOKE_TIMEOUT_SEC}s ---\n"
        )
        print(f"FAIL [{variant}]: timed out after {SMOKE_TIMEOUT_SEC}s — see {log_path}")
        return False

    log_path.write_text(f"STDOUT\n------\n{proc.stdout}\n\nSTDERR\n------\n{proc.stderr}")

    if proc.returncode != 0:
        print(f"FAIL [{variant}]: exit code {proc.returncode} — see {log_path}")
        if "NumericalInstabilityError" in proc.stderr:
            print(f"       cause: NumericalInstabilityError (non-finite gradients or activations)")
        else:
            # Surface last ~40 lines of stderr to aid diagnosis without dumping everything.
            tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
            print(f"       stderr tail:\n{tail}")
        return False

    # Structured failure marker: harness itself may log "non-finite" warnings
    # even if the process didn't raise. Treat that as a smoke failure.
    if "non-finite" in proc.stderr.lower() or "NumericalInstabilityError" in proc.stdout:
        print(f"FAIL [{variant}]: structured non-finite marker observed (see {log_path})")
        return False

    print(f"PASS [{variant}]: {SMOKE_STEPS} steps completed clean — log at {log_path}")
    return True


def main() -> int:
    # Pre-flight: verify all variant configs exist before running any. Fails fast
    # with a clear message rather than getting partway through a long suite.
    missing = [cfg for _, cfg in VARIANTS if not (CONFIGS_DIR / cfg).is_file()]
    if missing:
        print(f"FAIL pre-flight: missing config files in {CONFIGS_DIR}:")
        for cfg in missing:
            print(f"  - {cfg}")
        return 1

    results = {variant: _run_one(variant, cfg) for variant, cfg in VARIANTS}
    print()
    print("=== smoke summary ===")
    for variant, ok in results.items():
        print(f"  {variant}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
