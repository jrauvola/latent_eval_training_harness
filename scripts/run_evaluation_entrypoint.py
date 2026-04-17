from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")

# Avoid shadowing third-party packages from repo-local module names.
blocked_paths = {
    "",
    str(ROOT),
    str(ROOT / "src" / "latent_harness" / "evaluation"),
}
sys.path = [SRC] + [path for path in sys.path if path not in blocked_paths]

from latent_harness.evaluation.cli import main


if __name__ == "__main__":
    main()
