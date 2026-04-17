#!/usr/bin/env python3
"""
Write SPAR_LAMBDA_PRIVATE_KEY from a .env file to a PEM file for ssh -i.

Multiline PEM cannot be sourced in bash; this is the supported path for
connect_lambda_gpu.sh and manual SSH.

Does not print the key; only prints the output path on success.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def extract_pem(text: str) -> str | None:
    marker = "SPAR_LAMBDA_PRIVATE_KEY="
    if marker not in text:
        return None
    body = text.split(marker, 1)[1].lstrip()
    for end in ("-----END RSA PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"):
        j = body.find(end)
        if j != -1:
            return body[: j + len(end)].strip() + "\n"
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="Path to .env containing SPAR_LAMBDA_PRIVATE_KEY (PEM block).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "spar_lambda_ssh_key.pem",
        help="Where to write the PEM (default: $TMPDIR/spar_lambda_ssh_key.pem).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    text = args.env_file.read_text(encoding="utf-8")
    pem = extract_pem(text)
    if not pem:
        print(
            f"No SPAR_LAMBDA_PRIVATE_KEY PEM block found in {args.env_file}",
            file=sys.stderr,
        )
        return 1
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pem, encoding="utf-8")
    os.chmod(out, 0o600)
    print(str(out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
