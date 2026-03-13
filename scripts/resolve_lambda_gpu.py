#!/usr/bin/env python3
"""Resolve a Lambda Cloud GPU instance host via the Cloud API."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = os.getenv("LAMBDA_CLOUD_BASE_URL", "https://cloud.lambdalabs.com")
TOKEN_ENV_VARS = (
    "LAMBDA_CLOUD_API_TOKEN",
    "LAMBDA_CLOUD_TOKEN",
    "LAMBDA_API_TOKEN",
)
DEFAULT_STATUSES = ("active",)


@dataclass(slots=True)
class InstanceRecord:
    instance_id: str
    name: str
    status: str
    ip: str
    instance_type: str
    region: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve or list Lambda Cloud instances for SSH helpers."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Lambda Cloud API base URL.")
    parser.add_argument("--token", help="Cloud API token. Defaults to env lookup.")
    parser.add_argument("--instance-id", default=os.getenv("LAMBDA_GPU_INSTANCE_ID"))
    parser.add_argument("--name", default=os.getenv("LAMBDA_GPU_NAME"))
    parser.add_argument("--name-contains", default=os.getenv("LAMBDA_GPU_NAME_CONTAINS"))
    parser.add_argument("--instance-type", default=os.getenv("LAMBDA_GPU_INSTANCE_TYPE"))
    parser.add_argument("--region", default=os.getenv("LAMBDA_GPU_REGION"))
    parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        help="Allowed status. Repeat to allow multiple values. Defaults to active.",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Do not filter by status.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching instances instead of printing the selected host.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit matching instances as JSON.",
    )
    parser.add_argument(
        "--instances-json",
        help="Read the API payload from a local JSON file instead of calling Lambda.",
    )
    return parser.parse_args()


def resolve_token(explicit_token: str | None) -> str:
    if explicit_token:
        return explicit_token
    for env_name in TOKEN_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    raise SystemExit(
        "No Lambda Cloud API token found. Set one of "
        f"{', '.join(TOKEN_ENV_VARS)} or provide --token."
    )


def load_instances_from_file(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    data = payload.get("data")
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON object with a top-level 'data' list in {path}.")
    return data


def fetch_instances(base_url: str, token: str) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/instances"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "latent-harness-lambda-helper/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        message = body or exc.reason
        raise SystemExit(f"Lambda API request failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Lambda API request failed: {exc.reason}") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise SystemExit("Lambda API response did not contain a top-level 'data' list.")
    return data


def normalize_instance(payload: dict[str, Any]) -> InstanceRecord:
    region = payload.get("region") or {}
    instance_type = payload.get("instance_type") or {}
    return InstanceRecord(
        instance_id=str(payload.get("id", "")),
        name=str(payload.get("name") or ""),
        status=str(payload.get("status") or ""),
        ip=str(payload.get("ip") or ""),
        instance_type=str(instance_type.get("name") or ""),
        region=str(region.get("name") or ""),
    )


def filter_instances(
    instances: list[InstanceRecord],
    *,
    instance_id: str | None,
    name: str | None,
    name_contains: str | None,
    instance_type: str | None,
    region: str | None,
    statuses: tuple[str, ...] | None,
) -> list[InstanceRecord]:
    filtered = instances
    if instance_id:
        filtered = [item for item in filtered if item.instance_id == instance_id]
    if name:
        filtered = [item for item in filtered if item.name == name]
    if name_contains:
        needle = name_contains.lower()
        filtered = [item for item in filtered if needle in item.name.lower()]
    if instance_type:
        filtered = [item for item in filtered if item.instance_type == instance_type]
    if region:
        filtered = [item for item in filtered if item.region == region]
    if statuses is not None:
        allowed = {status.lower() for status in statuses}
        filtered = [item for item in filtered if item.status.lower() in allowed]
    return sorted(
        filtered,
        key=lambda item: (
            item.status != "active",
            item.name or "~",
            item.instance_type or "~",
            item.instance_id,
        ),
    )


def format_instances(instances: list[InstanceRecord]) -> str:
    if not instances:
        return "No matching instances."

    headers = ("id", "name", "status", "type", "region", "ip")
    rows = [
        (
            item.instance_id,
            item.name or "-",
            item.status or "-",
            item.instance_type or "-",
            item.region or "-",
            item.ip or "-",
        )
        for item in instances
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def render_row(values: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))

    lines = [render_row(headers), render_row(tuple("-" * width for width in widths))]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def choose_instance(instances: list[InstanceRecord]) -> InstanceRecord:
    if not instances:
        raise SystemExit(
            "No matching Lambda instances found. Run with --list or relax the current filters."
        )
    with_ip = [item for item in instances if item.ip]
    if not with_ip:
        raise SystemExit(
            "Found matching instances but none have a public IP yet. "
            "Try again once the instance finishes booting or run with --list."
        )
    if len(with_ip) == 1:
        return with_ip[0]

    table = format_instances(with_ip)
    raise SystemExit(
        "Multiple matching Lambda instances have public IPs. Narrow the selection with "
        "--instance-id, --name, --name-contains, --instance-type, or --region.\n\n"
        f"{table}"
    )


def main() -> int:
    args = parse_args()
    statuses = None if args.all_statuses else tuple(args.statuses or DEFAULT_STATUSES)

    if args.instances_json:
        raw_instances = load_instances_from_file(args.instances_json)
    else:
        token = resolve_token(args.token)
        raw_instances = fetch_instances(args.base_url, token)

    instances = [normalize_instance(item) for item in raw_instances]
    filtered = filter_instances(
        instances,
        instance_id=args.instance_id,
        name=args.name,
        name_contains=args.name_contains,
        instance_type=args.instance_type,
        region=args.region,
        statuses=statuses,
    )

    if args.json:
        print(json.dumps([asdict(item) for item in filtered], indent=2))
        return 0

    if args.list:
        print(format_instances(filtered))
        return 0 if filtered else 1

    selected = choose_instance(filtered)
    print(selected.ip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
