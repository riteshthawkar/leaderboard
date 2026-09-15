#!/usr/bin/env python3
"""Check resolved Compose resource limits without logging its secret environment."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MIB = 1024 ** 2


def memory_bytes(value: object) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([kmgt](?:i?b)?|b)?", str(value).lower())
    if not match:
        raise ValueError("Memory limits must be positive, finite byte sizes.")
    suffix = match[2] or "b"
    return int(match[1]) * 1024 ** ("bkmgt".index(suffix[0]))


def validate_capacity(compose: dict, *, ram: int, swap: int) -> dict:
    services = compose["services"]
    limits = {}
    swap_required = 0
    for name in ("api", "frontend", "caddy"):
        service = services[name]
        limit = memory_bytes(service.get("mem_limit"))
        combined = memory_bytes(service.get("memswap_limit"))
        if combined < limit:
            raise ValueError(f"{name}: memory plus swap must cover its memory limit.")
        swap_required += combined - limit
        limits[name] = limit
    reserve = max(256 * MIB, ram // 4)
    if sum(limits.values()) + reserve > ram:
        raise ValueError("Container memory limits leave insufficient RAM for the host (reserve: at least 256 MiB or 25%).")
    if swap_required > swap:
        raise ValueError("Container swap allowances exceed installed swap capacity.")
    api_env = services["api"].get("environment") or {}
    if str(api_env.get("WEB_CONCURRENCY")) != "1":
        raise ValueError("This SQLite/cache deployment requires exactly one API worker.")
    threads = int(api_env.get("GUNICORN_THREADS", 4))
    if not 1 <= threads <= 16:
        raise ValueError("GUNICORN_THREADS must be between 1 and 16.")
    if limits["api"] < 1024 * MIB and threads != 1:
        raise ValueError("API memory below 1 GiB requires one request thread.")
    go_limit = memory_bytes((services["caddy"].get("environment") or {}).get("GOMEMLIMIT"))
    if go_limit >= limits["caddy"]:
        raise ValueError("Caddy's Go memory target must be below its container limit.")
    # Compressed inputs can expand substantially; this is a guard, not a load test.
    defaults = {
        "MAX_CONTENT_LENGTH": 16 * MIB,
        "MAX_FILE_SIZE_PER_SUBMISSION": 16 * MIB,
        "MAX_SPATIAL_ARCHIVE_BYTES": 32 * MIB,
        "MAX_SPATIAL_MULTIPART_BYTES": 36 * MIB,
        "MAX_SPATIAL_SUBMISSION_BYTES": 64 * MIB,
    }
    uploads = {key: int(api_env.get(key, default)) for key, default in defaults.items()}
    if any(value <= 0 for value in uploads.values()):
        raise ValueError("Upload limits must be positive.")
    if uploads["MAX_FILE_SIZE_PER_SUBMISSION"] > uploads["MAX_CONTENT_LENGTH"]:
        raise ValueError("Visual upload limit exceeds the request body limit.")
    if uploads["MAX_SPATIAL_ARCHIVE_BYTES"] >= uploads["MAX_SPATIAL_MULTIPART_BYTES"]:
        raise ValueError("Spatial multipart limit must allow space for upload metadata.")
    largest_payload = max(uploads.values())
    if threads * largest_payload * 4 > limits["api"]:
        raise ValueError("Upload limits and concurrency exceed the API memory budget; lower limits or increase verified capacity.")
    return {
        "capacity_check": "passed",
        "container_memory_mib": {key: value // MIB for key, value in limits.items()},
        "host_reserve_mib": reserve // MIB,
        "api_threads": threads,
        "note": "Capacity guard only; representative load testing and availability review remain required.",
    }


def main() -> int:
    try:
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        result = validate_capacity(
            json.load(sys.stdin),
            ram=int(meminfo["MemTotal"].split()[0]) * 1024,
            swap=int(meminfo["SwapTotal"].split()[0]) * 1024,
        )
    except (ValueError, KeyError, TypeError, OSError):
        # Do not print malformed input values: Compose includes credentials.
        print("Capacity check failed. Review host RAM/swap, container limits, one-worker policy, threads, and upload budgets.", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
