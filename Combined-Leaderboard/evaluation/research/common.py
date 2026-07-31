"""Shared, dependency-light helpers for generated research experiment bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "ms-vista-research-experiment-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "type": "file",
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    if resolved.is_dir():
        digest = hashlib.sha256()
        total_bytes = 0
        file_count = 0
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            relative = child.relative_to(resolved).as_posix()
            child_hash = sha256_file(child)
            child_size = child.stat().st_size
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(child_hash.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(child_size).encode("ascii"))
            digest.update(b"\n")
            total_bytes += child_size
            file_count += 1
        return {
            "path": str(resolved),
            "type": "directory",
            "sha256": digest.hexdigest(),
            "bytes": total_bytes,
            "file_count": file_count,
        }
    raise FileNotFoundError(f"Manifest path does not exist: {resolved}")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    project_root: Path,
    experiment: str,
    parameters: dict[str, Any],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    item_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "parameters": parameters,
        "item_count": item_count,
        "inputs": {
            name: describe_path(path) for name, path in sorted(inputs.items())
        },
        "outputs": {
            name: describe_path(path) for name, path in sorted(outputs.items())
        },
    }
