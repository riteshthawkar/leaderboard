#!/usr/bin/env python3
"""Atomically refresh the static frontend's frozen API responses."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "frontend" / "src" / "data" / "snapshot"
ENDPOINTS = {
    "statistics-overview.json": "/api/statistics/overview",
    "leaderboard-visual-cognition.json": "/api/leaderboard/visual-cognition",
    "leaderboard-spatial.json": "/api/leaderboard/spatial",
    "task-do_you_see_me.json": "/api/tasks/do_you_see_me/info",
    "task-minds_eye.json": "/api/tasks/minds_eye/info",
    "task-spatial.json": "/api/tasks/spatial/info",
    "auth-providers.json": "/api/auth/providers",
}
MODEL_REPORTS_FILENAME = "model-reports.json"
ENCODE_COMPONENT_SAFE = "-_.!~*'()"


def _fetch_json(api_base: str, endpoint: str, timeout: float):
    request = Request(
        f"{api_base.rstrip('/')}{endpoint}",
        headers={"Accept": "application/json", "User-Agent": "ms-vista-snapshot-refresh/1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if response.status != 200:
                raise RuntimeError(f"{endpoint} returned HTTP {response.status}")
            if content_type != "application/json":
                raise RuntimeError(
                    f"{endpoint} returned {content_type}, expected application/json"
                )
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"{endpoint} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"{endpoint} could not be reached: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"{endpoint} returned invalid JSON") from exc
    if not isinstance(payload, (dict, list)):
        raise RuntimeError(f"{endpoint} returned an unsupported JSON value")
    return payload


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_report_endpoints(leaderboard_payload) -> list[str]:
    if not isinstance(leaderboard_payload, dict):
        raise RuntimeError("The visual leaderboard snapshot must be a JSON object")
    rows = leaderboard_payload.get("leaderboard")
    if not isinstance(rows, list):
        raise RuntimeError("The visual leaderboard snapshot has no leaderboard array")

    endpoints = []
    seen_names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Leaderboard row {index} is not a JSON object")
        model_name = str(row.get("model_name") or "").strip()
        if not model_name:
            raise RuntimeError(f"Leaderboard row {index} has no model_name")
        if model_name in seen_names:
            raise RuntimeError(f"The visual leaderboard repeats model '{model_name}'")
        seen_names.add(model_name)
        encoded_name = quote(model_name, safe=ENCODE_COMPONENT_SAFE)
        endpoints.append(f"/api/model/{encoded_name}/report")
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default=os.getenv("API_BASE_URL", "http://localhost:5050"),
        help="Running leaderboard API origin (default: API_BASE_URL or localhost:5050).",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    snapshots = {
        filename: _fetch_json(args.api_base, endpoint, args.timeout)
        for filename, endpoint in ENDPOINTS.items()
    }
    leaderboard = snapshots["leaderboard-visual-cognition.json"]
    snapshots[MODEL_REPORTS_FILENAME] = {
        endpoint: _fetch_json(args.api_base, endpoint, args.timeout)
        for endpoint in _model_report_endpoints(leaderboard)
    }
    for filename, payload in snapshots.items():
        _atomic_write_json(SNAPSHOT_DIR / filename, payload)
        print(f"refreshed {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
