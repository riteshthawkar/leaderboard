"""Finalize approved Track-3 context-limit failures as scored-incorrect rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatial_harness.run_track3_vllm import (
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
)
from spatial_harness.submission_contract import (
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
)


CONTEXT_ERROR_RE = re.compile(
    r"Input length \((?P<input_length>\d+)\) exceeds model's maximum "
    r"context length \((?P<max_length>\d+)\)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prediction_paths(input_dir: Path) -> list[Path]:
    return [
        input_dir / f"pred_{mode}_{prompt_mode}.jsonl"
        for mode in MODES
        for prompt_mode in PROMPT_MODES
    ]


def _sample_id(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}:{row.get('index')}"


def finalize_context_failures(
    input_dir: Path,
    report_path: Path,
    allowed_samples: set[str],
    expected_condition_rows: int,
) -> dict[str, Any]:
    run_config_path = input_dir / "run_config.json"
    run_config = _read_json(run_config_path)
    if (
        run_config.get("schema_version") != 5
        or run_config.get("harness_contract") != HARNESS_CONTRACT
        or run_config.get("modes") != list(MODES)
        or run_config.get("prompt_modes") != list(PROMPT_MODES)
    ):
        raise ValueError("Track-3 run_config.json is not the frozen v5 contract")
    server_metadata = run_config.get("server_metadata")
    if not isinstance(server_metadata, dict):
        raise ValueError("Track-3 run_config.json has no server metadata")
    configured_context = int(server_metadata.get("max_model_len") or 0)
    if configured_context <= 0:
        raise ValueError("Track-3 run_config.json has no valid max_model_len")

    paths = _prediction_paths(input_dir)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files: {missing}")

    before_hashes = {path.name: _sha256(path) for path in paths}
    before_config_hash = _sha256(run_config_path)
    rows_by_path = {path: _read_jsonl(path) for path in paths}
    blockers: list[str] = []
    finalized: list[dict[str, Any]] = []
    condition_counts: Counter[str] = Counter()
    seen_keys: set[tuple[str, str, str, str]] = set()

    for path, rows in rows_by_path.items():
        for row in rows:
            key = (
                str(row.get("dataset") or ""),
                str(row.get("index") or ""),
                str(row.get("mode") or ""),
                str(row.get("pmode") or ""),
            )
            if not all(key):
                blockers.append(f"{path.name}: invalid sample-condition key {key}")
                continue
            if key in seen_keys:
                blockers.append(f"{path.name}: duplicate sample-condition key {key}")
                continue
            seen_keys.add(key)

            output = str(row.get("output") or "").strip()
            error = str(row.get("error") or "")
            terminal = row.get("terminal_failure")
            if output and not error and terminal is None:
                continue

            match = CONTEXT_ERROR_RE.search(error)
            sample_id = _sample_id(row)
            reasons = []
            if output:
                reasons.append("failure row contains nonempty model output")
            if match is None:
                reasons.append("error is not an input-context-length exception")
            if sample_id not in allowed_samples:
                reasons.append("sample is not explicitly approved")
            if match is not None:
                input_length = int(match.group("input_length"))
                max_length = int(match.group("max_length"))
                if max_length != configured_context:
                    reasons.append(
                        f"error context {max_length} differs from configured "
                        f"context {configured_context}"
                    )
                if input_length <= max_length:
                    reasons.append("recorded input length does not exceed context")
            if reasons:
                blockers.append(f"{path.name}/{sample_id}: {', '.join(reasons)}")
                continue

            condition = f"{row['mode']}_{row['pmode']}"
            marker = {
                "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
                "policy": INFERENCE_FAILURE_POLICY,
                "category": "input_context_exceeded",
                "disposition": INFERENCE_FAILURE_DISPOSITION,
                "input_length": input_length,
                "max_model_len": max_length,
            }
            if terminal not in (None, marker):
                blockers.append(
                    f"{path.name}/{sample_id}: conflicting terminal failure metadata"
                )
                continue
            row["terminal_failure"] = marker
            row["finish_reason"] = "inference_error"
            finalized.append(
                {
                    "dataset": row["dataset"],
                    "index": str(row["index"]),
                    "mode": row["mode"],
                    "prompt_mode": row["pmode"],
                    "condition": condition,
                    "input_length": input_length,
                    "max_model_len": max_length,
                }
            )
            condition_counts[condition] += 1

    unique_samples = sorted(
        {f"{item['dataset']}:{item['index']}" for item in finalized}
    )
    if set(unique_samples) != allowed_samples:
        blockers.append(
            "finalized unique sample set differs from approved samples: "
            f"found={unique_samples}, approved={sorted(allowed_samples)}"
        )
    if len(finalized) != expected_condition_rows:
        blockers.append(
            f"found {len(finalized)} eligible condition rows; "
            f"expected {expected_condition_rows}"
        )
    if blockers:
        raise ValueError(
            "Refusing to finalize Track-3 inference failures:\n- "
            + "\n- ".join(blockers)
        )

    finalized_at = datetime.now(timezone.utc).isoformat()
    policy = {
        "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
        "policy": INFERENCE_FAILURE_POLICY,
        "disposition": INFERENCE_FAILURE_DISPOSITION,
        "eligible_category": "input_context_exceeded",
        "configured_max_model_len": configured_context,
        "unique_samples": len(unique_samples),
        "condition_rows": len(finalized),
        "sample_ids": unique_samples,
        "condition_counts": dict(sorted(condition_counts.items())),
        "finalized_at": finalized_at,
    }
    existing_policy = run_config.get("terminal_inference_failures")
    if existing_policy is not None:
        comparable = {
            key: value
            for key, value in existing_policy.items()
            if key != "finalized_at"
        }
        if comparable != {key: value for key, value in policy.items() if key != "finalized_at"}:
            raise ValueError("run_config.json contains a conflicting failure policy")
        policy["finalized_at"] = str(existing_policy.get("finalized_at") or finalized_at)
    run_config["terminal_inference_failures"] = policy

    for path, rows in rows_by_path.items():
        _atomic_jsonl(path, rows)
    _atomic_json(run_config_path, run_config)
    report = {
        **policy,
        "artifacts": {
            path.name: {
                "before_sha256": before_hashes[path.name],
                "after_sha256": _sha256(path),
                "rows": len(rows_by_path[path]),
            }
            for path in paths
        },
        "run_config": {
            "before_sha256": before_config_hash,
            "after_sha256": _sha256(run_config_path),
        },
        "failures": sorted(
            finalized,
            key=lambda item: (
                item["dataset"],
                item["index"],
                item["condition"],
            ),
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize approved Track-3 context-limit failures"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Approved dataset:index sample ID; repeat for each sample",
    )
    parser.add_argument("--expected-condition-rows", type=int, required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.report = (
        args.report.expanduser().resolve()
        if args.report
        else args.input / "inference_failure_resolution.json"
    )
    if args.expected_condition_rows < 1:
        parser.error("--expected-condition-rows must be positive")
    if any(":" not in sample for sample in args.sample):
        parser.error("--sample values must use dataset:index format")
    return args


def main() -> None:
    args = parse_args()
    report = finalize_context_failures(
        args.input,
        args.report,
        set(args.sample),
        args.expected_condition_rows,
    )
    print(
        "Finalized "
        f"{report['condition_rows']} condition rows across "
        f"{report['unique_samples']} unique samples."
    )


if __name__ == "__main__":
    main()
