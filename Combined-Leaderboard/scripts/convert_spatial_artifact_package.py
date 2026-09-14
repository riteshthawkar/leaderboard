#!/usr/bin/env python3
"""Convert completed Track 3 v3 evidence into the lightweight v1 package."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from spatial_harness.artifact_package import (  # noqa: E402
    ANSWERS_SCHEMA_VERSION,
    ARCHIVE_NAME,
    PACKAGE_SCHEMA_VERSION,
    SCORE_SOURCE,
    SCORE_UNIT,
    VERIFICATION_LEVEL,
    aggregate_claimed_scores,
    canonical_json_bytes,
    raw_output_row,
    read_artifact_package,
    sha256_bytes,
    write_artifact_package,
    write_gzip_jsonl,
)


LEGACY_MEMBERS = {
    "submission.jsonl",
    "run_manifest.json",
    "leaderboard.json",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _read_json(value: bytes, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return parsed


def _read_jsonl(value: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw_line in enumerate(value.decode("utf-8-sig").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{name} line {line_number} must contain one JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} contains no rows")
    return rows


def _read_legacy_package(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)) or set(names) != LEGACY_MEMBERS:
            raise ValueError(
                "The source ZIP must be a completed v3 package containing exactly "
                "submission.jsonl, run_manifest.json, and leaderboard.json"
            )
        submission = _read_jsonl(archive.read("submission.jsonl"), "submission.jsonl")
        manifest = _read_json(archive.read("run_manifest.json"), "run_manifest.json")
        report = _read_json(archive.read("leaderboard.json"), "leaderboard.json")
    return submission, manifest, report


def _answer_rows(legacy_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    seen = set()
    for row_number, row in enumerate(legacy_rows, start=1):
        key = (str(row.get("condition") or ""), str(row.get("question_id") or ""))
        if not all(key) or key in seen:
            raise ValueError(f"Legacy submission contains an invalid or duplicate key at row {row_number}")
        seen.add(key)
        if type(row.get("correct")) is not bool:
            raise ValueError(f"Legacy submission row {row_number} has no boolean correct field")
        answer_type = str(row.get("answer_type") or "mcq")
        if answer_type not in {"mcq", "vqa"}:
            raise ValueError(f"Legacy submission row {row_number} has an invalid answer_type")
        final_answer = str(row.get("answer") or "").strip()
        if not final_answer:
            raise ValueError(f"Legacy submission row {row_number} has an empty answer")
        converted.append(
            {
                "schema_version": ANSWERS_SCHEMA_VERSION,
                "dataset": str(row.get("dataset") or "").strip(),
                "question_id": key[1],
                "evaluation_group": str(row.get("evaluation_group") or "").strip(),
                "answer_type": answer_type,
                "condition": key[0],
                "final_answer": final_answer,
                "claimed_credit": int(row["correct"]),
            }
        )
    return converted


def _assert_legacy_report_matches(
    computed_scores: dict[str, Any], legacy_report: dict[str, Any]
) -> None:
    report_rows = legacy_report.get("datasets")
    if not isinstance(report_rows, list):
        raise ValueError("Legacy leaderboard.json has no dataset rows")
    report_by_dataset = {str(row.get("dataset") or ""): row for row in report_rows}
    if set(report_by_dataset) != set(computed_scores["datasets"]):
        raise ValueError("Legacy leaderboard datasets do not match submission.jsonl")
    for dataset, conditions in computed_scores["datasets"].items():
        experiments = report_by_dataset[dataset].get("experiments") or {}
        for condition, expected in conditions.items():
            if condition.startswith("no_image_plus_"):
                mode = "no_image_plus"
                prompt_mode = condition[len("no_image_plus_") :]
            elif condition.startswith("no_image_"):
                mode = "no_image"
                prompt_mode = condition[len("no_image_") :]
            else:
                mode = "main"
                prompt_mode = condition[len("main_") :]
            received = ((experiments.get(mode) or {}).get(prompt_mode) or {})
            if received.get("correct") != expected["correct"] or received.get("total") != expected["total"]:
                raise ValueError(
                    f"Legacy leaderboard counts disagree with submission evidence for {dataset}/{condition}"
                )


def _raw_rows_from_final_answers(
    answers: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for row in answers:
        yield {
            "schema_version": "ms-vista-track3-raw-outputs/v1",
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "evaluation_group": row["evaluation_group"],
            "condition": row["condition"],
            "raw_output": row["final_answer"],
        }


def _raw_rows_from_source(
    path: Path,
    expected: dict[tuple[str, str], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                source = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is invalid JSON: {exc}") from exc
            if not isinstance(source, dict):
                raise ValueError(f"{path} line {line_number} must be one JSON object")
            row = raw_output_row(source)
            key = (row["condition"], row["question_id"])
            answer = expected.get(key)
            if answer is None:
                raise ValueError(f"Raw output source contains unexpected key {key}")
            if key in seen:
                raise ValueError(f"Raw output source repeats key {key}")
            if (
                row["dataset"] != answer["dataset"]
                or row["evaluation_group"] != answer["evaluation_group"]
            ):
                raise ValueError(f"Raw output source metadata differs for {key}")
            seen.add(key)
            yield row
    missing = set(expected) - seen
    if missing:
        raise ValueError(
            f"Raw output source is missing {len(missing)} answer rows; examples={sorted(missing)[:5]}"
        )


def _configuration_sha256(run_manifest: dict[str, Any]) -> str:
    source_config = ((run_manifest.get("artifacts") or {}).get("source_run_config") or {}).get("sha256")
    if isinstance(source_config, str) and SHA256_RE.fullmatch(source_config):
        return source_config
    return sha256_bytes(canonical_json_bytes(run_manifest))


def convert_package(
    source_package: Path,
    output_path: Path,
    raw_outputs_path: Path | None = None,
    model_name: str | None = None,
    benchmark_manifest_path: Path | None = None,
) -> Path:
    legacy_rows, run_manifest, legacy_report = _read_legacy_package(source_package)
    answers = _answer_rows(legacy_rows)
    datasets = [str(row.get("dataset") or "") for row in legacy_report.get("datasets") or []]
    conditions = [str(value) for value in legacy_report.get("conditions") or []]
    if not datasets or not conditions:
        raise ValueError("Legacy leaderboard.json does not declare datasets and conditions")
    claimed_scores = aggregate_claimed_scores(answers, datasets, conditions)
    _assert_legacy_report_matches(claimed_scores, legacy_report)

    legacy_model = run_manifest.get("model") or legacy_report.get("model") or {}
    public_name = str(model_name or legacy_model.get("name") or "").strip()
    if not public_name:
        raise ValueError("The completed package does not declare a model name")
    model = {
        "name": public_name,
        "served_name": str(legacy_model.get("served_name") or public_name),
        "revision": str(legacy_model.get("revision") or ""),
    }
    benchmark_version = str(run_manifest.get("benchmark_version") or "").strip()
    benchmark_sha256 = str(run_manifest.get("benchmark_manifest_sha256") or "").strip()
    if SHA256_RE.fullmatch(benchmark_sha256) is None:
        raise ValueError("Legacy run_manifest.json has no valid benchmark manifest SHA-256")
    if benchmark_manifest_path is not None:
        benchmark_manifest_bytes = benchmark_manifest_path.read_bytes()
        if sha256_bytes(benchmark_manifest_bytes) != benchmark_sha256:
            raise ValueError("The supplied benchmark manifest does not match the completed run")
        benchmark_manifest = _read_json(
            benchmark_manifest_bytes,
            str(benchmark_manifest_path),
        )
        benchmark_version = str(benchmark_manifest.get("benchmark_version") or "").strip()
    if not benchmark_version:
        raise ValueError(
            "Benchmark version is absent from the legacy run; provide --benchmark-manifest"
        )

    judge = run_manifest.get("judge") or {}
    expected = {(row["condition"], row["question_id"]): row for row in answers}
    with tempfile.TemporaryDirectory(prefix="track3-artifact-") as temporary_dir:
        temporary = Path(temporary_dir)
        answers_path = temporary / "answers.jsonl.gz"
        raw_path = temporary / "raw_outputs.jsonl.gz"
        answer_count = write_gzip_jsonl(answers_path, answers)
        if raw_outputs_path is None:
            raw_scope = "final_answer_only_legacy_conversion"
            raw_count = write_gzip_jsonl(raw_path, _raw_rows_from_final_answers(answers))
        else:
            raw_scope = "complete_model_response"
            raw_count = write_gzip_jsonl(
                raw_path,
                _raw_rows_from_source(raw_outputs_path, expected),
            )
        if answer_count != len(answers) or raw_count != len(answers):
            raise ValueError("Converted answer and raw-output row counts differ")

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "created_at": str(run_manifest.get("created_at") or ""),
            "verification_level": VERIFICATION_LEVEL,
            "score_source": SCORE_SOURCE,
            "model": model,
            "benchmark": {
                "version": benchmark_version,
                "manifest_sha256": benchmark_sha256,
            },
            "evaluation": {
                "harness_contract": str(run_manifest.get("harness_contract") or "legacy-track3"),
                "harness_version": str(run_manifest.get("harness_version") or "legacy"),
                "harness_commit": run_manifest.get("harness_commit"),
                "configuration_sha256": _configuration_sha256(run_manifest),
                "judge": {
                    "name": str(judge.get("model") or judge.get("served_model") or ""),
                    "revision": str(judge.get("revision") or ""),
                },
            },
            "evidence": {
                "answer_rows": answer_count,
                "raw_output_rows": raw_count,
                "raw_output_scope": raw_scope,
            },
            "scoring": {"source": SCORE_SOURCE, "unit": SCORE_UNIT},
            "source_conversion": {
                "schema_version": str(run_manifest.get("schema_version") or ""),
                "package_sha256": sha256_bytes(source_package.read_bytes()),
            },
        }
        package = write_artifact_package(
            output_path,
            manifest,
            claimed_scores,
            answers_path,
            raw_path,
        )

    validated = read_artifact_package(package)
    print(
        json.dumps(
            {
                "package": str(package),
                "sha256": sha256_bytes(package.read_bytes()),
                "model": validated.manifest["model"],
                "benchmark": validated.manifest["benchmark"],
                "answer_rows": len(validated.answer_rows),
                "raw_output_scope": validated.manifest["evidence"]["raw_output_scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, help="Completed v3 spatial_reasoning_submission.zip")
    parser.add_argument("--output", type=Path, help=f"Destination {ARCHIVE_NAME}")
    parser.add_argument("--raw-outputs", type=Path, help="Completed judged.jsonl containing original model responses")
    parser.add_argument("--benchmark-manifest", type=Path, help="Public manifest.json matching the completed run")
    parser.add_argument("--model-name", default="", help="Optional public model-name override")
    parser.add_argument("--validate", type=Path, help="Validate an existing v1 artifact package and exit")
    args = parser.parse_args()
    if bool(args.validate) == bool(args.source_package):
        parser.error("Provide exactly one of --validate or --source-package")
    if args.source_package and not args.output:
        args.output = args.source_package.with_name(ARCHIVE_NAME)
    return args


def main() -> None:
    args = parse_args()
    if args.validate:
        package = read_artifact_package(args.validate.expanduser().resolve())
        print(
            json.dumps(
                {
                    "valid": True,
                    "model": package.manifest["model"],
                    "benchmark": package.manifest["benchmark"],
                    "answer_rows": len(package.answer_rows),
                    "raw_output_scope": package.manifest["evidence"]["raw_output_scope"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    convert_package(
        args.source_package.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.raw_outputs.expanduser().resolve() if args.raw_outputs else None,
        args.model_name.strip() or None,
        args.benchmark_manifest.expanduser().resolve() if args.benchmark_manifest else None,
    )


if __name__ == "__main__":
    main()
