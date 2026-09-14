"""Lightweight, artifact-backed Track 3 submission packages.

This contract intentionally does not grade answers. It validates package
integrity, public sample coverage, and arithmetic consistency between the
submitter's per-sample credit flags and claimed aggregate counts.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator


ARCHIVE_NAME = "track3_artifact_submission.zip"
PACKAGE_SCHEMA_VERSION = "ms-vista-track3-artifact-package/v1"
ANSWERS_SCHEMA_VERSION = "ms-vista-track3-claimed-answers/v1"
RAW_OUTPUTS_SCHEMA_VERSION = "ms-vista-track3-raw-outputs/v1"
SCORES_SCHEMA_VERSION = "ms-vista-track3-claimed-scores/v1"
CHECKSUMS_SCHEMA_VERSION = "ms-vista-track3-checksums/v1"
VERIFICATION_LEVEL = "self_reported_artifact_backed"
SCORE_SOURCE = "submitter_claimed"
SCORE_UNIT = "evaluation_group_all_rows_correct"

MANIFEST_MEMBER = "manifest.json"
SCORES_MEMBER = "claimed_scores.json"
ANSWERS_MEMBER = "answers.jsonl.gz"
RAW_OUTPUTS_MEMBER = "raw_outputs.jsonl.gz"
CHECKSUMS_MEMBER = "checksums.json"
ARCHIVE_MEMBERS = (
    MANIFEST_MEMBER,
    SCORES_MEMBER,
    ANSWERS_MEMBER,
    RAW_OUTPUTS_MEMBER,
    CHECKSUMS_MEMBER,
)
CHECKSUMMED_MEMBERS = tuple(
    member for member in ARCHIVE_MEMBERS if member != CHECKSUMS_MEMBER
)

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = {
    MANIFEST_MEMBER: 1024 * 1024,
    SCORES_MEMBER: 4 * 1024 * 1024,
    ANSWERS_MEMBER: 96 * 1024 * 1024,
    RAW_OUTPUTS_MEMBER: 384 * 1024 * 1024,
    CHECKSUMS_MEMBER: 1024 * 1024,
}
MAX_ANSWERS_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ANSWER_LINE_BYTES = 128 * 1024
MAX_ANSWER_ROWS = 1_000_000
MAX_COMPRESSION_RATIO = 250
SHA256_RE = re.compile(r"[0-9a-f]{64}")

ANSWER_FIELDS = {
    "schema_version",
    "dataset",
    "question_id",
    "evaluation_group",
    "answer_type",
    "condition",
    "final_answer",
    "claimed_credit",
}
class ArtifactPackageError(ValueError):
    """A stable validation failure suitable for CLI and API translation."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class ArtifactPackage:
    """Validated package members and arithmetic derived from compact answers."""

    members: dict[str, bytes]
    manifest: dict[str, Any]
    claimed_scores: dict[str, Any]
    answer_rows: list[dict[str, Any]]
    computed_scores: dict[str, Any]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(value: bytes, member: str) -> Any:
    try:
        text = value.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ArtifactPackageError(
            "invalid_artifact_encoding",
            f"{member} must be UTF-8 text.",
            member=member,
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPackageError(
            "invalid_artifact_json",
            f"{member} is not valid strict JSON: {exc}.",
            member=member,
        ) from exc


def _require_object(value: Any, member: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactPackageError(
            "invalid_artifact_shape",
            f"{member} must contain one JSON object.",
            member=member,
        )
    return value


def _valid_nonempty_text(value: Any, *, maximum: int = 4096) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ArtifactPackageError(
            "unsupported_artifact_package_version",
            f"manifest.json must use {PACKAGE_SCHEMA_VERSION}.",
        )
    if manifest.get("verification_level") != VERIFICATION_LEVEL:
        raise ArtifactPackageError(
            "invalid_artifact_verification_level",
            "manifest.json must identify the result as self-reported and artifact-backed.",
        )
    if manifest.get("score_source") != SCORE_SOURCE:
        raise ArtifactPackageError(
            "invalid_artifact_score_source",
            "manifest.json must identify scores as submitter-claimed.",
        )
    model = manifest.get("model")
    if not isinstance(model, dict) or not _valid_nonempty_text(model.get("name"), maximum=255):
        raise ArtifactPackageError(
            "invalid_artifact_model",
            "manifest.json must declare a non-empty model name.",
        )
    revision = model.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ArtifactPackageError(
            "invalid_artifact_model",
            "manifest.json model revision must be a string.",
        )
    benchmark = manifest.get("benchmark")
    if (
        not isinstance(benchmark, dict)
        or not _valid_nonempty_text(benchmark.get("version"), maximum=255)
        or not isinstance(benchmark.get("manifest_sha256"), str)
        or SHA256_RE.fullmatch(benchmark["manifest_sha256"]) is None
    ):
        raise ArtifactPackageError(
            "invalid_artifact_benchmark",
            "manifest.json must bind a benchmark version and lowercase SHA-256 manifest digest.",
        )
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ArtifactPackageError(
            "invalid_artifact_evaluation",
            "manifest.json must declare the evaluation harness and judge provenance.",
        )
    for field in ("harness_contract", "harness_version", "configuration_sha256"):
        value = evaluation.get(field)
        if field == "configuration_sha256":
            valid = isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
        else:
            valid = _valid_nonempty_text(value, maximum=255)
        if not valid:
            raise ArtifactPackageError(
                "invalid_artifact_evaluation",
                f"manifest.json evaluation.{field} is missing or invalid.",
            )
    judge = evaluation.get("judge")
    if not isinstance(judge, dict):
        raise ArtifactPackageError(
            "invalid_artifact_evaluation",
            "manifest.json evaluation.judge must be an object.",
        )
    for field in ("name", "revision"):
        value = judge.get(field)
        if value is not None and not isinstance(value, str):
            raise ArtifactPackageError(
                "invalid_artifact_evaluation",
                f"manifest.json evaluation.judge.{field} must be a string or null.",
            )
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        raise ArtifactPackageError(
            "invalid_artifact_evidence",
            "manifest.json must describe the submitted evidence.",
        )
    for field in ("answer_rows", "raw_output_rows"):
        value = evidence.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArtifactPackageError(
                "invalid_artifact_evidence",
                f"manifest.json evidence.{field} must be a positive integer.",
            )
    if evidence.get("raw_output_scope") not in {
        "complete_model_response",
        "final_answer_only_legacy_conversion",
    }:
        raise ArtifactPackageError(
            "invalid_artifact_evidence",
            "manifest.json declares an unsupported raw output scope.",
        )
    scoring = manifest.get("scoring")
    if not isinstance(scoring, dict) or scoring != {
        "source": SCORE_SOURCE,
        "unit": SCORE_UNIT,
    }:
        raise ArtifactPackageError(
            "invalid_artifact_scoring",
            "manifest.json scoring policy is missing or unsupported.",
        )


def _validate_checksums(
    checksums: dict[str, Any], members: dict[str, bytes]
) -> None:
    if checksums.get("schema_version") != CHECKSUMS_SCHEMA_VERSION:
        raise ArtifactPackageError(
            "unsupported_artifact_checksums_version",
            f"checksums.json must use {CHECKSUMS_SCHEMA_VERSION}.",
        )
    if checksums.get("algorithm") != "sha256":
        raise ArtifactPackageError(
            "unsupported_artifact_checksum_algorithm",
            "checksums.json must use SHA-256.",
        )
    files = checksums.get("files")
    if not isinstance(files, dict) or set(files) != set(CHECKSUMMED_MEMBERS):
        raise ArtifactPackageError(
            "invalid_artifact_checksums",
            "checksums.json must describe every package member except itself.",
        )
    for member in CHECKSUMMED_MEMBERS:
        metadata = files.get(member)
        content = members[member]
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"sha256", "size_bytes"}
            or metadata.get("sha256") != sha256_bytes(content)
            or metadata.get("size_bytes") != len(content)
        ):
            raise ArtifactPackageError(
                "artifact_checksum_mismatch",
                f"{member} does not match checksums.json.",
                member=member,
            )


def _open_zip(source: bytes | bytearray | str | Path | BinaryIO) -> zipfile.ZipFile:
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ArtifactPackageError(
                "empty_artifact_archive", "The Track 3 artifact package is empty."
            )
        if len(source) > MAX_ARCHIVE_BYTES:
            raise ArtifactPackageError(
                "artifact_archive_too_large",
                f"The Track 3 artifact package exceeds {MAX_ARCHIVE_BYTES} bytes.",
            )
        return zipfile.ZipFile(io.BytesIO(bytes(source)), "r")
    if isinstance(source, (str, Path)):
        path = Path(source)
        size = path.stat().st_size
        if size <= 0 or size > MAX_ARCHIVE_BYTES:
            raise ArtifactPackageError(
                "artifact_archive_too_large",
                f"The Track 3 artifact package must be between 1 and {MAX_ARCHIVE_BYTES} bytes.",
            )
        return zipfile.ZipFile(path, "r")
    return zipfile.ZipFile(source, "r")


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    limit = MAX_MEMBER_BYTES[info.filename]
    if info.file_size <= 0 or info.file_size > limit:
        raise ArtifactPackageError(
            "artifact_member_size_invalid",
            f"{info.filename} is empty or exceeds its {limit}-byte limit.",
            member=info.filename,
        )
    if info.flag_bits & 0x1:
        raise ArtifactPackageError(
            "encrypted_artifact_archive",
            "Encrypted Track 3 packages are not supported.",
            member=info.filename,
        )
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ArtifactPackageError(
            "unsupported_artifact_compression",
            "The Track 3 package uses an unsupported ZIP compression method.",
            member=info.filename,
        )
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.is_dir() or stat.S_ISLNK(mode) or "/" in info.filename or "\\" in info.filename:
        raise ArtifactPackageError(
            "unsafe_artifact_member",
            "The Track 3 package may contain only regular files at its root.",
            member=info.filename,
        )
    compressed = max(int(info.compress_size), 1)
    if info.file_size / compressed > MAX_COMPRESSION_RATIO:
        raise ArtifactPackageError(
            "unsafe_artifact_compression_ratio",
            f"{info.filename} exceeds the allowed compression ratio.",
            member=info.filename,
        )
    with archive.open(info, "r") as handle:
        content = handle.read(limit + 1)
    if len(content) != info.file_size or len(content) > limit:
        raise ArtifactPackageError(
            "artifact_member_size_mismatch",
            f"{info.filename} did not match its declared ZIP size.",
            member=info.filename,
        )
    return content


def _iter_gzip_jsonl(value: bytes, member: str) -> Iterator[tuple[int, dict[str, Any]]]:
    if not value.startswith(b"\x1f\x8b"):
        raise ArtifactPackageError(
            "invalid_artifact_gzip",
            f"{member} is not a gzip stream.",
            member=member,
        )
    total_bytes = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(value), mode="rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_ANSWERS_UNCOMPRESSED_BYTES:
                    raise ArtifactPackageError(
                        "artifact_answers_too_large",
                        f"{member} exceeds the uncompressed safety limit.",
                        member=member,
                    )
                if len(raw_line) > MAX_ANSWER_LINE_BYTES:
                    raise ArtifactPackageError(
                        "artifact_answer_line_too_large",
                        f"{member} line {line_number} exceeds the line-size limit.",
                        member=member,
                        line_number=line_number,
                    )
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(
                        raw_line.decode("utf-8"),
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ArtifactPackageError(
                        "invalid_artifact_answer_json",
                        f"{member} line {line_number} is not valid strict UTF-8 JSON: {exc}.",
                        member=member,
                        line_number=line_number,
                    ) from exc
                if not isinstance(row, dict):
                    raise ArtifactPackageError(
                        "invalid_artifact_answer_row",
                        f"{member} line {line_number} must contain one JSON object.",
                        member=member,
                        line_number=line_number,
                    )
                yield line_number, row
    except (OSError, EOFError) as exc:
        raise ArtifactPackageError(
            "invalid_artifact_gzip",
            f"{member} could not be decompressed: {exc}.",
            member=member,
        ) from exc


def parse_answer_rows(value: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in _iter_gzip_jsonl(value, ANSWERS_MEMBER):
        if len(rows) >= MAX_ANSWER_ROWS:
            raise ArtifactPackageError(
                "too_many_artifact_answers",
                f"{ANSWERS_MEMBER} exceeds {MAX_ANSWER_ROWS} rows.",
            )
        missing = sorted(ANSWER_FIELDS - set(row))
        unexpected = sorted(set(row) - ANSWER_FIELDS)
        if missing or unexpected:
            raise ArtifactPackageError(
                "invalid_artifact_answer_fields",
                f"{ANSWERS_MEMBER} line {line_number} does not match the v1 schema.",
                line_number=line_number,
                missing_fields=missing,
                unexpected_fields=unexpected,
            )
        if row.get("schema_version") != ANSWERS_SCHEMA_VERSION:
            raise ArtifactPackageError(
                "unsupported_artifact_answers_version",
                f"{ANSWERS_MEMBER} line {line_number} uses an unsupported schema.",
                line_number=line_number,
            )
        for field in ("dataset", "question_id", "evaluation_group", "condition"):
            if not _valid_nonempty_text(row.get(field), maximum=255):
                raise ArtifactPackageError(
                    "invalid_artifact_answer_value",
                    f"{ANSWERS_MEMBER} line {line_number} has an invalid {field}.",
                    line_number=line_number,
                    field=field,
                )
        if row.get("answer_type") not in {"mcq", "vqa"}:
            raise ArtifactPackageError(
                "invalid_artifact_answer_type",
                f"{ANSWERS_MEMBER} line {line_number} has an invalid answer_type.",
                line_number=line_number,
            )
        answer = row.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ArtifactPackageError(
                "invalid_artifact_final_answer",
                f"{ANSWERS_MEMBER} line {line_number} must contain a non-empty final_answer.",
                line_number=line_number,
            )
        if row.get("claimed_credit") not in {0, 1} or isinstance(row.get("claimed_credit"), bool):
            raise ArtifactPackageError(
                "invalid_artifact_claimed_credit",
                f"{ANSWERS_MEMBER} line {line_number} claimed_credit must be integer 0 or 1.",
                line_number=line_number,
            )
        key = (row["condition"], row["question_id"])
        if key in seen:
            raise ArtifactPackageError(
                "duplicate_artifact_answer",
                f"{ANSWERS_MEMBER} repeats {key[0]}/{key[1]}.",
                line_number=line_number,
            )
        seen.add(key)
        rows.append({**row, "line_number": line_number, "row_index": len(rows) + 1})
    if not rows:
        raise ArtifactPackageError(
            "empty_artifact_answers", f"{ANSWERS_MEMBER} contains no answers."
        )
    return rows


def aggregate_claimed_scores(
    rows: Iterable[dict[str, Any]],
    datasets: Iterable[str],
    conditions: Iterable[str],
) -> dict[str, Any]:
    dataset_order = list(datasets)
    condition_order = list(conditions)
    group_states: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        key = (row["dataset"], row["condition"], row["evaluation_group"])
        group_states[key] = group_states.get(key, True) and row["claimed_credit"] == 1
    aggregate = {
        dataset: {condition: [0, 0] for condition in condition_order}
        for dataset in dataset_order
    }
    for (dataset, condition, _group), credited in group_states.items():
        if dataset not in aggregate or condition not in aggregate[dataset]:
            raise ArtifactPackageError(
                "artifact_score_dimension_mismatch",
                f"Answers contain undeclared dataset/condition {dataset}/{condition}.",
            )
        aggregate[dataset][condition][0] += int(credited)
        aggregate[dataset][condition][1] += 1
    return {
        "schema_version": SCORES_SCHEMA_VERSION,
        "score_source": SCORE_SOURCE,
        "score_unit": SCORE_UNIT,
        "conditions": condition_order,
        "datasets": {
            dataset: {
                condition: {
                    "correct": aggregate[dataset][condition][0],
                    "total": aggregate[dataset][condition][1],
                }
                for condition in condition_order
            }
            for dataset in dataset_order
        },
    }


def _validate_claimed_scores_shape(scores: dict[str, Any]) -> tuple[list[str], list[str]]:
    if scores.get("schema_version") != SCORES_SCHEMA_VERSION:
        raise ArtifactPackageError(
            "unsupported_artifact_scores_version",
            f"claimed_scores.json must use {SCORES_SCHEMA_VERSION}.",
        )
    if scores.get("score_source") != SCORE_SOURCE or scores.get("score_unit") != SCORE_UNIT:
        raise ArtifactPackageError(
            "invalid_artifact_score_policy",
            "claimed_scores.json uses an unsupported score source or unit.",
        )
    conditions = scores.get("conditions")
    datasets = scores.get("datasets")
    if (
        not isinstance(conditions, list)
        or not conditions
        or len(set(conditions)) != len(conditions)
        or any(not _valid_nonempty_text(value, maximum=64) for value in conditions)
        or not isinstance(datasets, dict)
        or not datasets
    ):
        raise ArtifactPackageError(
            "invalid_artifact_scores_shape",
            "claimed_scores.json must declare unique conditions and at least one dataset.",
        )
    for dataset, condition_results in datasets.items():
        if not _valid_nonempty_text(dataset, maximum=255) or not isinstance(condition_results, dict):
            raise ArtifactPackageError(
                "invalid_artifact_scores_shape",
                "claimed_scores.json contains invalid dataset results.",
            )
        if set(condition_results) != set(conditions):
            raise ArtifactPackageError(
                "invalid_artifact_scores_shape",
                f"claimed_scores.json has incomplete conditions for {dataset}.",
            )
        for condition, result in condition_results.items():
            if (
                not isinstance(result, dict)
                or set(result) != {"correct", "total"}
                or not isinstance(result.get("correct"), int)
                or isinstance(result.get("correct"), bool)
                or not isinstance(result.get("total"), int)
                or isinstance(result.get("total"), bool)
                or result["total"] <= 0
                or result["correct"] < 0
                or result["correct"] > result["total"]
            ):
                raise ArtifactPackageError(
                    "invalid_artifact_score_count",
                    f"claimed_scores.json has invalid counts for {dataset}/{condition}.",
                )
    return list(datasets), conditions


def validate_claimed_scores(
    claimed: dict[str, Any], computed: dict[str, Any]
) -> None:
    if canonical_json_bytes(claimed) != canonical_json_bytes(computed):
        raise ArtifactPackageError(
            "artifact_score_arithmetic_mismatch",
            "claimed_scores.json does not match the submitted per-sample claimed_credit values.",
        )


def read_artifact_package(
    source: bytes | bytearray | str | Path | BinaryIO,
) -> ArtifactPackage:
    """Read and validate a package without reference answers or model judging."""
    try:
        with _open_zip(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactPackageError(
                    "duplicate_artifact_members",
                    "The Track 3 package contains duplicate member names.",
                )
            if set(names) != set(ARCHIVE_MEMBERS):
                raise ArtifactPackageError(
                    "invalid_artifact_archive_contents",
                    "The Track 3 package must contain exactly the five v1 artifact members.",
                    expected=list(ARCHIVE_MEMBERS),
                    received=sorted(names),
                )
            members = {info.filename: _read_member(archive, info) for info in infos}
    except ArtifactPackageError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArtifactPackageError(
            "invalid_artifact_archive",
            f"The Track 3 artifact package is not a readable ZIP file: {exc}.",
        ) from exc

    manifest = _require_object(
        strict_json_loads(members[MANIFEST_MEMBER], MANIFEST_MEMBER),
        MANIFEST_MEMBER,
    )
    claimed_scores = _require_object(
        strict_json_loads(members[SCORES_MEMBER], SCORES_MEMBER),
        SCORES_MEMBER,
    )
    checksums = _require_object(
        strict_json_loads(members[CHECKSUMS_MEMBER], CHECKSUMS_MEMBER),
        CHECKSUMS_MEMBER,
    )
    _validate_manifest(manifest)
    _validate_checksums(checksums, members)
    datasets, conditions = _validate_claimed_scores_shape(claimed_scores)
    answer_rows = parse_answer_rows(members[ANSWERS_MEMBER])
    if not members[RAW_OUTPUTS_MEMBER].startswith(b"\x1f\x8b"):
        raise ArtifactPackageError(
            "invalid_artifact_gzip",
            f"{RAW_OUTPUTS_MEMBER} is not a gzip stream.",
            member=RAW_OUTPUTS_MEMBER,
        )
    if manifest["evidence"]["answer_rows"] != len(answer_rows):
        raise ArtifactPackageError(
            "artifact_answer_count_mismatch",
            "manifest.json answer row count does not match answers.jsonl.gz.",
        )
    computed_scores = aggregate_claimed_scores(answer_rows, datasets, conditions)
    validate_claimed_scores(claimed_scores, computed_scores)
    return ArtifactPackage(
        members=members,
        manifest=manifest,
        claimed_scores=claimed_scores,
        answer_rows=answer_rows,
        computed_scores=computed_scores,
    )


def report_from_claimed_scores(
    scores: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Build the existing leaderboard report shape from claimed integer counts."""
    datasets, conditions = _validate_claimed_scores_shape(scores)
    expected_conditions = [
        "main_noncot",
        "main_cot",
        "no_image_noncot",
        "no_image_cot",
        "no_image_plus_noncot",
        "no_image_plus_cot",
    ]
    if conditions != expected_conditions:
        raise ArtifactPackageError(
            "artifact_condition_set_mismatch",
            "The Track 3 package must contain the six conditions in canonical order.",
        )
    dataset_rows = []
    for dataset in datasets:
        experiments: dict[str, dict[str, dict[str, Any]]] = {}
        for mode in ("main", "no_image", "no_image_plus"):
            experiments[mode] = {}
            for prompt_mode in ("noncot", "cot"):
                result = scores["datasets"][dataset][f"{mode}_{prompt_mode}"]
                experiments[mode][prompt_mode] = {
                    **result,
                    "accuracy": round(result["correct"] / result["total"], 6),
                }
        dataset_rows.append({"dataset": dataset, "experiments": experiments})
    summary: dict[str, float] = {}
    for condition in conditions:
        values = [
            scores["datasets"][dataset][condition]["correct"]
            / scores["datasets"][dataset][condition]["total"]
            for dataset in datasets
        ]
        summary[condition] = round(sum(values) / len(values), 6)
    summary["cot_delta"] = round(summary["main_cot"] - summary["main_noncot"], 6)
    return {
        "schema_version": "ms-vista-track3-claimed-report/v1",
        "model": model,
        "conditions": conditions,
        "datasets": dataset_rows,
        "summary": summary,
    }


def checksums_document(members: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": CHECKSUMS_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": {
            member: {
                "sha256": sha256_bytes(members[member]),
                "size_bytes": len(members[member]),
            }
            for member in CHECKSUMMED_MEMBERS
        },
    }


def write_gzip_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write deterministic gzip JSONL and return the number of rows."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0, filename="") as gz_handle:
            for row in rows:
                gz_handle.write(canonical_json_bytes(row))
                count += 1
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(temporary, destination)
    return count


def write_artifact_package(
    output_path: str | Path,
    manifest: dict[str, Any],
    claimed_scores: dict[str, Any],
    answers_gzip_path: str | Path,
    raw_outputs_gzip_path: str | Path,
) -> Path:
    """Write a deterministic package from already-streamed gzip evidence files."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {
        MANIFEST_MEMBER: canonical_json_bytes(manifest),
        SCORES_MEMBER: canonical_json_bytes(claimed_scores),
        ANSWERS_MEMBER: Path(answers_gzip_path).read_bytes(),
        RAW_OUTPUTS_MEMBER: Path(raw_outputs_gzip_path).read_bytes(),
    }
    members[CHECKSUMS_MEMBER] = canonical_json_bytes(checksums_document(members))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    fixed_timestamp = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for member in ARCHIVE_MEMBERS:
            info = zipfile.ZipInfo(member, date_time=fixed_timestamp)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = (
                zipfile.ZIP_STORED
                if member in {ANSWERS_MEMBER, RAW_OUTPUTS_MEMBER}
                else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, members[member], compresslevel=9)
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    read_artifact_package(output)
    return output


def condition_from_source(mode: str, prompt_mode: str) -> str:
    mode_names = {
        "main": "main",
        "noimage": "no_image",
        "noimgpp": "no_image_plus",
    }
    if mode not in mode_names or prompt_mode not in {"noncot", "cot"}:
        raise ArtifactPackageError(
            "unsupported_source_condition",
            f"Unsupported source condition {mode}/{prompt_mode}.",
        )
    return f"{mode_names[mode]}_{prompt_mode}"


def raw_output_row(source: dict[str, Any]) -> dict[str, Any]:
    dataset = str(source.get("dataset") or "").strip()
    index = str(source.get("index") or "").strip()
    condition = condition_from_source(
        str(source.get("mode") or ""), str(source.get("pmode") or "")
    )
    output = source.get("output")
    if not isinstance(output, str) or not output.strip():
        terminal = source.get("terminal_failure")
        output = (
            "INFERENCE_FAILED: " + str(source.get("error") or "unknown error")
            if terminal is not None or source.get("error")
            else "INFERENCE_FAILED"
        )
    return {
        "schema_version": RAW_OUTPUTS_SCHEMA_VERSION,
        "dataset": dataset,
        "question_id": f"{dataset}:{index}",
        "evaluation_group": f"{dataset}:{str(source.get('group', index)).strip()}",
        "condition": condition,
        "raw_output": output,
    }


def artifact_members_from_archive(source: bytes | str | Path) -> set[str]:
    """Inspect member names only, for legacy/new format dispatch."""
    try:
        with _open_zip(source) as archive:
            return {info.filename for info in archive.infolist()}
    except ArtifactPackageError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArtifactPackageError(
            "invalid_artifact_archive", "The uploaded file is not a readable ZIP package."
        ) from exc
