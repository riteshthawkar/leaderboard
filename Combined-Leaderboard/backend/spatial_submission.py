"""Validation for official Spatial benchmark bundles and run manifests."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import re
import stat
import statistics
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config import (
    EVAL_CONDITIONS,
    GRADING,
    OFFICIAL_SPATIAL_MIN_SAMPLES,
    SPATIAL_BENCHMARK_SCHEMA_VERSION,
    SPATIAL_DATASET_KEYS,
    SPATIAL_REPORT_SCHEMA_VERSION,
    SPATIAL_RUN_SCHEMA_VERSION,
    SPATIAL_SUBMISSION_SCHEMA_VERSION,
)
from models.tasks import Diagnostics, GroupResult, TaskScore
from scoring.task_scorer import SubmissionValidationError
from constants import (
    MAX_SPATIAL_ARCHIVE_BYTES,
    MAX_SPATIAL_MANIFEST_BYTES,
    MAX_SPATIAL_REPORT_BYTES,
    MAX_SPATIAL_SUBMISSION_BYTES,
    MAX_SPATIAL_ZIP_COMPRESSION_RATIO,
)


_BUNDLE_HEALTH_CACHE = {}
SPATIAL_SUBMISSION_ARCHIVE_NAME = "spatial_reasoning_submission.zip"
SPATIAL_SUBMISSION_MEMBER = "submission.jsonl"
SPATIAL_MANIFEST_MEMBER = "run_manifest.json"
SPATIAL_REPORT_MEMBER = "leaderboard.json"
SPATIAL_ARCHIVE_MEMBERS = (
    SPATIAL_SUBMISSION_MEMBER,
    SPATIAL_MANIFEST_MEMBER,
    SPATIAL_REPORT_MEMBER,
)
SPATIAL_PUBLIC_ARTIFACT_NAMES = (
    SPATIAL_SUBMISSION_ARCHIVE_NAME,
    *SPATIAL_ARCHIVE_MEMBERS,
)
_SPATIAL_EVIDENCE_FIELDS = {
    "dataset",
    "question_id",
    "evaluation_group",
    "condition",
    "answer",
    "correct",
    "judge_method",
    "judge_attempts",
}

ContractSource = Path | bytes


def _archive_error(code: str, message: str, **details):
    raise SubmissionValidationError(code, message, field="file", **details)


def _read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size <= 0:
        _archive_error(
            "empty_spatial_archive_member",
            f"{info.filename} is empty. Rerun the spatial harness and upload its unchanged package.",
            archive_member=info.filename,
        )
    if info.file_size > limit:
        _archive_error(
            "spatial_archive_member_too_large",
            f"{info.filename} is larger than the allowed {limit // (1024 * 1024)} MB uncompressed limit.",
            archive_member=info.filename,
            declared_size=info.file_size,
            maximum_size=limit,
        )
    if info.flag_bits & 0x1:
        _archive_error(
            "encrypted_spatial_archive",
            "Encrypted ZIP packages are not supported. Upload the unchanged package produced by the harness.",
            archive_member=info.filename,
        )
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        _archive_error(
            "unsupported_spatial_archive_compression",
            "The spatial package uses an unsupported ZIP compression method. Rerun the current harness.",
            archive_member=info.filename,
        )
    mode = (info.external_attr >> 16) & 0xFFFF
    if info.is_dir() or stat.S_ISLNK(mode):
        _archive_error(
            "unsafe_spatial_archive_member",
            "The spatial package contains a directory or symbolic link. Only the three harness-generated files are allowed.",
            archive_member=info.filename,
        )
    if info.file_size and info.compress_size <= 0:
        _archive_error(
            "unsafe_spatial_archive_ratio",
            "The spatial package has an invalid compression ratio. Rerun the current harness.",
            archive_member=info.filename,
        )
    if info.file_size / max(info.compress_size, 1) > MAX_SPATIAL_ZIP_COMPRESSION_RATIO:
        _archive_error(
            "unsafe_spatial_archive_ratio",
            "The spatial package is compressed beyond the allowed safety ratio. Rerun the current harness instead of repackaging it.",
            archive_member=info.filename,
        )
    try:
        with archive.open(info, "r") as handle:
            value = handle.read(limit + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        _archive_error(
            "unreadable_spatial_archive_member",
            f"{info.filename} could not be read from the ZIP package. Rerun the harness and upload the new package.",
            archive_member=info.filename,
        )
    if len(value) > limit:
        _archive_error(
            "spatial_archive_member_too_large",
            f"{info.filename} exceeds its allowed uncompressed size.",
            archive_member=info.filename,
            maximum_size=limit,
        )
    if len(value) != info.file_size:
        _archive_error(
            "spatial_archive_size_mismatch",
            f"{info.filename} does not match the size declared by the ZIP package.",
            archive_member=info.filename,
        )
    return value


def read_spatial_submission_archive(archive_bytes: bytes) -> tuple[bytes, bytes, bytes]:
    """Read the exact three-file spatial package without extracting it to disk."""
    if not isinstance(archive_bytes, bytes) or not archive_bytes:
        _archive_error(
            "empty_spatial_submission_archive",
            f"The spatial submission package is empty. Select {SPATIAL_SUBMISSION_ARCHIVE_NAME} produced by the harness.",
        )
    if len(archive_bytes) > MAX_SPATIAL_ARCHIVE_BYTES:
        _archive_error(
            "spatial_submission_archive_too_large",
            f"The spatial ZIP package is larger than the allowed {MAX_SPATIAL_ARCHIVE_BYTES // (1024 * 1024)} MB limit.",
            maximum_size=MAX_SPATIAL_ARCHIVE_BYTES,
        )
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            duplicates = sorted(
                name for name, count in Counter(names).items() if count > 1
            )
            if duplicates:
                _archive_error(
                    "duplicate_spatial_archive_members",
                    "The spatial package contains duplicate files. Upload the unchanged harness-generated package.",
                    archive_members=names,
                )
            if len(names) != len(SPATIAL_ARCHIVE_MEMBERS) or set(names) != set(
                SPATIAL_ARCHIVE_MEMBERS
            ):
                missing = sorted(set(SPATIAL_ARCHIVE_MEMBERS) - set(names))
                unexpected = sorted(set(names) - set(SPATIAL_ARCHIVE_MEMBERS))
                _archive_error(
                    "invalid_spatial_archive_contents",
                    "The spatial package must contain exactly submission.jsonl, run_manifest.json, and leaderboard.json at its root.",
                    missing_members=missing,
                    unexpected_members=unexpected,
                )
            info_by_name = {info.filename: info for info in infos}
            submission_bytes = _read_zip_member(
                archive,
                info_by_name[SPATIAL_SUBMISSION_MEMBER],
                MAX_SPATIAL_SUBMISSION_BYTES,
            )
            manifest_bytes = _read_zip_member(
                archive,
                info_by_name[SPATIAL_MANIFEST_MEMBER],
                MAX_SPATIAL_MANIFEST_BYTES,
            )
            report_bytes = _read_zip_member(
                archive,
                info_by_name[SPATIAL_REPORT_MEMBER],
                MAX_SPATIAL_REPORT_BYTES,
            )
    except SubmissionValidationError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        _archive_error(
            "invalid_spatial_submission_archive",
            f"The uploaded file is not a readable ZIP package. Upload {SPATIAL_SUBMISSION_ARCHIVE_NAME} produced by the current harness.",
        )
    return submission_bytes, manifest_bytes, report_bytes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contract_source(source: ContractSource, label: str) -> bytes:
    if isinstance(source, bytes):
        if not source:
            raise ValueError(f"{label} is empty")
        return source
    try:
        value = Path(source).read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not value:
        raise ValueError(f"{label} is empty")
    return value


def _contract_sha256(source: ContractSource, label: str) -> str:
    return _sha256_bytes(_read_contract_source(source, label))


def _load_json_object(source: ContractSource, label: str) -> dict:
    try:
        text_value = _read_contract_source(source, label).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 text") from exc
    try:
        value = _strict_json_loads(text_value)
    except (json.JSONDecodeError, ValueError) as exc:
        parser_message = getattr(exc, "msg", str(exc))
        raise ValueError(f"{label} contains invalid JSON: {parser_message}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_official_benchmark_manifest(source: ContractSource) -> dict:
    manifest = _load_json_object(source, "Spatial benchmark manifest")
    if manifest.get("schema_version") != SPATIAL_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Spatial benchmark manifest has an unsupported schema_version")
    if manifest.get("demo") is not False:
        raise ValueError("Spatial benchmark manifest is a demo bundle")
    if manifest.get("datasets") != SPATIAL_DATASET_KEYS:
        raise ValueError("Spatial benchmark manifest does not contain the canonical 13 datasets")
    if manifest.get("dataset_count") != len(SPATIAL_DATASET_KEYS):
        raise ValueError("Spatial benchmark manifest dataset_count is incorrect")
    if manifest.get("required_conditions") != EVAL_CONDITIONS:
        raise ValueError("Spatial benchmark manifest does not contain the six required conditions")
    if manifest.get("primary_condition") != "main_noncot":
        raise ValueError("Spatial benchmark manifest primary_condition must be main_noncot")
    condition_counts = manifest.get("condition_counts")
    if not isinstance(condition_counts, dict) or set(condition_counts) != set(EVAL_CONDITIONS):
        raise ValueError("Spatial benchmark manifest condition_counts are incomplete")
    if any(not isinstance(value, int) or value <= 0 for value in condition_counts.values()):
        raise ValueError("Spatial benchmark manifest condition counts must be positive integers")
    condition_group_counts = manifest.get("condition_group_counts")
    if (
        not isinstance(condition_group_counts, dict)
        or set(condition_group_counts) != set(EVAL_CONDITIONS)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in condition_group_counts.values()
        )
    ):
        raise ValueError("Spatial benchmark manifest condition_group_counts are incomplete")
    for field in ("dataset_condition_counts", "dataset_condition_group_counts"):
        by_dataset = manifest.get(field)
        if not isinstance(by_dataset, dict) or set(by_dataset) != set(SPATIAL_DATASET_KEYS):
            raise ValueError(f"Spatial benchmark manifest {field} are incomplete")
        for dataset, counts in by_dataset.items():
            if (
                not isinstance(counts, dict)
                or set(counts) != set(EVAL_CONDITIONS)
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in counts.values()
                )
            ):
                raise ValueError(
                    f"Spatial benchmark manifest {field} are invalid for {dataset}"
                )
    if not str(manifest.get("benchmark_version") or "").strip():
        raise ValueError("Spatial benchmark manifest is missing benchmark_version")
    if not str(manifest.get("harness_version") or "").strip():
        raise ValueError("Spatial benchmark manifest is missing harness_version")
    dataset_files = manifest.get("dataset_files")
    if not isinstance(dataset_files, dict) or set(dataset_files) != set(SPATIAL_DATASET_KEYS):
        raise ValueError("Spatial benchmark manifest dataset file provenance is incomplete")
    for dataset, artifact in dataset_files.items():
        if (
            not isinstance(artifact, dict)
            or not str(artifact.get("filename") or "").endswith(".tsv")
            or not isinstance(artifact.get("size_bytes"), int)
            or artifact["size_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256") or ""))
        ):
            raise ValueError(f"Spatial benchmark manifest has invalid provenance for {dataset}")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != {"noncot", "cot"}:
        raise ValueError("Spatial benchmark manifest prompt provenance is incomplete")
    for prompt_mode, artifact in prompts.items():
        if not isinstance(artifact, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", str(artifact.get("sha256") or "")
        ):
            raise ValueError(f"Spatial benchmark manifest has invalid {prompt_mode} prompt provenance")
    ablation = manifest.get("ablation_manifest")
    if not isinstance(ablation, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(ablation.get("sha256") or "")
    ):
        raise ValueError("Spatial benchmark manifest ablation provenance is incomplete")
    judge = manifest.get("judge")
    expected_judge_decoding = {
        "strategy": "greedy",
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 4,
    }
    if (
        not isinstance(judge, dict)
        or judge.get("revision") != GRADING["spatial"].get("judge_model")
        or not re.fullmatch(r"[0-9a-f]{64}", str(judge.get("system_prompt_sha256") or ""))
        or judge.get("decoding") != expected_judge_decoding
    ):
        raise ValueError("Spatial benchmark manifest judge provenance is incomplete")
    decoding = manifest.get("decoding")
    expected_decoding = {
        "strategy": "greedy",
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
    if decoding != expected_decoding:
        raise ValueError("Spatial benchmark manifest decoding contract is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not {
        "questions",
        "submission_template",
    }.issubset(artifacts):
        raise ValueError("Spatial benchmark manifest public artifacts are incomplete")
    return manifest


def _iter_jsonl_objects(source: ContractSource, label: str):
    try:
        text_value = _read_contract_source(source, label).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 text") from exc
    for line_number, raw_line in enumerate(text_value.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = _strict_json_loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            parser_message = getattr(exc, "msg", str(exc))
            raise ValueError(
                f"{label} contains invalid JSON at line {line_number}: {parser_message}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        yield line_number, row


def _load_public_spatial_contract(
    manifest_path: ContractSource,
    template_path: ContractSource,
    questions_path: ContractSource,
) -> tuple[dict, dict[str, dict], set[tuple[str, str]]]:
    """Load and cross-check the public contract without reading ground truth."""
    manifest = load_official_benchmark_manifest(manifest_path)

    questions_artifact = (manifest.get("artifacts") or {}).get("questions") or {}
    if questions_artifact.get("sha256") != _contract_sha256(
        questions_path,
        "Spatial public question identifiers",
    ):
        raise ValueError("Spatial public question identifier hash does not match the manifest")
    questions: dict[str, dict] = {}
    dataset_condition_counts: dict[str, Counter] = {
        dataset: Counter() for dataset in SPATIAL_DATASET_KEYS
    }
    dataset_condition_groups: dict[str, dict[str, set[str]]] = {
        dataset: {condition: set() for condition in EVAL_CONDITIONS}
        for dataset in SPATIAL_DATASET_KEYS
    }
    for line_number, row in _iter_jsonl_objects(
        questions_path, "Spatial public question identifiers"
    ):
        question_id = str(row.get("question_id") or "").strip()
        dataset = str(row.get("dataset_key") or "").strip()
        evaluation_group = str(row.get("evaluation_group") or "").strip()
        conditions = row.get("conditions")
        if not question_id:
            raise ValueError(
                f"Spatial public question identifier line {line_number} has no question_id"
            )
        if question_id in questions:
            raise ValueError("Spatial public question identifiers contain duplicates")
        if dataset not in SPATIAL_DATASET_KEYS:
            raise ValueError(
                f"Spatial public question identifier line {line_number} has an invalid dataset_key"
            )
        if not evaluation_group:
            raise ValueError(
                f"Spatial public question identifier line {line_number} has no evaluation_group"
            )
        if (
            not isinstance(conditions, list)
            or not conditions
            or len(conditions) != len(set(conditions))
            or any(condition not in EVAL_CONDITIONS for condition in conditions)
            or conditions != sorted(conditions, key=EVAL_CONDITIONS.index)
        ):
            raise ValueError(
                f"Spatial public question identifier line {line_number} has invalid conditions"
            )
        questions[question_id] = {
            "dataset": dataset,
            "evaluation_group": evaluation_group,
            "conditions": conditions,
        }
        for condition in conditions:
            dataset_condition_counts[dataset][condition] += 1
            dataset_condition_groups[dataset][condition].add(evaluation_group)

    if questions_artifact.get("rows") != len(questions):
        raise ValueError("Spatial public question identifier count does not match the manifest")
    if manifest.get("unique_question_ids") != len(questions):
        raise ValueError("Spatial benchmark manifest unique_question_ids is incorrect")

    actual_dataset_counts = {
        dataset: {
            condition: dataset_condition_counts[dataset][condition]
            for condition in EVAL_CONDITIONS
        }
        for dataset in SPATIAL_DATASET_KEYS
    }
    actual_dataset_group_counts = {
        dataset: {
            condition: len(dataset_condition_groups[dataset][condition])
            for condition in EVAL_CONDITIONS
        }
        for dataset in SPATIAL_DATASET_KEYS
    }
    if actual_dataset_counts != manifest["dataset_condition_counts"]:
        raise ValueError("Spatial public question counts do not match the benchmark manifest")
    if actual_dataset_group_counts != manifest["dataset_condition_group_counts"]:
        raise ValueError("Spatial public scoring groups do not match the benchmark manifest")
    actual_condition_counts = {
        condition: sum(
            actual_dataset_counts[dataset][condition]
            for dataset in SPATIAL_DATASET_KEYS
        )
        for condition in EVAL_CONDITIONS
    }
    actual_condition_group_counts = {
        condition: sum(
            actual_dataset_group_counts[dataset][condition]
            for dataset in SPATIAL_DATASET_KEYS
        )
        for condition in EVAL_CONDITIONS
    }
    if actual_condition_counts != manifest["condition_counts"]:
        raise ValueError("Spatial condition counts do not match public question identifiers")
    if actual_condition_group_counts != manifest["condition_group_counts"]:
        raise ValueError("Spatial scoring group counts do not match public question identifiers")

    template_artifact = (manifest.get("artifacts") or {}).get("submission_template") or {}
    if template_artifact.get("sha256") != _contract_sha256(
        template_path,
        "Spatial submission template",
    ):
        raise ValueError("Spatial submission template hash does not match the manifest")
    template_keys: set[tuple[str, str]] = set()
    template_counts = Counter()
    for line_number, row in _iter_jsonl_objects(
        template_path, "Spatial submission template"
    ):
        question_id = str(row.get("question_id") or "").strip()
        condition = str(row.get("condition") or "").strip()
        question = questions.get(question_id)
        if question is None or condition not in question["conditions"]:
            raise ValueError(
                f"Spatial submission template line {line_number} has an invalid question_id or condition"
            )
        if "answer" not in row or row.get("answer") != "":
            raise ValueError(
                f"Spatial submission template line {line_number} must have an empty answer"
            )
        key = (condition, question_id)
        if key in template_keys:
            raise ValueError("Spatial submission template contains duplicate rows")
        template_keys.add(key)
        template_counts[condition] += 1
    expected_template_keys = {
        (condition, question_id)
        for question_id, question in questions.items()
        for condition in question["conditions"]
    }
    if template_keys != expected_template_keys:
        raise ValueError("Spatial submission template ID coverage is incorrect")
    if dict(template_counts) != manifest["condition_counts"]:
        raise ValueError("Spatial submission template condition counts are incorrect")
    if template_artifact.get("rows") != len(template_keys):
        raise ValueError("Spatial submission template row count does not match the manifest")
    return manifest, questions, template_keys


def _inspect_spatial_bundle(
    manifest_path: Path,
    template_path: Path,
    questions_path: Path,
) -> tuple[str, dict]:
    details = {
        "production_ready": False,
        "samples": 0,
        "min_samples": OFFICIAL_SPATIAL_MIN_SAMPLES,
        "datasets": 0,
        "conditions": 0,
    }
    try:
        manifest, questions, _template_keys = _load_public_spatial_contract(
            manifest_path,
            template_path,
            questions_path,
        )
        details.update(
            {
                "samples": len(questions),
                "datasets": len(manifest["datasets"]),
                "conditions": len(manifest["required_conditions"]),
                "benchmark_version": manifest["benchmark_version"],
                "manifest_sha256": _sha256_file(Path(manifest_path)),
                "verification_mode": "public_evidence",
            }
        )
        if len(questions) < OFFICIAL_SPATIAL_MIN_SAMPLES:
            raise ValueError(
                f"Spatial public contract contains {len(questions)} IDs; at least "
                f"{OFFICIAL_SPATIAL_MIN_SAMPLES} are required"
            )
        details["production_ready"] = True
        return "healthy", details
    except (OSError, ValueError) as exc:
        details["error"] = str(exc)
        return "unhealthy", details


def spatial_bundle_health(
    manifest_path: Path,
    template_path: Path,
    questions_path: Path,
) -> tuple[str, dict]:
    paths = tuple(
        Path(path).resolve()
        for path in (manifest_path, template_path, questions_path)
    )
    try:
        signatures = tuple(
            (path.stat().st_mtime_ns, path.stat().st_size) for path in paths
        )
    except OSError:
        return _inspect_spatial_bundle(*paths)
    cache_key = (
        tuple(str(path) for path in paths),
        signatures,
        OFFICIAL_SPATIAL_MIN_SAMPLES,
        GRADING["spatial"].get("judge_model"),
    )
    cached = _BUNDLE_HEALTH_CACHE.get(cache_key)
    if cached is not None:
        status, details = cached
        return status, dict(details)
    result = _inspect_spatial_bundle(*paths)
    _BUNDLE_HEALTH_CACHE.clear()
    _BUNDLE_HEALTH_CACHE[cache_key] = result
    return result[0], dict(result[1])


def _manifest_error(code: str, message: str, **details):
    raise SubmissionValidationError(code, message, field="file", **details)


def _strict_json_loads(value: str):
    def reject_constant(constant: str):
        raise ValueError(f"non-finite JSON value {constant}")

    return json.loads(value, parse_constant=reject_constant)


def parse_spatial_evidence(
    submission_bytes: bytes,
    benchmark_manifest_path: ContractSource,
    template_path: ContractSource,
    questions_path: ContractSource,
) -> tuple[list[dict], dict, dict]:
    """Validate public per-sample evidence and recompute its aggregate counts."""
    try:
        text_value = submission_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        _manifest_error(
            "invalid_spatial_evidence_encoding",
            "The packaged submission.jsonl is not valid UTF-8 text. Rerun the current harness and upload its unchanged package.",
        )

    manifest, questions, expected_keys = _load_public_spatial_contract(
        benchmark_manifest_path,
        template_path,
        questions_path,
    )
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    condition_counts = Counter()
    group_states: dict[tuple[str, str, str], dict] = {}
    for line_number, raw_line in enumerate(text_value.splitlines(), start=1):
        if not raw_line.strip():
            continue
        if len(raw_line) > 16_384:
            _manifest_error(
                "spatial_evidence_line_too_long",
                f"submission.jsonl line {line_number} is unexpectedly large. The public evidence file must contain compact final-answer records only.",
                line_number=line_number,
            )
        try:
            row = _strict_json_loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            _manifest_error(
                "invalid_spatial_evidence_json",
                f"submission.jsonl line {line_number} is not valid strict JSON: {exc}.",
                line_number=line_number,
            )
        if not isinstance(row, dict):
            _manifest_error(
                "invalid_spatial_evidence_row",
                f"submission.jsonl line {line_number} must be one JSON object.",
                line_number=line_number,
            )
        missing_fields = sorted(_SPATIAL_EVIDENCE_FIELDS - set(row))
        unexpected_fields = sorted(set(row) - _SPATIAL_EVIDENCE_FIELDS)
        if missing_fields or unexpected_fields:
            _manifest_error(
                "invalid_spatial_evidence_fields",
                f"submission.jsonl line {line_number} does not match the current public evidence schema. Rerun the current harness.",
                line_number=line_number,
                missing_fields=missing_fields,
                unexpected_fields=unexpected_fields,
            )

        question_id = str(row.get("question_id") or "").strip()
        condition = str(row.get("condition") or "").strip()
        dataset = str(row.get("dataset") or "").strip()
        evaluation_group = str(row.get("evaluation_group") or "").strip()
        answer = row.get("answer")
        judge_method = str(row.get("judge_method") or "").strip()
        judge_attempts = row.get("judge_attempts")
        correct = row.get("correct")
        question = questions.get(question_id)
        key = (condition, question_id)

        if key not in expected_keys or question is None:
            _manifest_error(
                "unknown_spatial_evidence_sample",
                f"submission.jsonl line {line_number} contains a question_id and condition pair that is not in the current public template.",
                line_number=line_number,
                question_id=question_id,
                condition=condition,
            )
        if key in seen:
            _manifest_error(
                "duplicate_spatial_evidence_sample",
                f"submission.jsonl line {line_number} repeats {condition}/{question_id}.",
                line_number=line_number,
                question_id=question_id,
                condition=condition,
            )
        if dataset != question["dataset"] or evaluation_group != question["evaluation_group"]:
            _manifest_error(
                "spatial_evidence_metadata_mismatch",
                f"submission.jsonl line {line_number} has dataset or scoring-group metadata that does not match the public benchmark contract.",
                line_number=line_number,
                question_id=question_id,
            )
        if not isinstance(answer, str) or not re.fullmatch(r"[A-Z0]", answer):
            _manifest_error(
                "invalid_spatial_evidence_answer",
                f"submission.jsonl line {line_number} must contain one uppercase option letter or 0 in answer.",
                line_number=line_number,
                question_id=question_id,
            )
        if type(correct) is not bool:
            _manifest_error(
                "invalid_spatial_evidence_correctness",
                f"submission.jsonl line {line_number} must contain a boolean correct field.",
                line_number=line_number,
                question_id=question_id,
            )
        if judge_method not in {"qwen_llm_judge", "explicit_abstention"}:
            _manifest_error(
                "invalid_spatial_judge_method",
                f"submission.jsonl line {line_number} has an unsupported judge_method. Rerun the current harness.",
                line_number=line_number,
                judge_method=judge_method,
            )
        if (
            not isinstance(judge_attempts, int)
            or isinstance(judge_attempts, bool)
            or judge_attempts < 0
            or judge_attempts > 100
        ):
            _manifest_error(
                "invalid_spatial_judge_attempts",
                f"submission.jsonl line {line_number} has an invalid judge_attempts value.",
                line_number=line_number,
            )
        if (
            (judge_method == "explicit_abstention" and judge_attempts != 0)
            or (judge_method == "qwen_llm_judge" and judge_attempts < 1)
        ):
            _manifest_error(
                "inconsistent_spatial_judge_evidence",
                f"submission.jsonl line {line_number} has inconsistent judge method and attempt metadata.",
                line_number=line_number,
            )

        seen.add(key)
        condition_counts[condition] += 1
        state = group_states.setdefault(
            (dataset, condition, evaluation_group),
            {"all_correct": True, "variants": 0},
        )
        state["all_correct"] = bool(state["all_correct"] and correct)
        state["variants"] += 1
        records.append(
            {
                "row_index": len(records) + 1,
                "line_number": line_number,
                "question_id": question_id,
                "condition": condition,
                "answer": answer,
                "dataset": dataset,
                "evaluation_group": evaluation_group,
                "correct": correct,
                "judge_method": judge_method,
                "judge_attempts": judge_attempts,
            }
        )

    missing_keys = expected_keys - seen
    if missing_keys:
        examples = [f"{condition}/{question_id}" for condition, question_id in sorted(missing_keys)[:10]]
        _manifest_error(
            "missing_spatial_evidence_samples",
            f"submission.jsonl is missing {len(missing_keys)} required public evidence row(s). Rerun the harness to completion.",
            missing_count=len(missing_keys),
            examples=examples,
        )
    if dict(condition_counts) != manifest["condition_counts"]:
        _manifest_error(
            "spatial_evidence_condition_count_mismatch",
            "submission.jsonl condition counts do not match the official public contract.",
            evidence_counts=dict(condition_counts),
            expected_counts=manifest["condition_counts"],
        )

    aggregate: dict[tuple[str, str], list[int]] = {
        (dataset, condition): [0, 0]
        for dataset in SPATIAL_DATASET_KEYS
        for condition in EVAL_CONDITIONS
    }
    for (dataset, condition, _group), state in group_states.items():
        aggregate[(dataset, condition)][0] += int(state["all_correct"])
        aggregate[(dataset, condition)][1] += 1
    for dataset in SPATIAL_DATASET_KEYS:
        for condition in EVAL_CONDITIONS:
            expected_total = manifest["dataset_condition_group_counts"][dataset][condition]
            if aggregate[(dataset, condition)][1] != expected_total:
                _manifest_error(
                    "spatial_evidence_group_count_mismatch",
                    f"Public evidence for {dataset}/{condition} contains an invalid number of scoring groups.",
                    dataset=dataset,
                    condition=condition,
                    expected_groups=expected_total,
                    evidence_groups=aggregate[(dataset, condition)][1],
                )

    dataset_rows = []
    for dataset in SPATIAL_DATASET_KEYS:
        experiments = {}
        for mode_name in ("main", "no_image", "no_image_plus"):
            experiments[mode_name] = {}
            for prompt_mode in ("noncot", "cot"):
                condition = f"{mode_name}_{prompt_mode}"
                correct_count, total_count = aggregate[(dataset, condition)]
                experiments[mode_name][prompt_mode] = {
                    "correct": correct_count,
                    "total": total_count,
                    "accuracy": round(correct_count / total_count, 6),
                }
        dataset_rows.append({"dataset": dataset, "experiments": experiments})
    summary = {}
    for condition in EVAL_CONDITIONS:
        values = [
            aggregate[(dataset, condition)][0] / aggregate[(dataset, condition)][1]
            for dataset in SPATIAL_DATASET_KEYS
        ]
        summary[condition] = round(sum(values) / len(values), 6)
    summary["cot_delta"] = round(
        summary["main_cot"] - summary["main_noncot"],
        6,
    )
    computed_report = {"datasets": dataset_rows, "summary": summary}
    return records, computed_report, manifest


def _close_metric(received, expected) -> bool:
    return (
        isinstance(received, (int, float))
        and not isinstance(received, bool)
        and math.isfinite(float(received))
        and abs(float(received) - float(expected)) <= 0.0000005
    )


def validate_spatial_report(
    report_bytes: bytes,
    model_name: str,
    computed_report: dict,
) -> dict:
    """Validate the submitted aggregate against the public per-sample evidence."""
    try:
        report_text = report_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        _manifest_error(
            "invalid_spatial_report_encoding",
            "The packaged leaderboard.json is not valid UTF-8 text. Rerun the current harness.",
        )
    try:
        report = _strict_json_loads(report_text)
    except (json.JSONDecodeError, ValueError) as exc:
        _manifest_error(
            "invalid_spatial_report_json",
            f"The packaged leaderboard.json is not valid strict JSON: {exc}.",
        )
    if not isinstance(report, dict):
        _manifest_error(
            "invalid_spatial_report_shape",
            "The packaged leaderboard.json must contain one JSON object.",
        )
    if report.get("schema_version") != SPATIAL_REPORT_SCHEMA_VERSION:
        _manifest_error(
            "unsupported_spatial_report_version",
            f"The aggregate report schema is unsupported. Expected {SPATIAL_REPORT_SCHEMA_VERSION}; rerun the current harness.",
            received=report.get("schema_version"),
        )
    report_model = str((report.get("model") or {}).get("name") or "").strip()
    if report_model != model_name:
        _manifest_error(
            "spatial_report_model_mismatch",
            "The model name in leaderboard.json does not match the selected registered model.",
            report_model_name=report_model,
            form_model_name=model_name,
        )
    if report.get("conditions") != EVAL_CONDITIONS:
        _manifest_error(
            "spatial_report_condition_mismatch",
            "leaderboard.json does not contain the six conditions in canonical order.",
        )
    dataset_rows = report.get("datasets")
    expected_rows = computed_report["datasets"]
    if not isinstance(dataset_rows, list) or len(dataset_rows) != len(expected_rows):
        _manifest_error(
            "spatial_report_dataset_mismatch",
            "leaderboard.json must contain exactly one result row for each of the 13 datasets.",
        )
    for index, expected_row in enumerate(expected_rows):
        row = dataset_rows[index]
        if not isinstance(row, dict) or row.get("dataset") != expected_row["dataset"]:
            _manifest_error(
                "spatial_report_dataset_mismatch",
                "leaderboard.json dataset rows are missing or not in canonical order.",
                dataset_index=index,
            )
        experiments = row.get("experiments")
        if not isinstance(experiments, dict) or set(experiments) != {
            "main",
            "no_image",
            "no_image_plus",
        }:
            _manifest_error(
                "invalid_spatial_report_experiments",
                f"leaderboard.json has incomplete experiments for {expected_row['dataset']}.",
            )
        for mode_name, expected_prompts in expected_row["experiments"].items():
            prompts = experiments.get(mode_name)
            if not isinstance(prompts, dict) or set(prompts) != {"noncot", "cot"}:
                _manifest_error(
                    "invalid_spatial_report_experiments",
                    f"leaderboard.json has incomplete prompt modes for {expected_row['dataset']}/{mode_name}.",
                )
            for prompt_mode, expected_result in expected_prompts.items():
                result = prompts.get(prompt_mode)
                if (
                    not isinstance(result, dict)
                    or set(result) != {"correct", "total", "accuracy"}
                    or not isinstance(result.get("correct"), int)
                    or isinstance(result.get("correct"), bool)
                    or not isinstance(result.get("total"), int)
                    or isinstance(result.get("total"), bool)
                    or result["correct"] != expected_result["correct"]
                    or result["total"] != expected_result["total"]
                    or not _close_metric(result.get("accuracy"), expected_result["accuracy"])
                ):
                    _manifest_error(
                        "spatial_report_evidence_mismatch",
                        f"The aggregate for {expected_row['dataset']}/{mode_name}/{prompt_mode} does not match submission.jsonl.",
                        dataset=expected_row["dataset"],
                        mode=mode_name,
                        prompt_mode=prompt_mode,
                    )
    summary = report.get("summary")
    expected_summary = computed_report["summary"]
    if not isinstance(summary, dict) or set(summary) != set(expected_summary):
        _manifest_error(
            "invalid_spatial_report_summary",
            "leaderboard.json has an incomplete summary block.",
        )
    for metric, expected_value in expected_summary.items():
        if not _close_metric(summary.get(metric), expected_value):
            _manifest_error(
                "spatial_report_evidence_mismatch",
                f"The reported {metric} summary does not match the public per-sample evidence.",
                metric=metric,
            )
    return report


_SPATIAL_DATASET_LABELS = {
    "BLINK": "BLINK",
    "CV-Bench-2D": "CV-Bench (2D)",
    "CV-Bench-3D": "CV-Bench (3D)",
    "MMVP": "MMVP",
    "RealWorldQA": "RealWorldQA",
    "VStarBench": "V*Bench",
    "MMSIBench_wo_circular": "MMSI-Bench",
    "3DSRBench": "3DSRBench",
    "VSR_MCQ": "VSR",
    "SpatialBench": "SpatialBench",
    "MindCube": "MindCube",
    "OmniSpatial": "OmniSpatial",
    "SAT-Real": "SAT (Real)",
}


def build_spatial_task_score(
    report: dict,
    model_name: str,
    model_meta: dict,
    run_metadata: dict,
    submission_id: str | None = None,
) -> TaskScore:
    """Build a leaderboard result from validated public evidence, without GT scoring."""
    primary_rows = []
    groups = {}
    for row in report["datasets"]:
        result = row["experiments"]["main"]["noncot"]
        label = _SPATIAL_DATASET_LABELS[row["dataset"]]
        groups[label] = GroupResult(
            name=label,
            total_samples=result["total"],
            correct_samples=result["correct"],
            accuracy=float(result["accuracy"]),
            meta={"dataset_key": row["dataset"]},
        )
        primary_rows.append(result)
    correct_samples = sum(result["correct"] for result in primary_rows)
    total_samples = sum(result["total"] for result in primary_rows)
    micro_accuracy = correct_samples / total_samples if total_samples else 0.0
    dataset_accuracies = [float(result["accuracy"]) for result in primary_rows]
    task_spread = statistics.stdev(dataset_accuracies) if len(dataset_accuracies) > 1 else 0.0
    summary = report["summary"]
    submission_id = str(submission_id or uuid.uuid4())
    evidence_base = f"/api/public/submissions/{submission_id}"
    diagnostics = Diagnostics(
        conditions_present=list(EVAL_CONDITIONS),
        standard_accuracy=float(summary["main_noncot"]),
        cot_accuracy=float(summary["main_cot"]),
        cot_delta=float(summary["cot_delta"]),
        shortcut_score=float(summary["no_image_noncot"]),
        shortcut_score_cot=float(summary["no_image_cot"]),
        hallucination_resistance=float(summary["no_image_plus_noncot"]),
        hallucination_resistance_cot=float(summary["no_image_plus_cot"]),
    )
    return TaskScore(
        task_id="spatial",
        submission_id=submission_id,
        model_name=model_name,
        submitted_at=datetime.now(timezone.utc),
        accuracy=micro_accuracy,
        macro_accuracy=float(summary["main_noncot"]),
        task_spread=task_spread,
        accuracy_std=task_spread,
        random_baseline=None,
        score_method="dataset_macro_from_public_evidence",
        total_samples=total_samples,
        correct_samples=correct_samples,
        groups=groups,
        diagnostics=diagnostics,
        grading={
            "method": "harness_reported_public_evidence",
            "judge_model": run_metadata.get("judge_revision"),
            "paper": GRADING["spatial"].get("paper"),
            "backend": "public_evidence_consistency_validation",
            "llm_graded": True,
            "server_ground_truth_evaluation": False,
            "method_counts": run_metadata.get("judge_method_counts") or {},
        },
        model_meta=dict(model_meta or {}),
        metadata={
            "submission_format": "spatial_evidence_zip",
            "spatial_run": run_metadata,
            "public_evidence": {
                "available": True,
                "verification_level": "provenance_and_arithmetic",
                "server_ground_truth_evaluation": False,
                "url": f"{evidence_base}/evidence",
                "answers_url": f"{evidence_base}/answers.jsonl",
                "archive_url": (
                    f"{evidence_base}/artifacts/{SPATIAL_SUBMISSION_ARCHIVE_NAME}"
                ),
                "notice": (
                    "The server validates package provenance, public sample coverage, and score arithmetic. "
                    "It does not independently compare spatial answers with private ground truth."
                ),
            },
        },
    )


def validate_run_manifest(
    manifest_bytes: bytes,
    submission_bytes: bytes,
    report_bytes: bytes,
    model_name: str,
    records: Iterable[dict],
    benchmark_manifest_path: ContractSource,
) -> dict:
    if not isinstance(manifest_bytes, bytes) or not manifest_bytes:
        _manifest_error(
            "missing_spatial_run_manifest",
            "The spatial ZIP package is missing run_manifest.json. Rerun the current harness and upload its unchanged package.",
        )
    try:
        manifest_text = manifest_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        _manifest_error(
            "invalid_run_manifest_encoding",
            "The packaged run_manifest.json is not valid UTF-8 text. Rerun the harness and upload its unchanged ZIP package.",
        )
    try:
        run_manifest = _strict_json_loads(manifest_text)
    except (json.JSONDecodeError, ValueError) as exc:
        line_number = getattr(exc, "lineno", 1)
        parser_message = getattr(exc, "msg", str(exc))
        _manifest_error(
            "invalid_run_manifest_json",
            f"The packaged run_manifest.json is not valid strict JSON ({parser_message} at line {line_number}). Rerun the harness and upload its unchanged ZIP package.",
            line_number=line_number,
        )
    if not isinstance(run_manifest, dict):
        _manifest_error(
            "invalid_run_manifest_shape",
            "The run manifest must be one JSON object, not an array or scalar value.",
        )

    if run_manifest.get("schema_version") != SPATIAL_RUN_SCHEMA_VERSION:
        _manifest_error(
            "unsupported_run_manifest_version",
            f"The run manifest schema is unsupported. Expected {SPATIAL_RUN_SCHEMA_VERSION}; rerun the current official harness.",
            received=run_manifest.get("schema_version"),
        )
    if run_manifest.get("submission_schema_version") != SPATIAL_SUBMISSION_SCHEMA_VERSION:
        _manifest_error(
            "unsupported_spatial_submission_version",
            "The spatial output file was produced by an incompatible harness version. Download and rerun the current harness.",
        )
    if run_manifest.get("debug") is not False:
        _manifest_error(
            "debug_spatial_run_not_allowed",
            "This manifest describes a debug or limited run. Leaderboard submissions must evaluate the complete official dataset.",
        )

    manifest_model = str((run_manifest.get("model") or {}).get("name") or "").strip()
    if manifest_model != model_name:
        _manifest_error(
            "spatial_model_name_mismatch",
            f"The form model name ({model_name}) does not match the harness manifest model name ({manifest_model or 'missing'}). Use the exact same name in both places.",
            form_model_name=model_name,
            manifest_model_name=manifest_model,
        )
    if run_manifest.get("datasets") != SPATIAL_DATASET_KEYS:
        _manifest_error(
            "spatial_dataset_set_mismatch",
            "The run manifest does not contain all 13 official datasets in canonical order. Rerun the harness without dataset overrides.",
        )
    if run_manifest.get("conditions") != EVAL_CONDITIONS:
        _manifest_error(
            "spatial_condition_set_mismatch",
            "The run manifest does not contain all six official conditions in canonical order. Rerun the harness without mode overrides.",
        )
    error_counts = run_manifest.get("error_counts")
    if not isinstance(error_counts, dict) or not error_counts:
        _manifest_error(
            "missing_spatial_error_counts",
            "The run manifest is missing inference and judge error counts. Rerun the current official harness.",
        )
    nonzero_errors = {
        key: value
        for key, value in error_counts.items()
        if not isinstance(value, int) or value != 0
    }
    if nonzero_errors:
        _manifest_error(
            "spatial_run_contains_errors",
            "The harness reported failed, missing, or unjudged outputs. Resolve every harness error and rerun before submitting.",
            error_counts=error_counts,
        )

    records = list(records)
    actual_counts = dict(Counter(str(row.get("condition") or "") for row in records))
    declared_counts = run_manifest.get("condition_counts")
    if declared_counts != actual_counts:
        _manifest_error(
            "spatial_condition_count_mismatch",
            "The packaged submission.jsonl row counts do not match run_manifest.json. Rerun the current harness and upload the new ZIP package unchanged.",
            response_counts=actual_counts,
            manifest_counts=declared_counts,
        )
    submission_artifact = (run_manifest.get("artifacts") or {}).get("submission") or {}
    actual_submission_hash = _sha256_bytes(submission_bytes)
    declared_hash = str(submission_artifact.get("sha256") or "")
    if not declared_hash or not hmac.compare_digest(declared_hash, actual_submission_hash):
        _manifest_error(
            "spatial_submission_hash_mismatch",
            "The response file hash does not match the run manifest. The files are from different runs or one was edited after evaluation.",
        )
    if submission_artifact.get("filename") != "submission.jsonl":
        _manifest_error(
            "invalid_spatial_submission_artifact",
            "The run manifest does not identify the official submission.jsonl artifact.",
        )
    if submission_artifact.get("rows") != len(records):
        _manifest_error(
            "spatial_submission_row_count_mismatch",
            f"The run manifest declares {submission_artifact.get('rows')} response rows, but the uploaded file contains {len(records)}.",
        )
    report_artifact = (run_manifest.get("artifacts") or {}).get(
        "leaderboard_report"
    ) or {}
    actual_report_hash = _sha256_bytes(report_bytes)
    declared_report_hash = str(report_artifact.get("sha256") or "")
    if (
        report_artifact.get("filename") != SPATIAL_REPORT_MEMBER
        or not declared_report_hash
        or not hmac.compare_digest(declared_report_hash, actual_report_hash)
        or report_artifact.get("size_bytes") != len(report_bytes)
        or report_artifact.get("dataset_count") != len(SPATIAL_DATASET_KEYS)
    ):
        _manifest_error(
            "spatial_report_artifact_mismatch",
            "The packaged leaderboard.json does not match the run manifest. The report is missing, edited, or from another run.",
        )

    benchmark_manifest = load_official_benchmark_manifest(benchmark_manifest_path)
    expected_benchmark_hash = _contract_sha256(
        benchmark_manifest_path,
        "Spatial benchmark manifest",
    )
    if run_manifest.get("benchmark_manifest_sha256") != expected_benchmark_hash:
        _manifest_error(
            "spatial_benchmark_version_mismatch",
            "This run used a different benchmark manifest than the server. Download the current manifest and rerun the harness.",
            expected_manifest_sha256=expected_benchmark_hash,
        )
    if declared_counts != benchmark_manifest["condition_counts"]:
        _manifest_error(
            "spatial_official_count_mismatch",
            "The run row counts do not match the current official benchmark bundle. Download the current manifest and rerun the harness.",
        )
    if run_manifest.get("harness_version") != benchmark_manifest.get("harness_version"):
        _manifest_error(
            "spatial_harness_version_mismatch",
            "This run used a different harness version than the official benchmark. Download and rerun the current harness.",
        )

    for field, label in (
        ("dataset_files", "dataset files"),
        ("prompts", "prompt files"),
        ("ablation_manifest", "ablation manifest"),
    ):
        if run_manifest.get(field) != benchmark_manifest.get(field):
            _manifest_error(
                "spatial_provenance_mismatch",
                f"The run's {label} do not match the official benchmark manifest. Rebuild the data with the current harness and rerun evaluation.",
                provenance_field=field,
            )
    expected_judge = benchmark_manifest.get("judge") or {}
    actual_judge = run_manifest.get("judge") or {}
    configured_revision = GRADING["spatial"].get("judge_model")
    if (
        actual_judge.get("revision") != configured_revision
        or actual_judge.get("revision") != expected_judge.get("revision")
        or actual_judge.get("system_prompt_sha256")
        != expected_judge.get("system_prompt_sha256")
        or actual_judge.get("decoding") != expected_judge.get("decoding")
    ):
        _manifest_error(
            "spatial_judge_mismatch",
            f"The run was not judged with the required {configured_revision} revision and official judge prompt. Rerun the current harness with the configured judge.",
        )
    expected_decoding = benchmark_manifest.get("decoding") or {}
    actual_decoding = run_manifest.get("decoding") or {}
    if any(actual_decoding.get(key) != value for key, value in expected_decoding.items()):
        _manifest_error(
            "spatial_decoding_mismatch",
            "The run did not use the official greedy decoding settings. Restore the harness defaults and rerun evaluation.",
        )
    method_counts = actual_judge.get("method_counts")
    evidence_method_counts = dict(
        Counter(str(row.get("judge_method") or "") for row in records)
    )
    if (
        not isinstance(method_counts, dict)
        or any(not isinstance(value, int) or value < 0 for value in method_counts.values())
        or sum(method_counts.values()) != len(records)
        or method_counts != evidence_method_counts
    ):
        _manifest_error(
            "spatial_judge_count_mismatch",
            "The run manifest judge counts do not cover every submitted output. Rerun the judge to completion.",
        )

    return {
        "schema_version": run_manifest["schema_version"],
        "harness_version": run_manifest.get("harness_version"),
        "run_manifest_sha256": _sha256_bytes(manifest_bytes),
        "report_sha256": actual_report_hash,
        "benchmark_manifest_sha256": expected_benchmark_hash,
        "benchmark_version": benchmark_manifest["benchmark_version"],
        "judge_revision": actual_judge.get("revision"),
        "judge_method_counts": method_counts,
        "condition_counts": declared_counts,
        "verification_mode": "public_evidence",
        "server_ground_truth_evaluation": False,
    }
