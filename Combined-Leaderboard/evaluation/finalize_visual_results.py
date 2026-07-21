from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.common.visual_pipeline import (
    ANSWER_PARSER_POLICY_ID,
    INVALID_FORMAT_ANSWER,
    INVALID_FORMAT_REASON,
    INTEGER_TOKEN_PATTERN,
    INTEGER_WORDS,
    LETTER_DISAMBIGUATION_MAX_LENGTH,
    MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
    MANDATORY_EXTRACTOR_MODEL_ID,
    MANDATORY_EXTRACTOR_MODEL_REVISION,
    has_valid_extractor_provenance,
    record_answer,
)


TRACKS = ("do_you_see_me", "minds_eye")
CURRENT_PIPELINE_REVISION = "unquantized-bf16-mandatory-independent-extraction-v11"
SUBMISSION_FIELDS = {"question_id", "condition", "answer"}
INVALID_FORMAT_METADATA_FIELDS = (
    "submission_status",
    "format_failure_reason",
    "raw_output_characters",
    "raw_output_bytes",
    "raw_output_sha256",
)


class FinalizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    source_dir: Path
    source_run: str
    track: str
    submission: Path
    diagnostics: Path
    run_config: Path
    source_manifest: Path | None
    model_id: str
    model_revision: str
    slug: str
    modified_at: float
    config: dict[str, Any]


def live_active_roots(results_root: Path) -> list[Path]:
    active_roots: list[Path] = []
    for marker in results_root.resolve().rglob(".active-run.json"):
        try:
            pid = int(read_json(marker).get("pid") or 0)
            os.kill(pid, 0)
        except (FinalizationError, OSError, ValueError):
            continue
        active_roots.append(marker.parent.resolve())
    return active_roots


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FinalizationError(f"Cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalizationError(
                f"{path} line {line_number} is invalid JSON: {exc.msg}."
            ) from exc
        if not isinstance(row, dict):
            raise FinalizationError(f"{path} line {line_number} must be an object.")
        rows.append(row)
    return rows


def expected_question_ids(project_root: Path, track: str) -> list[str]:
    rows = expected_questions(project_root, track)
    question_ids = [str(row.get("question_id") or "") for row in rows]
    return question_ids


def expected_questions(project_root: Path, track: str) -> list[dict[str, Any]]:
    questions = project_root / "tasks" / track / "questions.jsonl"
    rows = read_jsonl(questions)
    question_ids = [str(row.get("question_id") or "") for row in rows]
    if not question_ids or any(not question_id for question_id in question_ids):
        raise FinalizationError(f"Question bundle {questions} has missing identifiers.")
    if len(set(question_ids)) != len(question_ids):
        raise FinalizationError(f"Question bundle {questions} has duplicate identifiers.")
    return rows


def validate_submission(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != len(expected_ids):
        raise FinalizationError(
            f"{path} has {len(rows)} rows; expected {len(expected_ids)}."
        )
    actual_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if set(row) != SUBMISSION_FIELDS:
            raise FinalizationError(
                f"{path} row {index} has fields {sorted(row)}; expected "
                f"{sorted(SUBMISSION_FIELDS)}."
            )
        question_id = str(row["question_id"])
        answer = row["answer"]
        if row["condition"] != "standard":
            raise FinalizationError(f"{path} row {index} has a nonstandard condition.")
        if not isinstance(answer, str) or not answer.strip():
            raise FinalizationError(f"{path} row {index} has an empty answer.")
        actual_ids.append(question_id)
    if len(set(actual_ids)) != len(actual_ids):
        raise FinalizationError(f"{path} contains duplicate question identifiers.")
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise FinalizationError(
            f"{path} does not match question order and coverage; "
            f"missing={missing[:3]}, extra={extra[:3]}."
        )
    return rows


def validate_diagnostics(
    path: Path, expected_ids: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in by_id:
            raise FinalizationError(f"{path} has a missing or duplicate question identifier.")
        by_id[question_id] = row
    if set(by_id) != set(expected_ids):
        raise FinalizationError(f"{path} does not cover the complete question bundle.")
    return rows, by_id


def invalid_format_metadata(diagnostic: dict[str, Any]) -> dict[str, Any] | None:
    raw_output = diagnostic.get("output")
    if (
        diagnostic.get("error")
        or isinstance(raw_output, (dict, list))
        or raw_output is None
    ):
        return None
    raw_text = str(raw_output)
    if not raw_text.strip():
        return None
    raw_bytes = raw_text.encode("utf-8")
    return {
        "submission_status": "invalid_format",
        "format_failure_reason": INVALID_FORMAT_REASON,
        "raw_output_characters": len(raw_text),
        "raw_output_bytes": len(raw_bytes),
        "raw_output_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def legacy_unwrapped_answer_text(raw_output: Any) -> str:
    if isinstance(raw_output, (dict, list)) or raw_output is None:
        return ""
    text = str(raw_output).strip()
    if not text:
        return ""
    answer_blocks = re.findall(
        r"<answer>((?:(?!<answer>).)*?)</answer>", text, flags=re.I | re.S
    )
    if answer_blocks:
        text = answer_blocks[-1].strip()
    else:
        native_box_blocks = re.findall(
            r"<\|begin_of_box\|>"
            r"((?:(?!<\|begin_of_box\|>).)*?)"
            r"<\|end_of_box\|>",
            text,
            flags=re.I | re.S,
        )
        if native_box_blocks:
            text = native_box_blocks[-1].strip()
    if re.search(r"<think>", text, flags=re.I) and not re.search(
        r"</think>", text, flags=re.I
    ):
        return ""
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1].strip()
    return text.strip().strip("`").strip()


def legacy_integer_answer_v6(raw_output: Any) -> str:
    """Reproduce the broad v6 integer parser only for provenance migration."""
    text = legacy_unwrapped_answer_text(raw_output)
    if not text:
        return ""
    values = re.findall(
        r"(?<![A-Za-z0-9])-?\d(?:[\d,]*\d)?(?![A-Za-z0-9])", text
    )
    if values:
        return values[-1].replace(",", "")
    matches = re.findall(
        rf"\b({'|'.join(INTEGER_WORDS)})\b", text, flags=re.I
    )
    normalized = {INTEGER_WORDS[word.lower()] for word in matches}
    return str(normalized.pop()) if len(normalized) == 1 else ""


def legacy_integer_answer_v7(raw_output: Any) -> str:
    """Reproduce v7 explicit integer commitments before LaTeX-box support."""
    text = legacy_unwrapped_answer_text(raw_output)
    if not text:
        return ""
    exact = re.fullmatch(
        rf"\s*({INTEGER_TOKEN_PATTERN})\s*[.,:]?\s*", text, flags=re.I
    )
    explicit = re.findall(
        rf"(?i)(?:"
        rf"(?:final\s+)?(?:answer|response|count)\s*(?:is|:|=|-)?"
        rf"|there\s+(?:is|are|were)"
        rf"|(?:the\s+)?total\s+(?:is|equals|:|=|-)?"
        rf")\s*({INTEGER_TOKEN_PATTERN})(?=\W|$)",
        text,
    )
    final_line = text.splitlines()[-1].strip() if text.splitlines() else ""
    line_match = re.fullmatch(
        rf"({INTEGER_TOKEN_PATTERN})[.,:]?", final_line, flags=re.I
    )
    match = exact or (None if explicit else line_match)
    token = match.group(1) if match else (explicit[-1] if explicit else "")
    if not token:
        return ""
    return (
        str(INTEGER_WORDS[token.lower()])
        if token.lower() in INTEGER_WORDS
        else token.replace(",", "")
    )


def legacy_record_answers(source: dict[str, Any], answer_type: str) -> set[str]:
    if answer_type != "integer":
        answer = record_answer(source, answer_type)
        return {answer} if answer else set()
    answers: set[str] = set()
    for field in ("extracted_answer", "output"):
        if source.get(field) is None:
            continue
        answers.update(
            answer
            for answer in (
                legacy_integer_answer_v6(source[field]),
                legacy_integer_answer_v7(source[field]),
            )
            if answer
        )
        if answers and field == "extracted_answer":
            return answers
    return answers


def canonicalize_track_rows(
    submission_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_diagnostics = {
        str(row["question_id"]): dict(row) for row in diagnostic_rows
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    legacy_answers: dict[str, set[str]] = {}
    for question_id, source in source_diagnostics.items():
        question = questions_by_id[question_id]
        answer_type = str(question.get("answer_type") or "text")
        legacy_source = dict(source)
        legacy_source.pop("task", None)
        legacy_answer = legacy_record_answers(legacy_source, answer_type)
        if (
            str(question.get("task") or "") == "letter_disambiguation"
        ):
            legacy_answer = {
                answer.upper() if re.fullmatch(r"[A-Za-z]+", answer) else answer
                for answer in legacy_answer
            }
        legacy_answers[question_id] = legacy_answer
        diagnostic = dict(source)
        diagnostic["answer_type"] = answer_type
        diagnostic["task"] = str(question.get("task") or "")
        diagnostics[question_id] = diagnostic
    canonical_submissions: list[dict[str, Any]] = []
    for source_submission in submission_rows:
        submission = dict(source_submission)
        question_id = str(submission["question_id"])
        diagnostic = diagnostics[question_id]
        question = questions_by_id[question_id]
        answer_type = str(question.get("answer_type") or "text")
        task = str(question.get("task") or "")
        if not has_valid_extractor_provenance(diagnostic):
            raise FinalizationError(
                f"Diagnostic for {question_id} lacks mandatory independent "
                "extractor provenance."
            )
        parsed = record_answer(diagnostic, answer_type, task)
        if parsed:
            if submission["answer"] != parsed:
                if submission["answer"] not in {
                    INVALID_FORMAT_ANSWER,
                } | legacy_answers[question_id]:
                    raise FinalizationError(
                        f"Submission answer for {question_id} differs from its parsed diagnostic."
                    )
                submission["answer"] = parsed
            for field in INVALID_FORMAT_METADATA_FIELDS:
                diagnostic.pop(field, None)
        else:
            metadata = invalid_format_metadata(diagnostic)
            accepted_legacy_answers = {
                str(diagnostic.get("output")),
                INVALID_FORMAT_ANSWER,
            } | legacy_answers[question_id]
            accepted_legacy_answers.discard("")
            if metadata is None or submission["answer"] not in accepted_legacy_answers:
                raise FinalizationError(
                    f"Submission answer for {question_id} has no verified diagnostic source."
                )
            submission["answer"] = INVALID_FORMAT_ANSWER
            diagnostic.update(metadata)
        canonical_submissions.append(submission)
    canonical_diagnostics = [
        diagnostics[str(row["question_id"])] for row in diagnostic_rows
    ]
    return canonical_submissions, canonical_diagnostics


def answer_provenance_counts(
    submission_rows: list[dict[str, Any]],
    diagnostics_by_id: dict[str, dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, int]:
    strict_count = 0
    invalid_format_count = 0
    for submission in submission_rows:
        question_id = str(submission["question_id"])
        diagnostic = diagnostics_by_id[question_id]
        question = (questions_by_id or {}).get(question_id, {})
        answer_type = str(
            question.get("answer_type") or diagnostic.get("answer_type") or "text"
        )
        task = str(question.get("task") or diagnostic.get("task") or "")
        if not has_valid_extractor_provenance(diagnostic):
            raise FinalizationError(
                f"Diagnostic for {question_id} lacks mandatory independent "
                "extractor provenance."
            )
        parsed = record_answer(
            diagnostic, answer_type, task
        )
        if parsed:
            if submission["answer"] != parsed:
                raise FinalizationError(
                    f"Submission answer for {question_id} differs from its parsed diagnostic."
                )
            strict_count += 1
        else:
            metadata = invalid_format_metadata(diagnostic)
            if metadata is None or submission["answer"] not in {
                str(diagnostic.get("output")),
                INVALID_FORMAT_ANSWER,
            }:
                raise FinalizationError(
                    f"Submission answer for {question_id} has no verified diagnostic source."
                )
            if submission["answer"] == INVALID_FORMAT_ANSWER:
                mismatched = [
                    key for key, value in metadata.items() if diagnostic.get(key) != value
                ]
                if mismatched:
                    raise FinalizationError(
                        f"Invalid-format diagnostics for {question_id} have incorrect "
                        f"or missing fields: {', '.join(mismatched)}."
                    )
            invalid_format_count += 1
    return strict_count, invalid_format_count


def discover_candidates(
    results_root: Path, output_root: Path, project_root: Path
) -> list[Candidate]:
    expected = {
        track: expected_question_ids(project_root, track) for track in TRACKS
    }
    candidates: list[Candidate] = []
    resolved_output = output_root.resolve()
    active_roots = live_active_roots(results_root)
    for track in TRACKS:
        for submission in results_root.rglob(f"{track}_submission.jsonl"):
            source_dir = submission.parent
            if resolved_output == source_dir.resolve() or resolved_output in source_dir.resolve().parents:
                continue
            if any(
                active_root == source_dir.resolve()
                or active_root in source_dir.resolve().parents
                for active_root in active_roots
            ):
                continue
            diagnostics = source_dir / f"{track}.diagnostics.jsonl"
            run_config = source_dir / ".run_config.json"
            if not diagnostics.is_file() or not run_config.is_file():
                continue
            config = read_json(run_config)
            if (
                config.get("weight_loading") != "unquantized"
                or config.get("compute_dtype") != "bfloat16"
                or config.get("pipeline_revision") != CURRENT_PIPELINE_REVISION
            ):
                continue
            model_id = str(config.get("model_id") or "")
            model_revision = str(config.get("model_revision") or "")
            if not model_id or not model_revision:
                continue
            validate_submission(submission, expected[track])
            validate_diagnostics(diagnostics, expected[track])
            manifest = source_dir / "run_manifest.json"
            candidates.append(
                Candidate(
                    source_dir=source_dir,
                    source_run=str(source_dir.relative_to(results_root)),
                    track=track,
                    submission=submission,
                    diagnostics=diagnostics,
                    run_config=run_config,
                    source_manifest=manifest if manifest.is_file() else None,
                    model_id=model_id,
                    model_revision=model_revision,
                    slug=source_dir.name,
                    modified_at=submission.stat().st_mtime,
                    config=config,
                )
            )
    return candidates


def discover_canonical_candidates(
    output_root: Path, project_root: Path
) -> list[Candidate]:
    output_root = output_root.resolve()
    if not (output_root / "index.json").is_file():
        return []
    verify_canonical_results(
        output_root, project_root, allow_canonical_migration=True
    )
    index = read_json(output_root / "index.json")
    candidates: list[Candidate] = []
    for model_record in index["models"]:
        manifest_path = output_root / str(model_record["manifest"])
        manifest = read_json(manifest_path)
        model_dir = manifest_path.parent
        slug = str(model_record["slug"])
        for track in TRACKS:
            track_record = manifest["tracks"][track]
            run_config = model_dir / f"{track}.run_config.json"
            config = read_json(run_config)
            source_manifest_name = track_record.get("source_manifest")
            source_manifest = (
                model_dir / source_manifest_name
                if isinstance(source_manifest_name, str) and source_manifest_name
                else None
            )
            modified_at = datetime.fromisoformat(
                str(track_record["source_submission_modified_at"])
            ).timestamp()
            candidates.append(
                Candidate(
                    source_dir=model_dir,
                    source_run=str(
                        track_record.get("source_run") or f"final/{slug}"
                    ),
                    track=track,
                    submission=model_dir / f"{track}_submission.jsonl",
                    diagnostics=model_dir / f"{track}.diagnostics.jsonl",
                    run_config=run_config,
                    source_manifest=source_manifest,
                    model_id=str(manifest["model_id"]),
                    model_revision=str(manifest["model_revision"]),
                    slug=slug,
                    modified_at=modified_at,
                    config=config,
                )
            )
    return candidates


def select_complete_models(candidates: list[Candidate]) -> list[dict[str, Candidate]]:
    latest_by_track: dict[tuple[str, str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.model_id, candidate.model_revision, candidate.track)
        current = latest_by_track.get(key)
        if current is None or candidate.modified_at > current.modified_at:
            latest_by_track[key] = candidate

    revisions: dict[tuple[str, str], dict[str, Candidate]] = {}
    for (model_id, revision, track), candidate in latest_by_track.items():
        revisions.setdefault((model_id, revision), {})[track] = candidate

    complete_by_model: dict[str, tuple[float, dict[str, Candidate]]] = {}
    for (model_id, _revision), tracks in revisions.items():
        if set(tracks) != set(TRACKS):
            continue
        contract = {
            (
                candidate.config.get("weight_loading"),
                candidate.config.get("compute_dtype"),
                candidate.config.get("pipeline_revision"),
            )
            for candidate in tracks.values()
        }
        if len(contract) != 1:
            continue
        newest = max(candidate.modified_at for candidate in tracks.values())
        current = complete_by_model.get(model_id)
        if current is None or newest > current[0]:
            complete_by_model[model_id] = (newest, tracks)
    return [
        tracks
        for _model_id, (_modified_at, tracks) in sorted(complete_by_model.items())
    ]


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def copy_track(
    candidate: Candidate,
    destination: Path,
    results_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    source_submission_rows = read_jsonl(candidate.submission)
    source_diagnostic_rows, _diagnostics_by_id = validate_diagnostics(
        candidate.diagnostics,
        [str(row["question_id"]) for row in source_submission_rows],
    )
    questions_by_id = {
        str(row["question_id"]): row
        for row in expected_questions(project_root, candidate.track)
    }
    submission_rows, diagnostic_rows = canonicalize_track_rows(
        source_submission_rows,
        source_diagnostic_rows,
        questions_by_id,
    )
    diagnostics_by_id = {
        str(row["question_id"]): row for row in diagnostic_rows
    }
    strict_count, invalid_format_count = answer_provenance_counts(
        submission_rows, diagnostics_by_id, questions_by_id
    )

    copied: dict[str, dict[str, Any]] = {}
    submission_target = destination / candidate.submission.name
    diagnostics_target = destination / candidate.diagnostics.name
    write_jsonl(submission_target, submission_rows)
    write_jsonl(diagnostics_target, diagnostic_rows)
    copied[submission_target.name] = artifact_record(submission_target)
    copied[diagnostics_target.name] = artifact_record(diagnostics_target)

    source_files: list[Path] = []
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.attempt-*.diagnostics.jsonl")))
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.smoke*.diagnostics.jsonl")))
    for source in source_files:
        target = destination / source.name
        shutil.copy2(source, target)
        copied[target.name] = artifact_record(target)

    run_config_target = destination / f"{candidate.track}.run_config.json"
    shutil.copy2(candidate.run_config, run_config_target)
    copied[run_config_target.name] = artifact_record(run_config_target)

    source_manifest_name = None
    if candidate.source_manifest is not None:
        source_manifest_target = destination / f"{candidate.track}.source_manifest.json"
        shutil.copy2(candidate.source_manifest, source_manifest_target)
        copied[source_manifest_target.name] = artifact_record(source_manifest_target)
        source_manifest_name = source_manifest_target.name

    return {
        "row_count": len(submission_rows),
        "strict_answer_count": strict_count,
        "invalid_format_count": invalid_format_count,
        "source_run": candidate.source_run,
        "source_submission_modified_at": datetime.fromtimestamp(
            candidate.modified_at, timezone.utc
        ).isoformat(),
        "source_run_config": run_config_target.name,
        "source_manifest": source_manifest_name,
        "generation": candidate.config.get("generation", {}).get(candidate.track),
        "serving_engine": candidate.config.get("serving_engine"),
        "tensor_parallel_size": candidate.config.get("tensor_parallel_size"),
        "data_parallel_size": candidate.config.get("data_parallel_size"),
        "request_concurrency": candidate.config.get("request_concurrency"),
        "max_model_len": candidate.config.get("max_model_len"),
        "artifacts": copied,
    }


def verify_canonical_results(
    output_root: Path,
    project_root: Path,
    *,
    allow_canonical_migration: bool = False,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    index = read_json(output_root / "index.json")
    index_schema = int(index.get("schema_version") or 0)
    if index_schema not in {1, 2}:
        raise FinalizationError(
            f"Unsupported canonical index schema_version: {index_schema}."
        )
    models = index.get("models")
    if not isinstance(models, list) or index.get("model_count") != len(models):
        raise FinalizationError("Canonical index model_count does not match its model list.")
    expected = {
        track: expected_question_ids(project_root.resolve(), track) for track in TRACKS
    }
    verified_models: list[str] = []
    for model_record in models:
        if not isinstance(model_record, dict):
            raise FinalizationError("Canonical index contains a non-object model record.")
        manifest_relative = Path(str(model_record.get("manifest") or ""))
        manifest_path = (output_root / manifest_relative).resolve()
        if output_root not in manifest_path.parents or not manifest_path.is_file():
            raise FinalizationError(f"Invalid canonical manifest path: {manifest_relative}.")
        if sha256(manifest_path) != model_record.get("manifest_sha256"):
            raise FinalizationError(f"Canonical manifest hash mismatch: {manifest_relative}.")
        manifest = read_json(manifest_path)
        manifest_schema = int(manifest.get("schema_version") or 0)
        if manifest_schema not in {1, 2}:
            raise FinalizationError(
                f"Unsupported canonical manifest schema_version: {manifest_schema}."
            )
        if (
            manifest.get("model_id") != model_record.get("model_id")
            or manifest.get("model_revision") != model_record.get("model_revision")
        ):
            raise FinalizationError(f"Canonical index identity mismatch: {manifest_relative}.")
        tracks = manifest.get("tracks")
        if not isinstance(tracks, dict) or set(tracks) != set(TRACKS):
            raise FinalizationError(f"Canonical manifest is missing a track: {manifest_relative}.")
        model_dir = manifest_path.parent
        for track in TRACKS:
            track_record = tracks[track]
            artifacts = track_record.get("artifacts")
            if not isinstance(artifacts, dict):
                raise FinalizationError(f"Canonical {track} artifact map is missing.")
            for filename, expected_artifact in artifacts.items():
                artifact = (model_dir / filename).resolve()
                if model_dir not in artifact.parents or not artifact.is_file():
                    raise FinalizationError(f"Invalid canonical artifact path: {filename}.")
                if (
                    artifact.stat().st_size != expected_artifact.get("bytes")
                    or sha256(artifact) != expected_artifact.get("sha256")
                ):
                    raise FinalizationError(f"Canonical artifact hash mismatch: {artifact}.")
            submission = model_dir / f"{track}_submission.jsonl"
            diagnostics = model_dir / f"{track}.diagnostics.jsonl"
            submission_rows = validate_submission(submission, expected[track])
            diagnostic_rows, diagnostics_by_id = validate_diagnostics(
                diagnostics, expected[track]
            )
            questions_by_id = {
                str(row["question_id"]): row
                for row in expected_questions(project_root, track)
            }
            provenance_submission_rows = submission_rows
            provenance_diagnostics_by_id = diagnostics_by_id
            if allow_canonical_migration:
                (
                    provenance_submission_rows,
                    provenance_diagnostic_rows,
                ) = canonicalize_track_rows(
                    submission_rows,
                    diagnostic_rows,
                    questions_by_id,
                )
                provenance_diagnostics_by_id = {
                    str(row["question_id"]): row
                    for row in provenance_diagnostic_rows
                }
            strict_count, invalid_format_count = answer_provenance_counts(
                provenance_submission_rows,
                provenance_diagnostics_by_id,
                questions_by_id if manifest_schema >= 2 else None,
            )
            recorded_invalid_count = (
                track_record.get("invalid_format_count")
                if manifest_schema >= 2
                else track_record.get("exact_raw_output_fallback_count")
            )
            counts_match = (
                track_record.get("row_count") == len(submission_rows)
                and track_record.get("strict_answer_count") == strict_count
                and recorded_invalid_count == invalid_format_count
            )
            if not counts_match and allow_canonical_migration:
                recorded_strict_count = track_record.get("strict_answer_count")
                if (
                    track_record.get("row_count") != len(submission_rows)
                    or not isinstance(recorded_strict_count, int)
                    or not isinstance(recorded_invalid_count, int)
                    or recorded_strict_count + recorded_invalid_count
                    != len(submission_rows)
                ):
                    raise FinalizationError(
                        f"Canonical legacy provenance counts are inconsistent for "
                        f"{manifest.get('model_id')}/{track}."
                    )
                counts_match = True
            if not counts_match:
                raise FinalizationError(
                    f"Canonical answer provenance count mismatch for "
                    f"{manifest.get('model_id')}/{track}."
                )
        verified_models.append(str(manifest["model_id"]))
    return {"model_count": len(verified_models), "verified_models": verified_models}


def build_canonical_results(
    results_root: Path, output_root: Path, project_root: Path, dry_run: bool = False
) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    project_root = project_root.resolve()
    candidates = [
        *discover_canonical_candidates(output_root, project_root),
        *discover_candidates(results_root, output_root, project_root),
    ]
    selected = select_complete_models(candidates)
    if not selected:
        raise FinalizationError("No model has valid final submissions for both tracks.")

    plan = {
        tracks[TRACKS[0]].model_id: {
            track: candidate.source_run
            for track, candidate in tracks.items()
        }
        for tracks in selected
    }
    if dry_run:
        return {"selection": plan, "model_count": len(selected)}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        index_models: list[dict[str, Any]] = []
        for tracks in selected:
            representative = max(tracks.values(), key=lambda candidate: candidate.modified_at)
            destination = staging / representative.slug
            destination.mkdir(parents=True)
            track_records = {
                track: copy_track(
                    candidate, destination, results_root, project_root
                )
                for track, candidate in sorted(tracks.items())
            }
            manifest = {
                "schema_version": 2,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
                "selection_policy": {
                    "precision": "original-unquantized-bf16",
                    "pipeline_revision": CURRENT_PIPELINE_REVISION,
                    "track_selection": "newest-valid-submission-per-model-revision",
                    "required_tracks": list(TRACKS),
                    "canonical_answer_parser": ANSWER_PARSER_POLICY_ID,
                    "answer_acceptance": "mandatory-independent-llm-extraction",
                    "answer_extraction_method": MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
                    "answer_extractor_model": MANDATORY_EXTRACTOR_MODEL_ID,
                    "answer_extractor_revision": MANDATORY_EXTRACTOR_MODEL_REVISION,
                    "answer_extractor_image_access": False,
                    "answer_extractor_ground_truth_access": False,
                    "invalid_format_submission_value": INVALID_FORMAT_ANSWER,
                    "invalid_format_reason": INVALID_FORMAT_REASON,
                    "invalid_format_score_effect": "always-incorrect",
                    "letter_disambiguation_max_length": (
                        LETTER_DISAMBIGUATION_MAX_LENGTH
                    ),
                    "raw_output_retention": "diagnostics-only",
                },
                "model_id": representative.model_id,
                "model_revision": representative.model_revision,
                "weight_loading": "unquantized",
                "compute_dtype": "bfloat16",
                "tracks": track_records,
            }
            manifest_path = destination / "final_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            index_models.append(
                {
                    "slug": representative.slug,
                    "model_id": representative.model_id,
                    "model_revision": representative.model_revision,
                    "manifest": f"{representative.slug}/final_manifest.json",
                    "manifest_sha256": sha256(manifest_path),
                    "tracks": {
                        track: {
                            "row_count": record["row_count"],
                            "strict_answer_count": record["strict_answer_count"],
                            "invalid_format_count": record["invalid_format_count"],
                        }
                        for track, record in track_records.items()
                    },
                }
            )

        index = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_count": len(index_models),
            "models": sorted(index_models, key=lambda item: item["model_id"]),
        }
        (staging / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_canonical_results(staging, project_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        return index
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prune_source_runs(results_root: Path, output_root: Path) -> list[str]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    active_roots = live_active_roots(results_root)
    if active_roots:
        relative = ", ".join(str(path.relative_to(results_root)) for path in active_roots)
        raise FinalizationError(f"Refusing to prune active evaluation runs: {relative}.")
    removed: list[str] = []
    for path in sorted(results_root.glob("visual_suite*")):
        if not path.is_dir() or path.resolve() == output_root:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def prune_cache(results_root: Path) -> bool:
    results_root = results_root.resolve()
    active_roots = live_active_roots(results_root)
    if active_roots:
        relative = ", ".join(str(path.relative_to(results_root)) for path in active_roots)
        raise FinalizationError(f"Refusing to prune cache during active runs: {relative}.")
    cache = results_root / ".cache"
    if not cache.exists():
        return False
    shutil.rmtree(cache)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select, validate, and consolidate final DYS and Mind's Eye results."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--results-root", type=Path, default=project_root / "evaluation" / "results"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation" / "results" / "final",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prune-source-runs", action="store_true")
    parser.add_argument("--prune-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.verify_only:
            if args.dry_run or args.prune_source_runs or args.prune_cache:
                raise FinalizationError(
                    "--verify-only cannot be combined with --dry-run or pruning."
                )
            result = verify_canonical_results(args.output_root, args.project_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        result = build_canonical_results(
            args.results_root, args.output_root, args.project_root, args.dry_run
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.prune_source_runs or args.prune_cache:
            if args.dry_run:
                raise FinalizationError("Pruning cannot be used with --dry-run.")
            cleanup: dict[str, Any] = {}
            if args.prune_source_runs:
                cleanup["pruned_source_runs"] = prune_source_runs(
                    args.results_root, args.output_root
                )
            if args.prune_cache:
                cleanup["pruned_cache"] = prune_cache(args.results_root)
            print(json.dumps(cleanup, indent=2))
    except FinalizationError as exc:
        raise SystemExit(f"Finalization failed: {exc}") from exc


if __name__ == "__main__":
    main()
