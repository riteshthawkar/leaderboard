"""Prepare and assess a blinded audit of locally recovered answers.

The audit package is deliberately independent of benchmark images and ground
truth. Reviewers decide only whether a recovered answer faithfully represents
one answer clearly committed to in the evaluated model's raw response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.vllm_runner import (
    LOCAL_ANSWER_EXTRACTION_METHOD,
    UNCERTAINTY_PATTERN,
)
from evaluation.finalize_visual_results import (
    FinalizationError,
    read_json,
    read_jsonl,
    sha256,
    verify_canonical_results,
)


TRACKS = ("do_you_see_me", "minds_eye")
REVIEW_LABELS = ("faithful", "unsupported", "ambiguous", "unclear")
FINAL_LABELS = ("faithful", "unsupported", "ambiguous")
SCHEMA_VERSION = 1
DEFAULT_SAMPLE_SIZE = 350
DEFAULT_SEED = "local-extractor-human-audit-v1"
MIN_PROBABILITY_SAMPLE_SIZE = 300
MIN_REVIEWER_KAPPA = 0.80
MAX_FALSE_ACCEPT_RATE = 0.01
HIGH_RISK_FLAGS = {
    "source_truncated",
    "response_at_least_4000_characters",
    "uncertainty_language",
    "prior_legacy_extraction",
    "extractor_nonstop_finish",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _stable_digest(seed: str, namespace: str, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{key}".encode("utf-8")).digest()


def _write_private(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_private_json(path: Path, value: Any) -> None:
    _write_private(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_private_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(_canonical_json(row) + "\n" for row in rows)
    _write_private(path, payload)


def _question_bundle(project_root: Path, track: str) -> dict[str, dict[str, Any]]:
    path = project_root / "tasks" / track / "questions.jsonl"
    rows = read_jsonl(path)
    by_id = {str(row.get("question_id") or ""): row for row in rows}
    if not rows or len(by_id) != len(rows) or "" in by_id:
        raise FinalizationError(f"Invalid or duplicate question IDs in {path}.")
    forbidden = {"answer", "correct_answer", "ground_truth", "target"}
    for question_id, row in by_id.items():
        leaked = sorted(forbidden.intersection(row))
        if leaked:
            raise FinalizationError(
                f"Public question {track}/{question_id} contains forbidden "
                f"answer field(s): {', '.join(leaked)}."
            )
    return by_id


def _answer_contract(answer_type: str, task: str) -> str:
    if answer_type == "integer":
        return "One integer."
    if answer_type == "mcq_letter":
        return "One option letter."
    if answer_type == "mcq_index_1_4":
        return "One option index from 1 through 4."
    if task == "letter_disambiguation":
        return "One through nine uppercase letters without separators."
    if task in {"form_constancy", "visual_form_constancy"}:
        return "Exactly Yes or No."
    return "One short answer in the format requested by the question."


def _risk_flags(
    source: dict[str, Any], staged: dict[str, Any]
) -> list[str]:
    output = str(source.get("output") or "")
    flags: list[str] = []
    if str(source.get("finish_reason") or "").lower() in {
        "length",
        "max_tokens",
    }:
        flags.append("source_truncated")
    if len(output) >= 4000:
        flags.append("response_at_least_4000_characters")
    if UNCERTAINTY_PATTERN.search(output):
        flags.append("uncertainty_language")
    if source.get("extracted_answer"):
        flags.append("prior_legacy_extraction")
    extractor_finish_reason = str(
        staged.get("extractor_finish_reason") or ""
    ).lower()
    if extractor_finish_reason and extractor_finish_reason != "stop":
        flags.append("extractor_nonstop_finish")
    if staged.get("extractor_attempts"):
        flags.append("multiple_extractor_attempts")
    return flags


def _load_population(
    *,
    project_root: Path,
    final_root: Path,
    staging_root: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    project_root = project_root.resolve()
    final_root = final_root.resolve()
    staging_root = staging_root.resolve()
    verify_canonical_results(final_root, project_root)

    report_path = staging_root / "extraction_report.json"
    report_hash = sha256(report_path)
    report = read_json(report_path)
    locked_inputs: dict[Path, str] = {report_path: report_hash}
    verification = report.get("verification") or {}
    extractor = report.get("extractor") or {}
    if verification.get("status") != "passed" or not verification.get(
        "canonical_sources_unchanged"
    ):
        raise FinalizationError(
            "The extraction report has not passed immutable-source verification."
        )
    if extractor.get("ground_truth_access") is not False:
        raise FinalizationError("Extractor ground-truth isolation is not proven.")
    if extractor.get("image_access") is not False:
        raise FinalizationError("Extractor image isolation is not proven.")
    if Path(str(report.get("source_final_root"))).resolve() != final_root:
        raise FinalizationError("Extraction report references a different final root.")
    if Path(str(report.get("staging_root"))).resolve() != staging_root:
        raise FinalizationError("Extraction report references a different staging root.")

    index_path = final_root / "index.json"
    index = read_json(index_path)
    locked_inputs[index_path] = sha256(index_path)
    models = list(index.get("models") or [])
    track_reports = {
        (str(row.get("slug") or ""), str(row.get("track") or "")): row
        for row in report.get("tracks") or []
    }
    questions = {
        track: _question_bundle(project_root, track) for track in TRACKS
    }
    for track in TRACKS:
        question_path = project_root / "tasks" / track / "questions.jsonl"
        locked_inputs[question_path] = sha256(question_path)
    population: list[dict[str, Any]] = []

    for model in models:
        slug = str(model.get("slug") or "")
        if not slug:
            raise FinalizationError("Canonical index contains a model without a slug.")
        for track in TRACKS:
            track_report = track_reports.get((slug, track))
            if track_report is None:
                raise FinalizationError(f"Extraction report is missing {slug}/{track}.")

            source_diagnostics_path = final_root / slug / f"{track}.diagnostics.jsonl"
            staged_diagnostics_path = (
                staging_root / slug / f"{track}.diagnostics.jsonl"
            )
            source_submission_path = final_root / slug / f"{track}_submission.jsonl"
            staged_submission_path = staging_root / slug / f"{track}_submission.jsonl"
            expected_hashes = {
                source_diagnostics_path: track_report.get(
                    "source_diagnostics_sha256"
                ),
                staged_diagnostics_path: track_report.get(
                    "staged_diagnostics_sha256"
                ),
                source_submission_path: track_report.get(
                    "source_submission_sha256"
                ),
                staged_submission_path: track_report.get(
                    "staged_submission_sha256"
                ),
            }
            for path, expected_hash in expected_hashes.items():
                if not expected_hash or sha256(path) != expected_hash:
                    raise FinalizationError(f"Locked artifact hash mismatch: {path}")
                locked_inputs[path] = str(expected_hash)

            source_diagnostics = read_jsonl(source_diagnostics_path)
            staged_diagnostics = read_jsonl(staged_diagnostics_path)
            source_submissions = read_jsonl(source_submission_path)
            staged_submissions = read_jsonl(staged_submission_path)
            lengths = {
                len(source_diagnostics),
                len(staged_diagnostics),
                len(source_submissions),
                len(staged_submissions),
            }
            if len(lengths) != 1:
                raise FinalizationError(
                    f"Artifact lengths differ for {slug}/{track}."
                )

            recovered_for_track = 0
            for source, staged, source_submission, staged_submission in zip(
                source_diagnostics,
                staged_diagnostics,
                source_submissions,
                staged_submissions,
                strict=True,
            ):
                ids = {
                    str(source.get("question_id") or ""),
                    str(staged.get("question_id") or ""),
                    str(source_submission.get("question_id") or ""),
                    str(staged_submission.get("question_id") or ""),
                }
                if len(ids) != 1 or "" in ids:
                    raise FinalizationError(
                        f"Artifact row order differs for {slug}/{track}."
                    )
                question_id = next(iter(ids))
                extracted_answer = str(staged.get("extracted_answer") or "")
                is_recovered = bool(
                    staged.get("answer_extraction_method")
                    == LOCAL_ANSWER_EXTRACTION_METHOD
                    and extracted_answer
                )
                if not is_recovered:
                    continue
                recovered_for_track += 1
                if str(staged_submission.get("answer") or "") != extracted_answer:
                    raise FinalizationError(
                        f"Staged answer/provenance mismatch at {slug}/{track}/"
                        f"{question_id}."
                    )
                if source.get("output") != staged.get("output"):
                    raise FinalizationError(
                        f"Raw response changed at {slug}/{track}/{question_id}."
                    )
                output_hash = _text_sha256(source.get("output"))
                if staged.get("extractor_source_output_sha256") != output_hash:
                    raise FinalizationError(
                        f"Extractor source hash mismatch at {slug}/{track}/"
                        f"{question_id}."
                    )
                question = questions[track].get(question_id)
                if question is None:
                    raise FinalizationError(
                        f"Unknown question {track}/{question_id}."
                    )
                answer_type = str(question.get("answer_type") or "text")
                task = str(question.get("task") or "")
                key = f"{slug}/{track}/{question_id}"
                population.append(
                    {
                        "key": key,
                        "slug": slug,
                        "track": track,
                        "question_id": question_id,
                        "question": str(question.get("question") or ""),
                        "question_sha256": _value_sha256(question),
                        "answer_type": answer_type,
                        "answer_contract": _answer_contract(answer_type, task),
                        "task": task,
                        "difficulty": str(question.get("difficulty") or ""),
                        "candidate_response": str(source.get("output") or ""),
                        "extracted_answer": extracted_answer,
                        "source_output_sha256": output_hash,
                        "source_finish_reason": str(
                            source.get("finish_reason") or ""
                        ),
                        "source_status": (
                            "unsupported_legacy_extraction"
                            if source.get("extracted_answer")
                            else "invalid_format"
                        ),
                        "risk_flags": _risk_flags(source, staged),
                    }
                )

            if recovered_for_track != int(
                track_report.get("recovered_count") or 0
            ):
                raise FinalizationError(
                    f"Recovered count mismatch for {slug}/{track}: "
                    f"{recovered_for_track} != {track_report.get('recovered_count')}."
                )

    keys = [row["key"] for row in population]
    if len(set(keys)) != len(keys):
        raise FinalizationError("Recovered-answer population contains duplicate keys.")
    if len(population) != int(report.get("recovered_count") or 0):
        raise FinalizationError(
            "Recovered-answer population does not match the extraction report."
        )
    for path, expected_hash in locked_inputs.items():
        if sha256(path) != expected_hash:
            raise FinalizationError(
                f"Locked audit input changed while the package was built: {path}"
            )
    return report, report_hash, population


def _review_content(row: dict[str, Any], audit_id: str) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "question": row["question"],
        "answer_type": row["answer_type"],
        "answer_contract": row["answer_contract"],
        "task": row["task"],
        "candidate_response": row["candidate_response"],
        "extracted_answer": row["extracted_answer"],
    }


def _protocol_text() -> str:
    return """# Local Extractor Blinded Review Protocol

## Review question

Does `extracted_answer` faithfully represent one final answer that the evaluated
model clearly commits to in `candidate_response`?

## Blinding rules

- Do not inspect the benchmark image, ground-truth answer, score, model identity,
  or the other reviewer's decisions.
- Do not solve the benchmark question. Judge only the textual evidence in the
  candidate response.
- Edit only `review_label` and `review_notes` in your assigned reviewer file.

## Labels

- `faithful`: the response clearly commits to exactly the extracted answer.
- `unsupported`: the extracted answer is absent or is only mentioned as an
  intermediate possibility.
- `ambiguous`: the response supports multiple final answers or never commits to
  one answer.
- `unclear`: the reviewer cannot decide; adjudication is required.

Reviewers must work independently. An adjudicator completes `adjudication.jsonl`
only for disagreements or `unclear` decisions. Any confirmed `unsupported` or
`ambiguous` row fails the current extraction policy; do not remove that row from
the audit sample or repair it manually.
"""


def prepare_audit(
    *,
    project_root: Path,
    final_root: Path,
    staging_root: Path,
    output_root: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    if sample_size < MIN_PROBABILITY_SAMPLE_SIZE:
        raise FinalizationError(
            f"Probability sample must contain at least "
            f"{MIN_PROBABILITY_SAMPLE_SIZE} rows."
        )
    report, report_hash, population = _load_population(
        project_root=project_root,
        final_root=final_root,
        staging_root=staging_root,
    )
    if sample_size > len(population):
        sample_size = len(population)

    probability_rows = sorted(
        population,
        key=lambda row: _stable_digest(seed, "probability", row["key"]),
    )[:sample_size]
    probability_keys = {row["key"] for row in probability_rows}
    risk_keys = {
        row["key"]
        for row in population
        if HIGH_RISK_FLAGS.intersection(row["risk_flags"])
    }

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in population:
        stratum = (row["slug"], row["track"], row["task"])
        strata.setdefault(stratum, []).append(row)
    coverage_keys = {
        min(
            rows,
            key=lambda row: _stable_digest(seed, "coverage", row["key"]),
        )["key"]
        for rows in strata.values()
    }
    selected_keys = probability_keys | risk_keys | coverage_keys
    selected_rows = sorted(
        (row for row in population if row["key"] in selected_keys),
        key=lambda row: _stable_digest(seed, "review-order", row["key"]),
    )

    review_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    adjudication_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        audit_id = hashlib.sha256(
            f"{seed}:{report_hash}:{row['key']}".encode("utf-8")
        ).hexdigest()[:20]
        content = _review_content(row, audit_id)
        content_hash = _value_sha256(content)
        review_rows.append(
            {
                **content,
                "content_sha256": content_hash,
                "review_label": "",
                "review_notes": "",
            }
        )
        mapping_rows.append(
            {
                "audit_id": audit_id,
                "content_sha256": content_hash,
                "key": row["key"],
                "slug": row["slug"],
                "track": row["track"],
                "question_id": row["question_id"],
                "question_sha256": row["question_sha256"],
                "source_output_sha256": row["source_output_sha256"],
                "source_finish_reason": row["source_finish_reason"],
                "source_status": row["source_status"],
                "response_characters": len(row["candidate_response"]),
                "risk_flags": row["risk_flags"],
                "sample_groups": {
                    "probability_sample": row["key"] in probability_keys,
                    "risk_census": row["key"] in risk_keys,
                    "stratum_coverage": row["key"] in coverage_keys,
                },
            }
        )
        adjudication_rows.append(
            {
                "audit_id": audit_id,
                "final_label": "",
                "adjudication_notes": "",
            }
        )

    output_root = output_root.resolve()
    reviewer_1_path = output_root / "reviewer_1.jsonl"
    reviewer_2_path = output_root / "reviewer_2.jsonl"
    mapping_path = output_root / "audit_mapping.jsonl"
    adjudication_path = output_root / "adjudication.jsonl"
    protocol_path = output_root / "PROTOCOL.md"
    _write_private_jsonl(reviewer_1_path, review_rows)
    _write_private_jsonl(reviewer_2_path, review_rows)
    _write_private_jsonl(mapping_path, mapping_rows)
    _write_private_jsonl(adjudication_path, adjudication_rows)
    _write_private(protocol_path, _protocol_text())

    probability_upper_bound = 1.0 - math.pow(0.05, 1.0 / sample_size)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_independent_review",
        "package_root": str(output_root),
        "sampling": {
            "seed": seed,
            "population_size": len(population),
            "probability_sample_size": len(probability_keys),
            "risk_census_size": len(risk_keys),
            "risk_census_flags": sorted(HIGH_RISK_FLAGS),
            "stratum_count": len(strata),
            "stratum_coverage_size": len(coverage_keys),
            "total_review_items": len(selected_rows),
            "zero_error_one_sided_95_percent_upper_bound": round(
                probability_upper_bound, 8
            ),
        },
        "population": {
            "by_track": dict(
                sorted(Counter(row["track"] for row in population).items())
            ),
            "by_model": dict(
                sorted(Counter(row["slug"] for row in population).items())
            ),
            "risk_flags": dict(
                sorted(
                    Counter(
                        flag
                        for row in population
                        for flag in row["risk_flags"]
                    ).items()
                )
            ),
        },
        "blinding": {
            "ground_truth_access": False,
            "image_access": False,
            "model_identity_in_reviewer_files": False,
            "score_in_reviewer_files": False,
            "reviewers_are_independent": True,
        },
        "release_gate": {
            "minimum_probability_sample_size": MIN_PROBABILITY_SAMPLE_SIZE,
            "minimum_cohens_kappa": MIN_REVIEWER_KAPPA,
            "maximum_false_accept_rate": MAX_FALSE_ACCEPT_RATE,
            "require_zero_false_accepts_in_selected_items": True,
            "require_zero_false_accepts_in_risk_census": True,
            "require_all_disagreements_adjudicated": True,
        },
        "source": {
            "project_root": str(project_root.resolve()),
            "final_root": str(final_root.resolve()),
            "staging_root": str(staging_root.resolve()),
            "extraction_report_path": str(
                (staging_root / "extraction_report.json").resolve()
            ),
            "extraction_report_sha256": report_hash,
            "extractor": report.get("extractor"),
            "extraction_verification": report.get("verification"),
        },
        "files": {
            path.name: {
                "sha256_at_creation": sha256(path),
                "mode": oct(path.stat().st_mode & 0o777),
            }
            for path in (
                reviewer_1_path,
                reviewer_2_path,
                mapping_path,
                adjudication_path,
                protocol_path,
            )
        },
    }
    _write_private_json(output_root / "manifest.json", manifest)
    return manifest


def _load_review_file(
    path: Path, mapping: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    content_fields = {
        "audit_id",
        "question",
        "answer_type",
        "answer_contract",
        "task",
        "candidate_response",
        "extracted_answer",
    }
    for row in rows:
        audit_id = str(row.get("audit_id") or "")
        if not audit_id or audit_id in by_id:
            raise FinalizationError(f"Duplicate or missing audit ID in {path}.")
        expected = mapping.get(audit_id)
        if expected is None:
            raise FinalizationError(f"Unexpected audit ID {audit_id} in {path}.")
        immutable_content = {key: row.get(key) for key in content_fields}
        if _value_sha256(immutable_content) != expected.get("content_sha256"):
            raise FinalizationError(
                f"Immutable review content changed for {audit_id} in {path}."
            )
        label = str(row.get("review_label") or "").strip().lower()
        if label and label not in REVIEW_LABELS:
            raise FinalizationError(
                f"Invalid review label {label!r} for {audit_id} in {path}."
            )
        row["review_label"] = label
        by_id[audit_id] = row
    if set(by_id) != set(mapping):
        raise FinalizationError(f"Review coverage is incomplete in {path}.")
    return by_id


def _cohens_kappa(labels_1: list[str], labels_2: list[str]) -> float | None:
    if not labels_1 or len(labels_1) != len(labels_2):
        return None
    observed = sum(a == b for a, b in zip(labels_1, labels_2, strict=True)) / len(
        labels_1
    )
    counts_1 = Counter(labels_1)
    counts_2 = Counter(labels_2)
    expected = sum(
        (counts_1[label] / len(labels_1)) * (counts_2[label] / len(labels_2))
        for label in REVIEW_LABELS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def assess_audit(
    *,
    audit_root: Path,
    reviewer_1_id: str,
    reviewer_2_id: str,
    adjudicator_id: str = "",
    project_root: Path | None = None,
) -> dict[str, Any]:
    audit_root = audit_root.resolve()
    reviewer_1_id = reviewer_1_id.strip()
    reviewer_2_id = reviewer_2_id.strip()
    adjudicator_id = adjudicator_id.strip()
    if not reviewer_1_id or not reviewer_2_id:
        raise FinalizationError("Both reviewer identities are required.")
    if reviewer_1_id.casefold() == reviewer_2_id.casefold():
        raise FinalizationError("Reviewers must be different people.")

    manifest = read_json(audit_root / "manifest.json")
    source = manifest.get("source") or {}
    required_source_paths = ("project_root", "final_root", "staging_root")
    missing_source_paths = [
        field for field in required_source_paths if not source.get(field)
    ]
    if missing_source_paths:
        raise FinalizationError(
            "Audit manifest is missing source path(s): "
            + ", ".join(missing_source_paths)
        )
    recorded_project_root = Path(str(source["project_root"])).resolve()
    resolved_project_root = (
        project_root.resolve() if project_root is not None else recorded_project_root
    )

    def relocated_source_path(field: str) -> Path:
        recorded = Path(str(source[field])).resolve()
        if project_root is None:
            return recorded
        try:
            relative = recorded.relative_to(recorded_project_root)
        except ValueError as exc:
            raise FinalizationError(
                f"Recorded {field} is outside the recorded project root."
            ) from exc
        return resolved_project_root / relative

    _, extraction_report_hash, current_population = _load_population(
        project_root=resolved_project_root,
        final_root=relocated_source_path("final_root"),
        staging_root=relocated_source_path("staging_root"),
    )
    if extraction_report_hash != source.get("extraction_report_sha256"):
        raise FinalizationError(
            "The extraction report changed after the audit package was prepared."
        )
    mapping_path = audit_root / "audit_mapping.jsonl"
    protocol_path = audit_root / "PROTOCOL.md"
    for path in (mapping_path, protocol_path):
        expected = (
            (manifest.get("files") or {}).get(path.name) or {}
        ).get("sha256_at_creation")
        if not expected or sha256(path) != expected:
            raise FinalizationError(f"Immutable audit file changed: {path}")

    mapping_rows = read_jsonl(mapping_path)
    mapping = {str(row.get("audit_id") or ""): row for row in mapping_rows}
    if "" in mapping or len(mapping) != len(mapping_rows):
        raise FinalizationError("Audit mapping has missing or duplicate IDs.")
    current_by_key = {row["key"]: row for row in current_population}
    for audit_id, mapping_row in mapping.items():
        current = current_by_key.get(str(mapping_row.get("key") or ""))
        if current is None:
            raise FinalizationError(
                f"Audited recovery is absent from current staging: {audit_id}."
            )
        if mapping_row.get("source_output_sha256") != current.get(
            "source_output_sha256"
        ) or mapping_row.get("question_sha256") != current.get("question_sha256"):
            raise FinalizationError(
                f"Audited source changed after package creation: {audit_id}."
            )
        if _value_sha256(_review_content(current, audit_id)) != mapping_row.get(
            "content_sha256"
        ):
            raise FinalizationError(
                f"Audited answer changed after package creation: {audit_id}."
            )
    reviews_1 = _load_review_file(audit_root / "reviewer_1.jsonl", mapping)
    reviews_2 = _load_review_file(audit_root / "reviewer_2.jsonl", mapping)
    adjudication_rows = read_jsonl(audit_root / "adjudication.jsonl")
    adjudications = {
        str(row.get("audit_id") or ""): row for row in adjudication_rows
    }
    if set(adjudications) != set(mapping) or len(adjudications) != len(
        adjudication_rows
    ):
        raise FinalizationError("Adjudication file has incomplete or duplicate IDs.")

    completed_pairs: list[tuple[str, str]] = []
    final_labels: dict[str, str] = {}
    pending: list[str] = []
    disagreements: list[str] = []
    for audit_id in sorted(mapping):
        label_1 = reviews_1[audit_id]["review_label"]
        label_2 = reviews_2[audit_id]["review_label"]
        if not label_1 or not label_2:
            pending.append(audit_id)
            continue
        completed_pairs.append((label_1, label_2))
        needs_adjudication = (
            label_1 != label_2
            or label_1 == "unclear"
            or label_2 == "unclear"
        )
        if needs_adjudication:
            disagreements.append(audit_id)
            final_label = str(
                adjudications[audit_id].get("final_label") or ""
            ).strip().lower()
            if final_label not in FINAL_LABELS:
                pending.append(audit_id)
                continue
            final_labels[audit_id] = final_label
        else:
            final_labels[audit_id] = label_1

    labels_1 = [pair[0] for pair in completed_pairs]
    labels_2 = [pair[1] for pair in completed_pairs]
    kappa = _cohens_kappa(labels_1, labels_2)
    nonfaithful_ids = {
        audit_id for audit_id, label in final_labels.items() if label != "faithful"
    }
    probability_ids = {
        audit_id
        for audit_id, row in mapping.items()
        if (row.get("sample_groups") or {}).get("probability_sample")
    }
    risk_ids = {
        audit_id
        for audit_id, row in mapping.items()
        if (row.get("sample_groups") or {}).get("risk_census")
    }
    probability_false_accepts = len(nonfaithful_ids & probability_ids)
    risk_false_accepts = len(nonfaithful_ids & risk_ids)
    probability_sample_size = len(probability_ids)
    probability_review_complete = probability_ids.issubset(final_labels)
    observed_false_accept_rate = (
        probability_false_accepts / probability_sample_size
        if probability_sample_size and probability_review_complete
        else None
    )
    zero_error_upper_bound = (
        1.0 - math.pow(0.05, 1.0 / probability_sample_size)
        if (
            probability_sample_size
            and probability_review_complete
            and probability_false_accepts == 0
        )
        else None
    )
    review_complete = not pending and len(final_labels) == len(mapping)
    adjudication_complete = not disagreements or bool(adjudicator_id)
    gate_checks = {
        "review_complete": review_complete,
        "independent_reviewer_ids": True,
        "minimum_probability_sample_size": (
            probability_sample_size >= MIN_PROBABILITY_SAMPLE_SIZE
        ),
        "minimum_cohens_kappa": bool(
            kappa is not None and kappa >= MIN_REVIEWER_KAPPA
        ),
        "all_disagreements_adjudicated": (
            review_complete and adjudication_complete
        ),
        "zero_false_accepts_in_selected_items": (
            review_complete and not nonfaithful_ids
        ),
        "zero_false_accepts_in_risk_census": (
            review_complete and risk_false_accepts == 0
        ),
        "false_accept_upper_bound_at_most_one_percent": bool(
            zero_error_upper_bound is not None
            and zero_error_upper_bound <= MAX_FALSE_ACCEPT_RATE
        ),
    }
    approved = all(gate_checks.values())
    status = "approved" if approved else ("pending" if pending else "failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "approved_for_research_promotion": approved,
        "reviewers": {
            "reviewer_1": reviewer_1_id,
            "reviewer_2": reviewer_2_id,
            "adjudicator": adjudicator_id or None,
        },
        "counts": {
            "review_items": len(mapping),
            "completed_review_pairs": len(completed_pairs),
            "pending_items": len(set(pending)),
            "disagreements_or_unclear": len(disagreements),
            "nonfaithful_final_labels": len(nonfaithful_ids),
            "probability_sample_size": probability_sample_size,
            "probability_false_accepts": probability_false_accepts,
            "risk_census_size": len(risk_ids),
            "risk_false_accepts": risk_false_accepts,
        },
        "statistics": {
            "cohens_kappa": round(kappa, 8) if kappa is not None else None,
            "observed_probability_false_accept_rate": (
                round(observed_false_accept_rate, 8)
                if observed_false_accept_rate is not None
                else None
            ),
            "zero_error_one_sided_95_percent_upper_bound": (
                round(zero_error_upper_bound, 8)
                if zero_error_upper_bound is not None
                else None
            ),
        },
        "gate_checks": gate_checks,
        "methodology": {
            "ground_truth_access": False,
            "image_access": False,
            "scores_used_for_review": False,
            "review_question": (
                "Whether the extracted answer is a faithful final commitment "
                "in the raw response."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare or assess a blinded local-extractor audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--final-root",
        type=Path,
        default=project_root / "evaluation" / "results" / "final",
    )
    prepare.add_argument(
        "--staging-root",
        type=Path,
        default=(
            project_root
            / "evaluation"
            / "results"
            / "local_extractor_review"
        ),
    )
    prepare.add_argument(
        "--output-root",
        type=Path,
        default=(
            project_root
            / "evaluation"
            / "results"
            / "private"
            / "local_extractor_human_audit"
        ),
    )
    prepare.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    prepare.add_argument("--seed", default=DEFAULT_SEED)

    assess = subparsers.add_parser("assess")
    assess.add_argument(
        "--audit-root",
        type=Path,
        default=(
            project_root
            / "evaluation"
            / "results"
            / "private"
            / "local_extractor_human_audit"
        ),
    )
    assess.add_argument("--reviewer-1-id", required=True)
    assess.add_argument("--reviewer-2-id", required=True)
    assess.add_argument("--adjudicator-id", default="")
    assess.add_argument(
        "--project-root",
        type=Path,
        help="Override the project root recorded when the package was prepared.",
    )
    assess.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "evaluation"
            / "results"
            / "private"
            / "local_extractor_human_audit"
            / "assessment.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        report = prepare_audit(
            project_root=project_root,
            final_root=args.final_root,
            staging_root=args.staging_root,
            output_root=args.output_root,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "output_root": report["package_root"],
                    "status": report["status"],
                    "sampling": report["sampling"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    assessment = assess_audit(
        audit_root=args.audit_root,
        reviewer_1_id=args.reviewer_1_id,
        reviewer_2_id=args.reviewer_2_id,
        adjudicator_id=args.adjudicator_id,
        project_root=args.project_root,
    )
    _write_private_json(args.output.expanduser().resolve(), assessment)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["approved_for_research_promotion"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, OSError, ValueError) as exc:
        print(f"Local-extractor human audit failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
