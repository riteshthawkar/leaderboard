"""Finalize complete visual diagnostics after deterministic format failures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .visual_pipeline import (
    EvaluationPipelineError,
    STANDARD_CONDITION,
    SUPPORTED_ANSWER_TYPES,
    final_answer,
    read_diagnostics,
)


INVALID_MODEL_RESPONSE = "__INVALID_MODEL_RESPONSE__"
UNPARSEABLE_ANSWER_POLICY = "sentinel-after-exhausted-retries-v1"


def _questions(path: Path) -> list[dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationPipelineError(f"Question bundle not found: {resolved}")

    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} is invalid JSON ({exc.msg})."
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} must contain a JSON object."
            )
        question_id = str(row.get("question_id") or "").strip()
        answer_type = str(row.get("answer_type") or "text").strip().lower()
        if not question_id:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} is missing question_id."
            )
        if question_id in seen:
            raise EvaluationPipelineError(
                f"Question bundle line {line_number} repeats question_id '{question_id}'."
            )
        if answer_type not in SUPPORTED_ANSWER_TYPES:
            raise EvaluationPipelineError(
                f"Question '{question_id}' uses unsupported answer_type '{answer_type}'."
            )
        seen.add(question_id)
        questions.append({"question_id": question_id, "answer_type": answer_type})

    if not questions:
        raise EvaluationPipelineError("The question bundle contains no questions.")
    return questions


def _atomic_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, resolved)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def finalize_diagnostics(
    diagnostics_path: Path,
    questions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export complete diagnostics, scoring persistent format failures as incorrect."""
    questions = _questions(questions_path)
    expected_ids = [item["question_id"] for item in questions]
    expected_set = set(expected_ids)
    records = read_diagnostics([diagnostics_path])

    records_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        if question_id in records_by_id:
            duplicate_ids.append(question_id)
        records_by_id[question_id] = record

    received_set = set(records_by_id)
    missing = [question_id for question_id in expected_ids if question_id not in received_set]
    unknown = sorted(received_set - expected_set)
    if duplicate_ids or missing or unknown:
        details = []
        if duplicate_ids:
            details.append(f"duplicate IDs: {', '.join(sorted(set(duplicate_ids))[:5])}")
        if missing:
            details.append(f"{len(missing)} missing IDs, including {', '.join(missing[:5])}")
        if unknown:
            details.append(f"{len(unknown)} unknown IDs, including {', '.join(unknown[:5])}")
        raise EvaluationPipelineError(
            "Cannot finalize diagnostics because coverage is invalid: "
            + "; ".join(details)
            + "."
        )

    inference_errors = [
        question_id
        for question_id in expected_ids
        if records_by_id[question_id].get("error")
    ]
    if inference_errors:
        raise EvaluationPipelineError(
            "Cannot finalize diagnostics with infrastructure or inference errors: "
            f"{len(inference_errors)} affected response(s), including "
            f"{', '.join(inference_errors[:5])}. Retry those requests first."
        )

    rows: list[dict[str, str]] = []
    unparseable_ids: list[str] = []
    for question in questions:
        question_id = question["question_id"]
        answer = final_answer(
            records_by_id[question_id].get("output"), question["answer_type"]
        )
        if not answer:
            answer = INVALID_MODEL_RESPONSE
            unparseable_ids.append(question_id)
        rows.append(
            {
                "question_id": question_id,
                "condition": STANDARD_CONDITION,
                "answer": answer,
            }
        )

    _atomic_jsonl(output_path, rows)
    return {
        "condition": STANDARD_CONDITION,
        "invalid_answer_sentinel": INVALID_MODEL_RESPONSE,
        "output_path": str(Path(output_path).expanduser().resolve()),
        "policy": UNPARSEABLE_ANSWER_POLICY,
        "row_count": len(rows),
        "schema": ["question_id", "condition", "answer"],
        "unparseable_count": len(unparseable_ids),
        "unparseable_question_ids_preview": unparseable_ids[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a complete visual submission after all deterministic retries, "
            "marking persistent unparseable model outputs as incorrect."
        )
    )
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = finalize_diagnostics(args.diagnostics, args.questions, args.out)
    except (EvaluationPipelineError, OSError) as exc:
        print(f"Finalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
