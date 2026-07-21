#!/usr/bin/env python3
"""Normalize a combined model-output JSONL into upload-ready task files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scoring.task_scorer import SubmissionValidationError, TaskScorer


SUBSET_TO_TASK = {
    "dysm_2d_v1": "do_you_see_me",
    "dysm_3d_v1": "do_you_see_me",
    "minds_eye_fresh_v1": "minds_eye",
}
TASK_ORDER = ("do_you_see_me", "minds_eye")
ANSWER_FIELDS = ("answer", "prediction", "response", "output")
DEFAULT_NO_RESPONSE_TOKEN = "[NO MODEL RESPONSE]"


class NormalizationError(ValueError):
    """A source-file issue that must be corrected or explicitly acknowledged."""


def _scalar_answer(value: Any, *, line_number: int, question_id: str) -> str:
    if isinstance(value, (dict, list)):
        raise NormalizationError(
            f"Line {line_number} for question_id '{question_id}' has a structured "
            "answer. Expected a string, number, boolean, or null."
        )
    return "" if value is None else str(value).strip()


def _atomic_write(path: Path, content: str, *, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "unchanged"
        if not force:
            raise NormalizationError(
                f"Refusing to replace existing file '{path}'. Re-run with --force "
                "after reviewing the previous normalized output."
            )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return "written"


def normalize_file(
    source: Path,
    output_dir: Path,
    *,
    model_name: str,
    empty_policy: str,
    no_response_token: str,
    force: bool = False,
) -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise NormalizationError(f"Source JSONL file not found: {source}")
    if source.suffix.lower() != ".jsonl":
        raise NormalizationError("The source file must use the .jsonl extension.")
    if not model_name.strip():
        raise NormalizationError("Model name cannot be empty.")
    if empty_policy not in {"error", "incorrect"}:
        raise NormalizationError("empty_policy must be 'error' or 'incorrect'.")
    if not no_response_token.strip():
        raise NormalizationError("The no-response token cannot be empty.")

    source_bytes = source.read_bytes()
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NormalizationError(
            f"The source file is not valid UTF-8 text (byte offset {exc.start})."
        ) from exc

    task_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    blank_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subset_counts: Counter[str] = Counter()
    answer_field_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    parsed_rows = 0

    for line_number, raw_line in enumerate(source_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NormalizationError(
                f"Line {line_number} is not valid JSON ({exc.msg})."
            ) from exc
        if not isinstance(row, dict):
            raise NormalizationError(
                f"Line {line_number} must contain a JSON object, not "
                f"{type(row).__name__}."
            )

        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise NormalizationError(
                f"Line {line_number} is missing a non-empty question_id."
            )
        subset = str(row.get("subset") or "").strip()
        task_id = SUBSET_TO_TASK.get(subset)
        if task_id is None:
            allowed = ", ".join(sorted(SUBSET_TO_TASK))
            raise NormalizationError(
                f"Line {line_number} for question_id '{question_id}' uses unknown "
                f"subset '{subset}'. Expected one of: {allowed}."
            )

        condition = str(row.get("condition") or "standard").strip().lower()
        if condition != "standard":
            raise NormalizationError(
                f"Line {line_number} for question_id '{question_id}' uses condition "
                f"'{condition}'. These two visual benchmarks accept only 'standard'."
            )

        present_answer_fields = [field for field in ANSWER_FIELDS if field in row]
        if not present_answer_fields:
            raise NormalizationError(
                f"Line {line_number} for question_id '{question_id}' has no answer "
                f"field. Accepted fields are: {', '.join(ANSWER_FIELDS)}."
            )
        if len(present_answer_fields) > 1:
            raise NormalizationError(
                f"Line {line_number} for question_id '{question_id}' has multiple "
                f"answer fields: {', '.join(present_answer_fields)}."
            )
        answer_field = present_answer_fields[0]
        answer = _scalar_answer(
            row.get(answer_field), line_number=line_number, question_id=question_id
        )

        key = (task_id, question_id)
        if key in seen:
            raise NormalizationError(
                f"Line {line_number} repeats question_id '{question_id}' for "
                f"benchmark '{task_id}'."
            )
        seen.add(key)

        if not answer:
            blank_rows[task_id].append(
                {"line_number": line_number, "question_id": question_id}
            )
            if empty_policy == "incorrect":
                answer = no_response_token.strip()

        task_rows[task_id].append(
            {
                "question_id": question_id,
                "condition": "standard",
                "answer": answer,
            }
        )
        subset_counts[subset] += 1
        answer_field_counts[answer_field] += 1
        answer_type_counts[str(row.get("answer_type") or "unspecified")] += 1
        parsed_rows += 1

    if parsed_rows == 0:
        raise NormalizationError("The source file contains no JSONL records.")

    total_blank = sum(len(rows) for rows in blank_rows.values())
    if total_blank and empty_policy == "error":
        examples = [
            item
            for task_id in TASK_ORDER
            for item in blank_rows.get(task_id, [])
        ][:5]
        preview = ", ".join(
            f"{item['question_id']} (line {item['line_number']})" for item in examples
        )
        raise NormalizationError(
            f"The source contains {total_blank} empty model output(s), including "
            f"{preview}. Empty outputs are rejected by the upload API. Review the "
            "generation failures, or re-run with --empty-policy incorrect to replace "
            "them with an explicit no-response token that will score as incorrect."
        )

    file_operations = {}
    task_reports = {}
    for task_id in TASK_ORDER:
        rows = task_rows.get(task_id, [])
        if not rows:
            raise NormalizationError(
                f"The source contains no rows for required benchmark '{task_id}'."
            )
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
        scorer = TaskScorer(task_id)
        try:
            score = scorer.score_submission_text(content, model_name=model_name.strip())
        except SubmissionValidationError as exc:
            raise NormalizationError(
                f"{task_id} coverage validation failed [{exc.code}]: {exc}"
            ) from exc

        output_path = output_dir / f"{source.stem}__{task_id}.jsonl"
        write_status = _atomic_write(output_path, content, force=force)
        file_operations[task_id] = write_status
        task_reports[task_id] = {
            "accuracy": round(score.accuracy, 8),
            "blank_outputs_counted_incorrect": len(blank_rows.get(task_id, [])),
            "correct_samples": score.correct_samples,
            "file": output_path.name,
            "row_count": len(rows),
            "total_samples": score.total_samples,
        }

    report = {
        "answer_field_counts": dict(sorted(answer_field_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "canonical_schema": ["question_id", "condition", "answer"],
        "empty_policy": empty_policy,
        "model_name": model_name.strip(),
        "no_response_token": no_response_token.strip() if total_blank else None,
        "source_file": source.name,
        "source_row_count": parsed_rows,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "subset_counts": dict(sorted(subset_counts.items())),
        "tasks": task_reports,
        "total_blank_outputs_counted_incorrect": total_blank,
    }
    report_content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path = output_dir / f"{source.stem}__normalization_report.json"
    report_write_status = _atomic_write(
        report_path, report_content, force=force
    )
    return {
        **report,
        "file_operations": {
            **file_operations,
            "report": report_write_status,
        },
        "report_file": report_path.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split combined Do You See Me and Mind's Eye model outputs into "
            "canonical, upload-ready JSONL files and validate them with the real scorer."
        )
    )
    parser.add_argument("source", type=Path, help="Combined source JSONL file")
    parser.add_argument("--model-name", required=True, help="Leaderboard model name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: <source directory>/normalized)",
    )
    parser.add_argument(
        "--empty-policy",
        choices=("error", "incorrect"),
        default="error",
        help="Reject empty outputs, or explicitly count them as incorrect",
    )
    parser.add_argument(
        "--no-response-token",
        default=DEFAULT_NO_RESPONSE_TOKEN,
        help="Token used only when --empty-policy incorrect is selected",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a different existing normalized file after review",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir or args.source.parent / "normalized"
    try:
        report = normalize_file(
            args.source,
            output_dir,
            model_name=args.model_name,
            empty_policy=args.empty_policy,
            no_response_token=args.no_response_token,
            force=args.force,
        )
    except (NormalizationError, OSError) as exc:
        print(f"Normalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
