#!/usr/bin/env python3
"""Audit raw inference and fixed-extractor provenance before research scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common.vllm_runner import ANSWER_EXTRACTION_METHOD  # noqa: E402
from evaluation.common.visual_pipeline import MISSING_ANSWER_TOKEN  # noqa: E402
from evaluation.research.common import describe_path, read_jsonl, write_json  # noqa: E402


def index_rows(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"{label} contains a row without question_id: {path}")
        if question_id in indexed:
            raise ValueError(f"{label} contains duplicate question_id {question_id}")
        indexed[question_id] = row
    return indexed


def token_summary(values: list[int]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def audit(
    *,
    questions_path: Path,
    diagnostics_path: Path,
    submission_path: Path,
    output: Path,
    extractor_model: str,
    extractor_revision: str,
) -> dict[str, Any]:
    questions = index_rows(questions_path, label="questions")
    diagnostics = index_rows(diagnostics_path, label="diagnostics")
    submission = index_rows(submission_path, label="submission")
    expected_ids = set(questions)

    issues: list[dict[str, str]] = []
    for label, rows in (("diagnostics", diagnostics), ("submission", submission)):
        for question_id in sorted(expected_ids - set(rows)):
            issues.append(
                {
                    "question_id": question_id,
                    "category": f"missing_{label}",
                    "detail": f"{label} does not contain the expected question",
                }
            )
        for question_id in sorted(set(rows) - expected_ids):
            issues.append(
                {
                    "question_id": question_id,
                    "category": f"unknown_{label}",
                    "detail": f"{label} contains an unknown question",
                }
            )

    item_rows: list[dict[str, Any]] = []
    finish_reasons: Counter[str] = Counter()
    extractor_statuses: Counter[str] = Counter()
    completion_tokens: list[int] = []
    unresolved_by_finish: Counter[str] = Counter()
    for question_id in sorted(expected_ids & set(diagnostics) & set(submission)):
        diagnostic = diagnostics[question_id]
        prediction = submission[question_id]
        raw_output = str(diagnostic.get("output") or "")
        finish_reason = str(diagnostic.get("finish_reason") or "missing")
        extractor_status = str(diagnostic.get("extractor_status") or "missing")
        extracted_answer = str(diagnostic.get("extracted_answer") or "")
        submitted_answer = str(prediction.get("answer") or "")
        item_issues: list[str] = []

        finish_reasons[finish_reason] += 1
        extractor_statuses[extractor_status] += 1
        if diagnostic.get("completion_tokens") is not None:
            completion_tokens.append(int(diagnostic["completion_tokens"]))
        if not raw_output.strip():
            item_issues.append("empty_model_output")
        if diagnostic.get("error") or diagnostic.get("inference_error"):
            item_issues.append("inference_error")
        if finish_reason not in {"stop", "length"}:
            item_issues.append("invalid_finish_reason")
        if diagnostic.get("answer_extraction_method") != ANSWER_EXTRACTION_METHOD:
            item_issues.append("extractor_contract_mismatch")
        if diagnostic.get("extractor_model") != extractor_model:
            item_issues.append("extractor_model_mismatch")
        if diagnostic.get("extractor_revision") != extractor_revision:
            item_issues.append("extractor_revision_mismatch")
        source_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        if diagnostic.get("extractor_source_output_sha256") != source_hash:
            item_issues.append("extractor_source_hash_mismatch")
        if extractor_status not in {"resolved", "unresolved"}:
            item_issues.append("extractor_failed")
        elif extractor_status == "resolved":
            if not extracted_answer or extracted_answer == MISSING_ANSWER_TOKEN:
                item_issues.append("resolved_answer_missing")
        elif extracted_answer != MISSING_ANSWER_TOKEN:
            item_issues.append("unresolved_token_mismatch")
        if submitted_answer != extracted_answer:
            item_issues.append("submission_extractor_mismatch")
        if prediction.get("condition") != "standard":
            item_issues.append("submission_condition_mismatch")
        if extracted_answer == MISSING_ANSWER_TOKEN:
            unresolved_by_finish[finish_reason] += 1

        for category in item_issues:
            issues.append(
                {
                    "question_id": question_id,
                    "category": category,
                    "detail": category.replace("_", " "),
                }
            )
        item_rows.append(
            {
                "question_id": question_id,
                "finish_reason": finish_reason,
                "completion_tokens": diagnostic.get("completion_tokens", ""),
                "extractor_status": extractor_status,
                "answer": submitted_answer,
                "quality_gate_passed": int(not item_issues),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "item_quality.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "question_id",
            "finish_reason",
            "completion_tokens",
            "extractor_status",
            "answer",
            "quality_gate_passed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item_rows)

    issue_counts = Counter(item["category"] for item in issues)
    summary = {
        "schema_version": "ms-vista-inference-quality-v1",
        "quality_gate_passed": not issues,
        "question_count": len(questions),
        "diagnostics_count": len(diagnostics),
        "submission_count": len(submission),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "completion_tokens": token_summary(completion_tokens),
        "extractor_status_counts": dict(sorted(extractor_statuses.items())),
        "unresolved_by_finish_reason": dict(sorted(unresolved_by_finish.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_preview": issues[:50],
        "inputs": {
            "questions": describe_path(questions_path),
            "diagnostics": describe_path(diagnostics_path),
            "submission": describe_path(submission_path),
        },
        "extractor": {
            "model": extractor_model,
            "revision": extractor_revision,
            "method": ANSWER_EXTRACTION_METHOD,
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--extractor-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit(
        questions_path=args.questions.resolve(),
        diagnostics_path=args.diagnostics.resolve(),
        submission_path=args.submission.resolve(),
        output=args.output.resolve(),
        extractor_model=args.extractor_model,
        extractor_revision=args.extractor_revision,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
