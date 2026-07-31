#!/usr/bin/env python3
"""Score persistently unparseable extractor responses as audited unresolved rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common.vllm_runner import (  # noqa: E402
    ANSWER_EXTRACTION_METHOD,
)
from evaluation.common.visual_pipeline import MISSING_ANSWER_TOKEN  # noqa: E402
from evaluation.research.common import (  # noqa: E402
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


POLICY = "persistent-unparseable-extractor-output-v1"
UNPARSEABLE_ERROR = "The extractor did not return a parseable answer block."


def _index_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"{label} contains a row without question_id")
        if question_id in indexed:
            raise ValueError(f"{label} contains duplicate question_id {question_id}")
        indexed[question_id] = row
    return indexed


def finalize_persistent_failures(
    *,
    diagnostics_path: Path,
    submission_path: Path,
    report_path: Path,
    extractor_model: str,
    extractor_revision: str,
    attempts: int,
) -> dict[str, Any]:
    if attempts < 2:
        raise ValueError("Persistent extractor finalization requires at least two attempts")

    diagnostics_rows = read_jsonl(diagnostics_path)
    submission_rows = read_jsonl(submission_path)
    diagnostics = _index_rows(diagnostics_rows, label="diagnostics")
    submissions = _index_rows(submission_rows, label="submission")
    if set(diagnostics) != set(submissions):
        raise ValueError("Diagnostics and submission question IDs do not match")

    diagnostics_before = sha256_file(diagnostics_path)
    submission_before = sha256_file(submission_path)
    finalized_ids: list[str] = []
    blockers: list[str] = []

    for question_id, diagnostic in diagnostics.items():
        if diagnostic.get("extractor_status") != "failed":
            continue
        reasons = []
        raw_output = str(diagnostic.get("output") or "")
        extractor_output = str(diagnostic.get("extractor_output") or "")
        if diagnostic.get("extractor_error") != UNPARSEABLE_ERROR:
            reasons.append("failure is not a completed unparseable extractor response")
        if not raw_output.strip():
            reasons.append("model output is empty")
        if not extractor_output.strip():
            reasons.append("extractor output is empty")
        if diagnostic.get("finish_reason") not in {"stop", "length"}:
            reasons.append("model finish reason is not complete")
        if diagnostic.get("extractor_finish_reason") not in {"stop", "length"}:
            reasons.append("extractor finish reason is not complete")
        if diagnostic.get("answer_extraction_method") != ANSWER_EXTRACTION_METHOD:
            reasons.append("extractor contract does not match")
        if diagnostic.get("extractor_model") != extractor_model:
            reasons.append("extractor model does not match")
        if diagnostic.get("extractor_revision") != extractor_revision:
            reasons.append("extractor revision does not match")
        source_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        if diagnostic.get("extractor_source_output_sha256") != source_hash:
            reasons.append("extractor source hash does not match")
        if reasons:
            blockers.append(f"{question_id}: {', '.join(reasons)}")
            continue

        diagnostic["extractor_status"] = "unresolved"
        diagnostic["extracted_answer"] = MISSING_ANSWER_TOKEN
        diagnostic["extractor_terminal_resolution"] = {
            "policy": POLICY,
            "attempts": attempts,
            "disposition": "unresolved_scored_incorrect",
            "original_status": "failed",
            "original_error": UNPARSEABLE_ERROR,
        }
        submissions[question_id]["answer"] = MISSING_ANSWER_TOKEN
        finalized_ids.append(question_id)

    if blockers:
        raise ValueError(
            "Cannot conservatively finalize extractor failures:\n- "
            + "\n- ".join(blockers)
        )
    if not finalized_ids:
        raise ValueError("No eligible persistent extractor failures were found")

    write_jsonl(diagnostics_path, diagnostics_rows)
    write_jsonl(submission_path, submission_rows)
    report = {
        "schema_version": "ms-vista-extractor-terminal-resolution-v1",
        "policy": POLICY,
        "attempts": attempts,
        "extractor": {
            "model": extractor_model,
            "revision": extractor_revision,
            "method": ANSWER_EXTRACTION_METHOD,
        },
        "disposition": "unresolved_scored_incorrect",
        "finalized_count": len(finalized_ids),
        "question_ids": sorted(finalized_ids),
        "artifacts": {
            "diagnostics": {
                "before_sha256": diagnostics_before,
                "after_sha256": sha256_file(diagnostics_path),
            },
            "submission": {
                "before_sha256": submission_before,
                "after_sha256": sha256_file(submission_path),
            },
        },
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--extractor-revision", required=True)
    parser.add_argument("--attempts", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = finalize_persistent_failures(
        diagnostics_path=args.diagnostics.resolve(),
        submission_path=args.submission.resolve(),
        report_path=args.report.resolve(),
        extractor_model=args.extractor_model,
        extractor_revision=args.extractor_revision,
        attempts=args.attempts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
