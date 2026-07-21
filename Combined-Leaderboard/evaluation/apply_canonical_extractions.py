"""Apply a completed gold-blind commitment audit to a new canonical result copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.common.visual_pipeline import final_answer
from evaluation.extract_canonical_answers import (
    GroundTruthError,
    METHOD,
    candidate_key,
    commitment_verdict,
    evidence_supports,
    load_gold_answers,
    load_candidates,
    valid_answer,
)
from evaluation.finalize_visual_results import (
    TRACKS,
    answer_provenance_counts,
    artifact_record,
    read_json,
    read_jsonl,
    sha256,
    validate_diagnostics,
    validate_submission,
    verify_canonical_results,
)


ACCEPTED_STATUSES = {"gold_committed", "other_committed"}
PRIOR_EXTRACTOR_FIELDS = (
    "answer_extraction_method",
    "extractor_model",
    "extractor_output",
    "extractor_finish_reason",
    "extractor_completion_tokens",
    "extractor_error",
    "extractor_source_diagnostics",
    "extractor_source_output_sha256",
    "extractor_evidence",
    "extractor_commitment_verdict",
    "extractor_ground_truth_sha256",
    "extractor_contract_sha256",
    "ground_truth_supplied",
    "ground_truth_available_to_validator",
    "ground_truth_loaded_by_extractor_process",
    "ground_truth_supplied_to_extractor",
    "independent_extraction_status",
    "extracted_answer",
)


class ApplyExtractionError(RuntimeError):
    pass


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def load_completed_audit(
    audit_path: Path,
    expected: list[dict[str, Any]],
    ground_truth_sha256: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    expected_keys = {candidate_key(item) for item in expected}
    audit = {}
    for row in read_jsonl(audit_path):
        key = candidate_key(row)
        if key in audit:
            raise ApplyExtractionError(f"Audit repeats candidate {key}.")
        if row.get("method") != METHOD:
            raise ApplyExtractionError(f"Audit candidate {key} uses an incompatible method.")
        if row.get("ground_truth_sha256") != ground_truth_sha256:
            raise ApplyExtractionError(f"Audit candidate {key} uses different ground truth.")
        if row.get("ground_truth_available_to_validator") is not True:
            raise ApplyExtractionError(
                f"Audit candidate {key} lacks post-validation provenance."
            )
        if row.get("ground_truth_supplied_to_extractor") is not False:
            raise ApplyExtractionError(
                f"Audit candidate {key} is not from a gold-blind extractor."
            )
        if row.get("ground_truth_loaded_by_extractor_process") is not False:
            raise ApplyExtractionError(
                f"Audit candidate {key} is not process-isolated from ground truth."
            )
        contract_sha256 = str(row.get("extractor_contract_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", contract_sha256):
            raise ApplyExtractionError(
                f"Audit candidate {key} lacks an extractor contract hash."
            )
        audit[key] = row
    missing = expected_keys - set(audit)
    extra = set(audit) - expected_keys
    if missing or extra:
        raise ApplyExtractionError(
            f"Audit is incomplete or mismatched: {len(missing)} missing, "
            f"{len(extra)} unexpected records."
        )
    return audit


def apply_result(
    diagnostic: dict[str, Any],
    submission: dict[str, Any],
    audit: dict[str, Any],
    gold_answer: str,
    ground_truth_sha256: str,
) -> str | None:
    if audit.get("status") not in ACCEPTED_STATUSES:
        return None
    output = str(diagnostic.get("output") or "")
    output_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
    if output_sha256 != audit.get("response_sha256"):
        raise ApplyExtractionError(
            f"Response hash changed for {diagnostic.get('question_id')}."
        )
    if audit.get("method") != METHOD:
        raise ApplyExtractionError(
            f"Extraction method changed for {diagnostic.get('question_id')}."
        )
    if audit.get("ground_truth_sha256") != ground_truth_sha256:
        raise ApplyExtractionError(
            f"Ground truth changed for {diagnostic.get('question_id')}."
        )
    if audit.get("ground_truth_supplied_to_extractor") is not False:
        raise ApplyExtractionError(
            f"Ground truth was exposed to the extractor for "
            f"{diagnostic.get('question_id')}."
        )
    if audit.get("ground_truth_loaded_by_extractor_process") is not False:
        raise ApplyExtractionError(
            f"Extractor process loaded ground truth for "
            f"{diagnostic.get('question_id')}."
        )
    contract_sha256 = str(audit.get("extractor_contract_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", contract_sha256):
        raise ApplyExtractionError(
            f"Extractor contract is missing for {diagnostic.get('question_id')}."
        )
    answer_type = str(audit["answer_type"])
    answer = final_answer(audit.get("answer"), answer_type)
    evidence = str(audit.get("evidence") or "")
    if not answer or not valid_answer(answer, answer_type):
        raise ApplyExtractionError(
            f"Accepted audit has invalid answer for {diagnostic.get('question_id')}."
        )
    expected_verdict = commitment_verdict(answer, gold_answer, answer_type)
    if audit.get("verdict") != expected_verdict:
        raise ApplyExtractionError(
            f"Commitment verdict disagrees with ground truth for "
            f"{diagnostic.get('question_id')}."
        )
    if audit.get("status") != expected_verdict.casefold():
        raise ApplyExtractionError(
            f"Commitment status is inconsistent for {diagnostic.get('question_id')}."
        )
    if not evidence_supports(output, evidence, answer, answer_type):
        return None
    if expected_verdict == "GOLD_COMMITTED":
        answer = final_answer(gold_answer, answer_type)

    prior = {
        field: diagnostic[field]
        for field in PRIOR_EXTRACTOR_FIELDS
        if field in diagnostic
    }
    attempts = list(diagnostic.get("extractor_attempts", []))
    if prior:
        attempts.append(prior)
    for field in PRIOR_EXTRACTOR_FIELDS:
        diagnostic.pop(field, None)
    diagnostic["answer_extraction_method"] = METHOD
    diagnostic["extractor_model"] = str(audit["extractor_model"])
    diagnostic["extractor_output"] = str(audit.get("extractor_output") or "")
    diagnostic["extractor_evidence"] = evidence
    diagnostic["extractor_commitment_verdict"] = expected_verdict
    diagnostic["extractor_ground_truth_sha256"] = ground_truth_sha256
    diagnostic["extractor_contract_sha256"] = contract_sha256
    diagnostic["ground_truth_available_to_validator"] = True
    diagnostic["ground_truth_loaded_by_extractor_process"] = False
    diagnostic["ground_truth_supplied_to_extractor"] = False
    diagnostic["extractor_finish_reason"] = audit.get("finish_reason")
    diagnostic["extractor_completion_tokens"] = audit.get("completion_tokens")
    diagnostic["extractor_source_output_sha256"] = output_sha256
    diagnostic["extracted_answer"] = answer
    diagnostic["independent_extraction_status"] = str(audit["status"])
    if attempts:
        diagnostic["extractor_attempts"] = attempts
    submission["answer"] = answer
    return str(audit["status"])


def apply_completed_audit(
    project_root: Path,
    canonical_root: Path,
    audit_path: Path,
    output_root: Path,
    policy: str,
    ground_truth_paths: list[Path],
) -> dict[str, Any]:
    project_root = project_root.resolve()
    canonical_root = canonical_root.resolve()
    output_root = output_root.resolve()
    gold_answers, ground_truth_sha256 = load_gold_answers(
        project_root, ground_truth_paths
    )
    expected = load_candidates(project_root, canonical_root, policy, gold_answers)
    audit = load_completed_audit(audit_path, expected, ground_truth_sha256)
    expected_ids = {
        track: [
            str(row["question_id"])
            for row in read_jsonl(project_root / "tasks" / track / "questions.jsonl")
        ]
        for track in TRACKS
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    shutil.rmtree(staging)
    shutil.copytree(canonical_root, staging)
    summary = Counter()
    by_model_track: dict[tuple[str, str], Counter] = defaultdict(Counter)
    audit_sha256 = sha256(audit_path)
    try:
        index_path = staging / "index.json"
        index = read_json(index_path)
        index_by_slug = {str(item["slug"]): item for item in index["models"]}
        for model_dir in sorted(path for path in staging.iterdir() if path.is_dir()):
            manifest_path = model_dir / "final_manifest.json"
            manifest = read_json(manifest_path)
            model_changed = False
            for track in TRACKS:
                diagnostics_path = model_dir / f"{track}.diagnostics.jsonl"
                submission_path = model_dir / f"{track}_submission.jsonl"
                diagnostics = read_jsonl(diagnostics_path)
                submissions = read_jsonl(submission_path)
                diagnostics_by_id = {
                    str(row["question_id"]): row for row in diagnostics
                }
                submissions_by_id = {
                    str(row["question_id"]): row for row in submissions
                }
                track_audit = []
                for key, result in audit.items():
                    if key[0] != model_dir.name or key[1] != track:
                        continue
                    track_audit.append(result)
                    question_id = key[2]
                    applied = apply_result(
                        diagnostics_by_id[question_id],
                        submissions_by_id[question_id],
                        result,
                        gold_answers[question_id],
                        ground_truth_sha256,
                    )
                    if applied:
                        summary[applied] += 1
                        by_model_track[(model_dir.name, track)][applied] += 1
                        model_changed = True
                if not track_audit:
                    continue
                write_jsonl(diagnostics_path, diagnostics)
                write_jsonl(submission_path, submissions)
                audit_copy = model_dir / f"{track}.independent_extraction.jsonl"
                write_jsonl(audit_copy, track_audit)
                submission_rows = validate_submission(
                    submission_path, expected_ids[track]
                )
                _rows, diagnostics_by_id = validate_diagnostics(
                    diagnostics_path, expected_ids[track]
                )
                strict_count, exact_raw_count = answer_provenance_counts(
                    submission_rows, diagnostics_by_id
                )
                track_record = manifest["tracks"][track]
                track_record["strict_answer_count"] = strict_count
                track_record["exact_raw_output_fallback_count"] = exact_raw_count
                for artifact in (diagnostics_path, submission_path, audit_copy):
                    track_record["artifacts"][artifact.name] = artifact_record(artifact)
                track_record["independent_answer_extraction"] = {
                    "method": METHOD,
                    "extractor_model": next(
                        (str(item["extractor_model"]) for item in track_audit), ""
                    ),
                    "audit_sha256": audit_sha256,
                    "extractor_contract_sha256": next(
                        (
                            str(item["extractor_contract_sha256"])
                            for item in track_audit
                        ),
                        "",
                    ),
                    "candidate_count": len(track_audit),
                    "status_counts": dict(
                        Counter(str(item["status"]) for item in track_audit)
                    ),
                    "applied_counts": dict(by_model_track[(model_dir.name, track)]),
                    "input_fields": [
                        "question",
                        "answer_type",
                        "candidate_response",
                    ],
                    "image_supplied": False,
                    "ground_truth_available_to_validator": True,
                    "ground_truth_loaded_by_extractor_process": False,
                    "ground_truth_supplied_to_extractor": False,
                    "ground_truth_sha256": ground_truth_sha256,
                    "evidence_requirement": "literal-source-quote-and-commitment-validation",
                }
            if model_changed or any(
                manifest["tracks"][track].get("independent_answer_extraction")
                for track in TRACKS
            ):
                manifest["independent_answer_extraction"] = {
                    "method": METHOD,
                    "source_audit_sha256": audit_sha256,
                    "ground_truth_sha256": ground_truth_sha256,
                    "ground_truth_available_to_validator": True,
                    "ground_truth_loaded_by_extractor_process": False,
                    "ground_truth_supplied_to_extractor": False,
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                index_record = index_by_slug[model_dir.name]
                index_record["manifest_sha256"] = sha256(manifest_path)
                index_record["tracks"] = {
                    track: {
                        "row_count": manifest["tracks"][track]["row_count"],
                        "strict_answer_count": manifest["tracks"][track][
                            "strict_answer_count"
                        ],
                        "exact_raw_output_fallback_count": manifest["tracks"][track][
                            "exact_raw_output_fallback_count"
                        ],
                    }
                    for track in TRACKS
                }
        index["independent_answer_extraction"] = {
            "method": METHOD,
            "source_audit_sha256": audit_sha256,
            "ground_truth_sha256": ground_truth_sha256,
            "ground_truth_available_to_validator": True,
            "ground_truth_loaded_by_extractor_process": False,
            "ground_truth_supplied_to_extractor": False,
            "candidate_count": len(expected),
            "applied_counts": dict(summary),
        }
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verification = verify_canonical_results(staging, project_root)
        if output_root.exists():
            raise ApplyExtractionError(f"Output root already exists: {output_root}")
        os.replace(staging, output_root)
        return {
            "candidate_count": len(expected),
            "applied_counts": dict(summary),
            "output_root": str(output_root),
            "verification": verification,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--canonical-root", type=Path, default=project_root / "evaluation/results/final"
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        action="append",
        dest="ground_truth_paths",
        type=Path,
        required=True,
        help=(
            "JSON/JSONL answer map; repeat as needed. Combined files must exactly "
            "cover all canonical visual question IDs."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("unresolved", "high_risk", "all_nonexact"),
        default="high_risk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = apply_completed_audit(
            args.project_root,
            args.canonical_root,
            args.audit,
            args.output_root,
            args.policy,
            args.ground_truth_paths,
        )
    except (ApplyExtractionError, GroundTruthError) as exc:
        raise SystemExit(f"Extraction apply failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()