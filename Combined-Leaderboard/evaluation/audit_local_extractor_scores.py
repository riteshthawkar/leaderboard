"""Privately score locked local-extractor artifacts against ground truth.

Extraction and scoring intentionally remain separate phases. This command never
changes an extracted answer and never writes a ground-truth answer to its report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.common.visual_pipeline import INVALID_FORMAT_ANSWER
from evaluation.finalize_visual_results import (
    FinalizationError,
    read_json,
    read_jsonl,
    sha256,
    verify_canonical_results,
)


TRACKS = ("do_you_see_me", "minds_eye")
REPORT_SCHEMA_VERSION = 1


def _default_scorer_factory(project_root: Path) -> Callable[[str], Any]:
    backend_path = str((project_root / "backend").resolve())
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from scoring.task_scorer import TaskScorer

    return TaskScorer


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
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


def _score_summary(score: Any) -> dict[str, Any]:
    payload = score.to_dict()
    return {
        key: payload.get(key)
        for key in (
            "accuracy",
            "macro_accuracy",
            "task_spread",
            "random_baseline",
            "score_method",
            "total_samples",
            "correct_samples",
            "groups",
            "analysis",
            "grading",
        )
    }


def _locked_answer_changes(
    canonical_rows: list[dict[str, Any]],
    staged_rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str, str]], int]:
    if len(canonical_rows) != len(staged_rows):
        raise FinalizationError("Canonical and staged submission lengths differ.")
    changed: list[tuple[str, str, str, str]] = []
    invalid = 0
    for canonical, staged in zip(canonical_rows, staged_rows, strict=True):
        canonical_key = (
            str(canonical.get("question_id") or ""),
            str(canonical.get("condition") or "standard"),
        )
        staged_key = (
            str(staged.get("question_id") or ""),
            str(staged.get("condition") or "standard"),
        )
        if canonical_key != staged_key:
            raise FinalizationError(
                f"Canonical and staged row order differs at {staged_key}."
            )
        canonical_answer = str(canonical.get("answer") or "")
        staged_answer = str(staged.get("answer") or "")
        if canonical_answer != staged_answer:
            changed.append(
                (staged_key[0], staged_key[1], canonical_answer, staged_answer)
            )
        if staged.get("answer") == INVALID_FORMAT_ANSWER:
            invalid += 1
    return changed, invalid


def build_private_audit(
    *,
    project_root: Path,
    final_root: Path,
    staging_root: Path,
    scorer_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    final_root = final_root.resolve()
    staging_root = staging_root.resolve()
    verify_canonical_results(final_root, project_root)

    extraction_report_path = staging_root / "extraction_report.json"
    extraction_report_hash = sha256(extraction_report_path)
    extraction_report = read_json(extraction_report_path)
    verification = extraction_report.get("verification") or {}
    if verification.get("status") != "passed" or not verification.get(
        "canonical_sources_unchanged"
    ):
        raise FinalizationError(
            "The extraction report has not passed immutable-source verification."
        )
    if Path(str(extraction_report.get("source_final_root"))).resolve() != final_root:
        raise FinalizationError("Extraction report references a different final root.")
    if Path(str(extraction_report.get("staging_root"))).resolve() != staging_root:
        raise FinalizationError("Extraction report references a different staging root.")

    track_reports = {
        (str(row.get("slug")), str(row.get("track"))): row
        for row in extraction_report.get("tracks") or []
    }
    index = read_json(final_root / "index.json")
    models = list(index.get("models") or [])
    factory = scorer_factory or _default_scorer_factory(project_root)
    scorers = {track: factory(track) for track in TRACKS}

    ground_truth_inputs: dict[str, list[dict[str, Any]]] = {}
    for track, scorer in scorers.items():
        files = [Path(path).resolve() for path in scorer.resolved_ground_truth_files()]
        ground_truth_inputs[track] = [
            {
                "id": f"{path.parent.name}/{path.name}",
                "sha256": sha256(path),
                "mode": oct(path.stat().st_mode & 0o777),
            }
            for path in files
        ]

    locked_inputs: dict[Path, str] = {
        extraction_report_path: extraction_report_hash,
        final_root / "index.json": sha256(final_root / "index.json"),
    }
    results: list[dict[str, Any]] = []
    total_changed = 0
    total_staged_correct = 0
    total_canonical_correct = 0
    total_scored_samples = 0
    total_changed_outcomes = {
        "improved": 0,
        "regressed": 0,
        "correct_both": 0,
        "incorrect_both": 0,
    }

    for model in models:
        slug = str(model.get("slug") or "")
        model_id = str(model.get("model_id") or slug)
        if not slug:
            raise FinalizationError("Canonical index contains a model without a slug.")
        for track in TRACKS:
            track_report = track_reports.get((slug, track))
            if track_report is None:
                raise FinalizationError(
                    f"Extraction report is missing {slug}/{track}."
                )
            canonical_path = final_root / slug / f"{track}_submission.jsonl"
            staged_path = staging_root / slug / f"{track}_submission.jsonl"
            canonical_hash = sha256(canonical_path)
            staged_hash = sha256(staged_path)
            if canonical_hash != track_report.get("source_submission_sha256"):
                raise FinalizationError(
                    f"Canonical submission hash mismatch for {slug}/{track}."
                )
            if staged_hash != track_report.get("staged_submission_sha256"):
                raise FinalizationError(
                    f"Staged submission hash mismatch for {slug}/{track}."
                )
            locked_inputs[canonical_path] = canonical_hash
            locked_inputs[staged_path] = staged_hash

            changed_rows, invalid = _locked_answer_changes(
                read_jsonl(canonical_path), read_jsonl(staged_path)
            )
            canonical_score = scorers[track].score(canonical_path, model_id)
            staged_score = scorers[track].score(staged_path, model_id)
            canonical_summary = _score_summary(canonical_score)
            staged_summary = _score_summary(staged_score)
            correct_delta = int(staged_score.correct_samples) - int(
                canonical_score.correct_samples
            )
            changed_outcomes = {
                "improved": 0,
                "regressed": 0,
                "correct_both": 0,
                "incorrect_both": 0,
            }
            for question_id, condition, canonical_answer, staged_answer in changed_rows:
                canonical_correct = bool(
                    scorers[track]._grade_condition(
                        question_id, canonical_answer, condition
                    )
                )
                staged_correct = bool(
                    scorers[track]._grade_condition(
                        question_id, staged_answer, condition
                    )
                )
                if staged_correct and not canonical_correct:
                    outcome = "improved"
                elif canonical_correct and not staged_correct:
                    outcome = "regressed"
                elif staged_correct:
                    outcome = "correct_both"
                else:
                    outcome = "incorrect_both"
                changed_outcomes[outcome] += 1
                total_changed_outcomes[outcome] += 1

            total_changed += len(changed_rows)
            total_staged_correct += int(staged_score.correct_samples)
            total_canonical_correct += int(canonical_score.correct_samples)
            total_scored_samples += int(staged_score.total_samples)
            results.append(
                {
                    "slug": slug,
                    "model_id": model_id,
                    "track": track,
                    "canonical_submission_sha256": canonical_hash,
                    "staged_submission_sha256": staged_hash,
                    "changed_answers": len(changed_rows),
                    "changed_answer_outcomes": changed_outcomes,
                    "staged_invalid_answers": invalid,
                    "candidate_count": int(track_report.get("candidate_count") or 0),
                    "recovered_count": int(track_report.get("recovered_count") or 0),
                    "unresolved_count": int(track_report.get("unresolved_count") or 0),
                    "canonical_score": canonical_summary,
                    "staged_score": staged_summary,
                    "correct_samples_delta": correct_delta,
                    "macro_accuracy_delta": round(
                        float(staged_score.macro_accuracy or 0.0)
                        - float(canonical_score.macro_accuracy or 0.0),
                        10,
                    ),
                }
            )

    for path, expected_hash in locked_inputs.items():
        if sha256(path) != expected_hash:
            raise FinalizationError(f"Locked input changed during scoring: {path}")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "extraction_completed_before_ground_truth_loading": True,
            "extractor_ground_truth_access": False,
            "extractor_image_access": False,
            "scoring_backend": "deterministic-private-ground-truth",
            "scoring_can_modify_extracted_answers": False,
            "ground_truth_answers_written_to_report": False,
            "promotion_performed": False,
        },
        "extractor": extraction_report.get("extractor"),
        "extraction_validation": extraction_report.get("validation"),
        "extraction_report_sha256": extraction_report_hash,
        "ground_truth_inputs": ground_truth_inputs,
        "summary": {
            "model_count": len(models),
            "track_count": len(results),
            "total_scored_samples": total_scored_samples,
            "changed_answers": total_changed,
            "changed_answer_outcomes": total_changed_outcomes,
            "canonical_correct_samples": total_canonical_correct,
            "staged_correct_samples": total_staged_correct,
            "correct_samples_delta": total_staged_correct - total_canonical_correct,
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Privately score locked local-extractor staging artifacts without "
            "exposing ground-truth answers or changing extracted responses."
        )
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=project_root / "evaluation" / "results" / "final",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=project_root / "evaluation" / "results" / "local_extractor_review",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "evaluation"
            / "results"
            / "private"
            / "local_extractor_score_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = build_private_audit(
        project_root=project_root,
        final_root=args.final_root,
        staging_root=args.staging_root,
    )
    output = args.output.expanduser().resolve()
    _write_private_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": oct(output.stat().st_mode & 0o777),
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, OSError, ValueError) as exc:
        print(f"Private extraction audit failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
