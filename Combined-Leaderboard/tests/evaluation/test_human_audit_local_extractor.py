import hashlib
import json
import stat
from pathlib import Path

import pytest

from evaluation.common.vllm_runner import LOCAL_ANSWER_EXTRACTION_METHOD
from evaluation.finalize_visual_results import FinalizationError, sha256
from evaluation.human_audit_local_extractor import (
    assess_audit,
    prepare_audit,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, monkeypatch, row_count: int = 300):
    project_root = tmp_path / "project"
    final_root = project_root / "evaluation" / "results" / "final"
    staging_root = project_root / "evaluation" / "results" / "staging"
    output_root = project_root / "evaluation" / "results" / "private" / "audit"
    (final_root / "index.json").parent.mkdir(parents=True, exist_ok=True)
    (final_root / "index.json").write_text(
        json.dumps({"models": [{"slug": "model-a", "model_id": "org/model-a"}]}),
        encoding="utf-8",
    )

    tracks = []
    total_recovered = 0
    for track in ("do_you_see_me", "minds_eye"):
        count = row_count if track == "do_you_see_me" else 0
        question_rows = [
            {
                "question_id": f"q{index:04d}",
                "question": "How many objects are visible?",
                "answer_type": "integer",
                "task": "visual_spatial",
                "difficulty": "easy",
                "image": "/private/image.png",
            }
            for index in range(max(1, count))
        ]
        _write_jsonl(
            project_root / "tasks" / track / "questions.jsonl",
            question_rows,
        )
        source_diagnostics = []
        staged_diagnostics = []
        source_submissions = []
        staged_submissions = []
        for index in range(count):
            question_id = f"q{index:04d}"
            output = "Final answer: 1"
            output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
            source = {
                "question_id": question_id,
                "output": output,
                "finish_reason": "stop",
            }
            staged = {
                **source,
                "answer_extraction_method": LOCAL_ANSWER_EXTRACTION_METHOD,
                "extracted_answer": "1",
                "extractor_finish_reason": "stop",
                "extractor_source_output_sha256": output_hash,
            }
            source_diagnostics.append(source)
            staged_diagnostics.append(staged)
            source_submissions.append(
                {
                    "question_id": question_id,
                    "condition": "standard",
                    "answer": "__INVALID_FORMAT__",
                }
            )
            staged_submissions.append(
                {
                    "question_id": question_id,
                    "condition": "standard",
                    "answer": "1",
                }
            )

        source_diagnostics_path = final_root / "model-a" / f"{track}.diagnostics.jsonl"
        staged_diagnostics_path = staging_root / "model-a" / f"{track}.diagnostics.jsonl"
        source_submission_path = final_root / "model-a" / f"{track}_submission.jsonl"
        staged_submission_path = staging_root / "model-a" / f"{track}_submission.jsonl"
        _write_jsonl(source_diagnostics_path, source_diagnostics)
        _write_jsonl(staged_diagnostics_path, staged_diagnostics)
        _write_jsonl(source_submission_path, source_submissions)
        _write_jsonl(staged_submission_path, staged_submissions)
        tracks.append(
            {
                "slug": "model-a",
                "track": track,
                "candidate_count": count,
                "recovered_count": count,
                "unresolved_count": 0,
                "source_diagnostics_sha256": sha256(source_diagnostics_path),
                "staged_diagnostics_sha256": sha256(staged_diagnostics_path),
                "source_submission_sha256": sha256(source_submission_path),
                "staged_submission_sha256": sha256(staged_submission_path),
            }
        )
        total_recovered += count

    report = {
        "source_final_root": str(final_root.resolve()),
        "staging_root": str(staging_root.resolve()),
        "recovered_count": total_recovered,
        "verification": {
            "status": "passed",
            "canonical_sources_unchanged": True,
        },
        "extractor": {
            "ground_truth_access": False,
            "image_access": False,
        },
        "tracks": tracks,
    }
    (staging_root / "extraction_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    monkeypatch.setattr(
        "evaluation.human_audit_local_extractor.verify_canonical_results",
        lambda *_args: None,
    )
    return project_root, final_root, staging_root, output_root


def _complete_reviews(path: Path, label: str = "faithful") -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["review_label"] = label
    _write_jsonl(path, rows)


def test_blinded_audit_package_can_pass_release_gate(tmp_path, monkeypatch):
    project_root, final_root, staging_root, output_root = _fixture(
        tmp_path, monkeypatch
    )
    manifest = prepare_audit(
        project_root=project_root,
        final_root=final_root,
        staging_root=staging_root,
        output_root=output_root,
        sample_size=300,
        seed="test-seed",
    )

    assert manifest["sampling"]["population_size"] == 300
    assert manifest["sampling"]["probability_sample_size"] == 300
    assert manifest["blinding"]["ground_truth_access"] is False
    reviewer_row = json.loads(
        (output_root / "reviewer_1.jsonl").read_text().splitlines()[0]
    )
    assert "slug" not in reviewer_row
    assert "track" not in reviewer_row
    assert "question_id" not in reviewer_row
    assert "image" not in reviewer_row
    assert "ground_truth" not in json.dumps(reviewer_row).lower()
    assert stat.S_IMODE((output_root / "reviewer_1.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700

    _complete_reviews(output_root / "reviewer_1.jsonl")
    _complete_reviews(output_root / "reviewer_2.jsonl")
    assessment = assess_audit(
        audit_root=output_root,
        reviewer_1_id="Reviewer A",
        reviewer_2_id="Reviewer B",
    )

    assert assessment["approved_for_research_promotion"] is True
    assert assessment["statistics"]["cohens_kappa"] == 1.0
    assert (
        assessment["statistics"][
            "zero_error_one_sided_95_percent_upper_bound"
        ]
        < 0.01
    )


def test_assessment_rejects_modified_review_content(tmp_path, monkeypatch):
    project_root, final_root, staging_root, output_root = _fixture(
        tmp_path, monkeypatch
    )
    prepare_audit(
        project_root=project_root,
        final_root=final_root,
        staging_root=staging_root,
        output_root=output_root,
        sample_size=300,
        seed="test-seed",
    )
    rows = [
        json.loads(line)
        for line in (output_root / "reviewer_1.jsonl").read_text().splitlines()
    ]
    rows[0]["candidate_response"] = "tampered"
    _write_jsonl(output_root / "reviewer_1.jsonl", rows)

    with pytest.raises(FinalizationError, match="Immutable review content changed"):
        assess_audit(
            audit_root=output_root,
            reviewer_1_id="Reviewer A",
            reviewer_2_id="Reviewer B",
        )


def test_assessment_rechecks_staged_source_hashes(tmp_path, monkeypatch):
    project_root, final_root, staging_root, output_root = _fixture(
        tmp_path, monkeypatch
    )
    prepare_audit(
        project_root=project_root,
        final_root=final_root,
        staging_root=staging_root,
        output_root=output_root,
        sample_size=300,
        seed="test-seed",
    )
    staged_submission = (
        staging_root / "model-a" / "do_you_see_me_submission.jsonl"
    )
    rows = [json.loads(line) for line in staged_submission.read_text().splitlines()]
    rows[0]["answer"] = "2"
    _write_jsonl(staged_submission, rows)

    with pytest.raises(FinalizationError, match="Locked artifact hash mismatch"):
        assess_audit(
            audit_root=output_root,
            reviewer_1_id="Reviewer A",
            reviewer_2_id="Reviewer B",
        )
