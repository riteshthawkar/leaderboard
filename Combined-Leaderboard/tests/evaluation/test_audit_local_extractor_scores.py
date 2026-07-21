import json
import stat
from pathlib import Path

from evaluation.audit_local_extractor_scores import (
    _write_private_json,
    build_private_audit,
)
from evaluation.finalize_visual_results import sha256


class _FakeScore:
    def __init__(self, correct: int, total: int):
        self.correct_samples = correct
        self.total_samples = total
        self.accuracy = correct / total
        self.macro_accuracy = self.accuracy

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "macro_accuracy": self.macro_accuracy,
            "task_spread": 0.0,
            "random_baseline": None,
            "score_method": "test-exact",
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "groups": {},
            "analysis": {},
            "grading": {"backend": "deterministic"},
        }


class _FakeScorer:
    def __init__(self, track: str, ground_truth_root: Path):
        self.track = track
        self.ground_truth_path = ground_truth_root / track / "ground_truth.jsonl"

    def resolved_ground_truth_files(self):
        return [self.ground_truth_path]

    def score(self, path: Path, _model_name: str):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        correct = sum(row["answer"] == "A" for row in rows)
        return _FakeScore(correct, len(rows))

    def _grade_condition(self, _question_id: str, answer: str, _condition: str):
        return answer == "A"


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_private_audit_scores_only_locked_answers_and_hides_ground_truth(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    final_root = project_root / "evaluation" / "results" / "final"
    staging_root = project_root / "evaluation" / "results" / "staging"
    ground_truth_root = project_root / "Ground_truths"
    final_root.mkdir(parents=True)
    (final_root / "index.json").write_text(
        json.dumps(
            {"models": [{"slug": "model-a", "model_id": "org/model-a"}]}
        ),
        encoding="utf-8",
    )

    tracks = []
    for track in ("do_you_see_me", "minds_eye"):
        ground_truth = ground_truth_root / track / "ground_truth.jsonl"
        _write_jsonl(
            ground_truth,
            [{"question_id": "q1", "answer": "PRIVATE_SECRET_ANSWER"}],
        )
        canonical_path = final_root / "model-a" / f"{track}_submission.jsonl"
        staged_path = staging_root / "model-a" / f"{track}_submission.jsonl"
        canonical_answer = "B" if track == "do_you_see_me" else "A"
        _write_jsonl(
            canonical_path,
            [{"question_id": "q1", "condition": "standard", "answer": canonical_answer}],
        )
        _write_jsonl(
            staged_path,
            [{"question_id": "q1", "condition": "standard", "answer": "A"}],
        )
        tracks.append(
            {
                "slug": "model-a",
                "track": track,
                "source_submission_sha256": sha256(canonical_path),
                "staged_submission_sha256": sha256(staged_path),
                "candidate_count": 1,
                "recovered_count": 1,
                "unresolved_count": 0,
            }
        )

    extraction_report = {
        "source_final_root": str(final_root.resolve()),
        "staging_root": str(staging_root.resolve()),
        "verification": {
            "status": "passed",
            "canonical_sources_unchanged": True,
        },
        "extractor": {
            "ground_truth_access": False,
            "image_access": False,
        },
        "validation": {"incorrect": 0},
        "tracks": tracks,
    }
    (staging_root / "extraction_report.json").write_text(
        json.dumps(extraction_report), encoding="utf-8"
    )
    monkeypatch.setattr(
        "evaluation.audit_local_extractor_scores.verify_canonical_results",
        lambda *_args: None,
    )

    report = build_private_audit(
        project_root=project_root,
        final_root=final_root,
        staging_root=staging_root,
        scorer_factory=lambda track: _FakeScorer(track, ground_truth_root),
    )

    assert report["summary"] == {
        "model_count": 1,
        "track_count": 2,
        "total_scored_samples": 2,
        "changed_answers": 1,
        "changed_answer_outcomes": {
            "improved": 1,
            "regressed": 0,
            "correct_both": 0,
            "incorrect_both": 0,
        },
        "canonical_correct_samples": 1,
        "staged_correct_samples": 2,
        "correct_samples_delta": 1,
    }
    assert "PRIVATE_SECRET_ANSWER" not in json.dumps(report)
    assert report["protocol"]["scoring_can_modify_extracted_answers"] is False

    output = tmp_path / "private" / "audit.json"
    _write_private_json(output, report)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
