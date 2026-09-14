import gzip
import json
import zipfile
from pathlib import Path

import pytest

from spatial_harness.artifact_package import (
    ANSWERS_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
    SCORE_SOURCE,
    SCORE_UNIT,
    VERIFICATION_LEVEL,
    ArtifactPackageError,
    aggregate_claimed_scores,
    read_artifact_package,
    write_artifact_package,
    write_gzip_jsonl,
)


def _rows():
    return [
        {
            "schema_version": ANSWERS_SCHEMA_VERSION,
            "dataset": "Dataset-A",
            "question_id": "Dataset-A:1",
            "evaluation_group": "Dataset-A:group-1",
            "answer_type": "mcq",
            "condition": "main_noncot",
            "final_answer": "A",
            "claimed_credit": 1,
        },
        {
            "schema_version": ANSWERS_SCHEMA_VERSION,
            "dataset": "Dataset-A",
            "question_id": "Dataset-A:2",
            "evaluation_group": "Dataset-A:group-2",
            "answer_type": "vqa",
            "condition": "main_noncot",
            "final_answer": "left of the chair",
            "claimed_credit": 0,
        },
    ]


def _manifest():
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at": "2026-08-02T00:00:00+00:00",
        "verification_level": VERIFICATION_LEVEL,
        "score_source": SCORE_SOURCE,
        "model": {"name": "Example Model", "revision": "abc123"},
        "benchmark": {"version": "test-v1", "manifest_sha256": "a" * 64},
        "evaluation": {
            "harness_contract": "test-contract",
            "harness_version": "1.0.0",
            "harness_commit": "commit",
            "configuration_sha256": "b" * 64,
            "judge": {"name": "User Judge", "revision": "judge-revision"},
        },
        "evidence": {
            "answer_rows": 2,
            "raw_output_rows": 2,
            "raw_output_scope": "complete_model_response",
        },
        "scoring": {"source": SCORE_SOURCE, "unit": SCORE_UNIT},
    }


def _package(tmp_path: Path) -> Path:
    rows = _rows()
    answers = tmp_path / "answers.jsonl.gz"
    raw_outputs = tmp_path / "raw_outputs.jsonl.gz"
    write_gzip_jsonl(answers, rows)
    # Normal validation intentionally does not parse the potentially large raw
    # artifact. Its exact bytes are retained and checksum-bound for later audit.
    raw_outputs.write_bytes(gzip.compress(b"opaque raw-output evidence\n", mtime=0))
    scores = aggregate_claimed_scores(rows, ["Dataset-A"], ["main_noncot"])
    return write_artifact_package(
        tmp_path / "track3_artifact_submission.zip",
        _manifest(),
        scores,
        answers,
        raw_outputs,
    )


def test_artifact_package_validates_integrity_and_claimed_arithmetic_only(tmp_path):
    package = read_artifact_package(_package(tmp_path))

    assert len(package.answer_rows) == 2
    assert package.computed_scores["datasets"]["Dataset-A"]["main_noncot"] == {
        "correct": 1,
        "total": 2,
    }
    assert package.manifest["verification_level"] == VERIFICATION_LEVEL


def test_artifact_package_rejects_member_tampering(tmp_path):
    package_path = _package(tmp_path)
    members = {}
    with zipfile.ZipFile(package_path, "r") as archive:
        for name in archive.namelist():
            members[name] = archive.read(name)
    scores = json.loads(members["claimed_scores.json"])
    scores["datasets"]["Dataset-A"]["main_noncot"]["correct"] = 2
    members["claimed_scores.json"] = json.dumps(scores).encode("utf-8")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(ArtifactPackageError) as captured:
        read_artifact_package(tampered)

    assert captured.value.code == "artifact_checksum_mismatch"


def test_artifact_package_rejects_self_consistent_hashes_with_bad_arithmetic(tmp_path):
    rows = _rows()
    answers = tmp_path / "answers.jsonl.gz"
    raw_outputs = tmp_path / "raw_outputs.jsonl.gz"
    write_gzip_jsonl(answers, rows)
    raw_outputs.write_bytes(gzip.compress(b"evidence\n", mtime=0))
    scores = aggregate_claimed_scores(rows, ["Dataset-A"], ["main_noncot"])
    scores["datasets"]["Dataset-A"]["main_noncot"]["correct"] = 2

    with pytest.raises(ArtifactPackageError) as captured:
        write_artifact_package(
            tmp_path / "bad-arithmetic.zip",
            _manifest(),
            scores,
            answers,
            raw_outputs,
        )

    assert captured.value.code == "artifact_score_arithmetic_mismatch"
