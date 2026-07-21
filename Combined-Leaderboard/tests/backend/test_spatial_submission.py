import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

import spatial_submission  # noqa: E402
from config import (  # noqa: E402
    EVAL_CONDITIONS,
    GRADING,
    SPATIAL_BENCHMARK_SCHEMA_VERSION,
    SPATIAL_DATASET_KEYS,
    SPATIAL_RUN_SCHEMA_VERSION,
    SPATIAL_SUBMISSION_SCHEMA_VERSION,
)
from scoring.task_scorer import SubmissionValidationError  # noqa: E402
import spatial_contract  # noqa: E402


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_backend_and_harness_contract_constants_match():
    assert list(spatial_contract.DATASETS) == SPATIAL_DATASET_KEYS
    assert list(spatial_contract.REQUIRED_CONDITIONS) == EVAL_CONDITIONS
    assert spatial_contract.BENCHMARK_MANIFEST_SCHEMA_VERSION == SPATIAL_BENCHMARK_SCHEMA_VERSION
    assert spatial_contract.RUN_MANIFEST_SCHEMA_VERSION == SPATIAL_RUN_SCHEMA_VERSION
    assert spatial_contract.SUBMISSION_SCHEMA_VERSION == SPATIAL_SUBMISSION_SCHEMA_VERSION


def _official_fixture(tmp_path: Path):
    submission_rows = [
        {
            "dataset": dataset,
            "question_id": f"{dataset}:q1",
            "evaluation_group": f"{dataset}:q1",
            "condition": condition,
            "answer": "A",
            "correct": True,
            "judge_method": "qwen_llm_judge",
            "judge_attempts": 1,
        }
        for dataset in SPATIAL_DATASET_KEYS
        for condition in EVAL_CONDITIONS
    ]
    submission = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        for row in submission_rows
    )
    template = b"".join(
        (
            json.dumps(
                {
                    "question_id": row["question_id"],
                    "condition": row["condition"],
                    "answer": "",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in submission_rows
    )
    template_path = tmp_path / "submission_template.jsonl"
    template_path.write_bytes(template)
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        "".join(
            json.dumps({
                "question_id": f"{dataset}:q1",
                "dataset_key": dataset,
                "evaluation_group": f"{dataset}:q1",
                "conditions": EVAL_CONDITIONS,
            }) + "\n"
            for dataset in SPATIAL_DATASET_KEYS
        ),
        encoding="utf-8",
    )
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                f"{dataset}:q1": {
                    "answer": "A",
                    "condition_answers": {condition: "A" for condition in EVAL_CONDITIONS},
                    "conditions": EVAL_CONDITIONS,
                    "dataset": dataset,
                    "evaluation_group": f"{dataset}:q1",
                }
                for dataset in SPATIAL_DATASET_KEYS
            }
        ),
        encoding="utf-8",
    )
    dataset_files = {
        dataset: {"filename": f"{dataset}.tsv", "size_bytes": 10, "sha256": "a" * 64}
        for dataset in SPATIAL_DATASET_KEYS
    }
    prompts = {
        "noncot": {"filename": "base_noncot.txt", "sha256": "b" * 64},
        "cot": {"filename": "cot_default.txt", "sha256": "c" * 64},
    }
    ablation = {"filename": "ablation_manifest.json", "sha256": "d" * 64}
    judge = {
        "revision": GRADING["spatial"]["judge_model"],
        "system_prompt_sha256": "e" * 64,
        "method_counts": {"qwen_llm_judge": len(submission_rows)},
        "decoding": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": 4,
        },
    }
    decoding = {
        "strategy": "greedy",
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
    manifest = {
        "schema_version": SPATIAL_BENCHMARK_SCHEMA_VERSION,
        "task_id": "spatial",
        "benchmark_version": "test-v1",
        "harness_version": spatial_contract.HARNESS_VERSION,
        "demo": False,
        "datasets": SPATIAL_DATASET_KEYS,
        "dataset_count": len(SPATIAL_DATASET_KEYS),
        "required_conditions": EVAL_CONDITIONS,
        "primary_condition": "main_noncot",
        "condition_counts": {
            condition: len(SPATIAL_DATASET_KEYS) for condition in EVAL_CONDITIONS
        },
        "condition_group_counts": {
            condition: len(SPATIAL_DATASET_KEYS) for condition in EVAL_CONDITIONS
        },
        "dataset_condition_counts": {
            dataset: {condition: 1 for condition in EVAL_CONDITIONS}
            for dataset in SPATIAL_DATASET_KEYS
        },
        "dataset_condition_group_counts": {
            dataset: {condition: 1 for condition in EVAL_CONDITIONS}
            for dataset in SPATIAL_DATASET_KEYS
        },
        "unique_question_ids": len(SPATIAL_DATASET_KEYS),
        "dataset_files": dataset_files,
        "prompts": prompts,
        "ablation_manifest": ablation,
        "judge": judge,
        "decoding": decoding,
        "artifacts": {
            "questions": {
                "filename": questions_path.name,
                "rows": len(SPATIAL_DATASET_KEYS),
                "sha256": _sha(questions_path.read_bytes()),
            },
            "submission_template": {
                "filename": template_path.name,
                "rows": len(submission_rows),
                "sha256": _sha(template),
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_manifest = {
        "schema_version": SPATIAL_RUN_SCHEMA_VERSION,
        "submission_schema_version": SPATIAL_SUBMISSION_SCHEMA_VERSION,
        "harness_version": spatial_contract.HARNESS_VERSION,
        "debug": False,
        "model": {"name": "Test Model"},
        "datasets": SPATIAL_DATASET_KEYS,
        "conditions": EVAL_CONDITIONS,
        "condition_counts": {
            condition: len(SPATIAL_DATASET_KEYS) for condition in EVAL_CONDITIONS
        },
        "error_counts": {"inference": 0, "judge": 0, "missing_outputs": 0},
        "benchmark_manifest_sha256": _sha(manifest_path.read_bytes()),
        "dataset_files": dataset_files,
        "prompts": prompts,
        "ablation_manifest": ablation,
        "judge": judge,
        "decoding": {**decoding, "max_tokens_noncot": 16384, "max_tokens_cot": 16384},
        "artifacts": {
            "submission": {
                "filename": "submission.jsonl",
                "rows": len(submission_rows),
                "sha256": _sha(submission),
            },
        },
    }
    records = [
        dict(row)
        for row in submission_rows
    ]
    report = {
        "schema_version": spatial_contract.REPORT_SCHEMA_VERSION,
        "model": {"name": "Test Model"},
        "conditions": EVAL_CONDITIONS,
        "datasets": [
            {
                "dataset": dataset,
                "experiments": {
                    mode: {
                        prompt: {"correct": 1, "total": 1, "accuracy": 1.0}
                        for prompt in ("noncot", "cot")
                    }
                    for mode in ("main", "no_image", "no_image_plus")
                },
            }
            for dataset in SPATIAL_DATASET_KEYS
        ],
        "summary": {
            **{condition: 1.0 for condition in EVAL_CONDITIONS},
            "cot_delta": 0.0,
        },
    }
    report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode("utf-8")
    run_manifest["artifacts"]["leaderboard_report"] = {
        "filename": "leaderboard.json",
        "sha256": _sha(report_bytes),
        "size_bytes": len(report_bytes),
        "dataset_count": len(SPATIAL_DATASET_KEYS),
    }
    return (
        manifest_path,
        ground_truth_path,
        template_path,
        questions_path,
        submission,
        run_manifest,
        records,
        report_bytes,
    )


def test_spatial_run_manifest_accepts_matching_official_artifacts(tmp_path):
    manifest_path, _ground_truth, _template, _questions, submission, run_manifest, records, report = _official_fixture(tmp_path)

    result = spatial_submission.validate_run_manifest(
        json.dumps(run_manifest).encode("utf-8"),
        submission,
        report,
        "Test Model",
        records,
        manifest_path,
    )

    assert result["benchmark_version"] == "test-v1"
    assert result["judge_revision"] == GRADING["spatial"]["judge_model"]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda manifest: manifest["model"].update(name="Other Model"), "spatial_model_name_mismatch"),
        (lambda manifest: manifest.update(debug=True), "debug_spatial_run_not_allowed"),
        (lambda manifest: manifest["judge"].update(revision="other"), "spatial_judge_mismatch"),
    ],
)
def test_spatial_run_manifest_rejects_provenance_mismatches(tmp_path, mutation, expected_code):
    manifest_path, _ground_truth, _template, _questions, submission, run_manifest, records, report = _official_fixture(tmp_path)
    mutation(run_manifest)

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.validate_run_manifest(
            json.dumps(run_manifest).encode("utf-8"),
            submission,
            report,
            "Test Model",
            records,
            manifest_path,
        )

    assert captured.value.code == expected_code
    assert captured.value.details["field"] == "file"


def test_spatial_run_manifest_rejects_edited_submission(tmp_path):
    manifest_path, _ground_truth, _template, _questions, submission, run_manifest, records, report = _official_fixture(tmp_path)

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.validate_run_manifest(
            json.dumps(run_manifest).encode("utf-8"),
            submission + b"\n",
            report,
            "Test Model",
            records,
            manifest_path,
        )

    assert captured.value.code == "spatial_submission_hash_mismatch"


def test_spatial_bundle_health_requires_non_demo_consistent_bundle(tmp_path, monkeypatch):
    manifest_path, _ground_truth, template, questions, _submission, _run_manifest, _records, _report = _official_fixture(tmp_path)
    monkeypatch.setattr(spatial_submission, "OFFICIAL_SPATIAL_MIN_SAMPLES", 1)

    status, details = spatial_submission.spatial_bundle_health(
        manifest_path,
        template,
        questions,
    )

    assert status == "healthy"
    assert details["production_ready"] is True

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["demo"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status, details = spatial_submission.spatial_bundle_health(
        manifest_path,
        template,
        questions,
    )
    assert status == "unhealthy"
    assert "demo" in details["error"].lower()


def test_public_evidence_recomputes_and_validates_aggregate_without_ground_truth(tmp_path):
    (
        manifest_path,
        _ground_truth,
        template,
        questions,
        submission,
        run_manifest,
        _records,
        report_bytes,
    ) = _official_fixture(tmp_path)

    records, computed, _manifest = spatial_submission.parse_spatial_evidence(
        submission,
        manifest_path,
        template,
        questions,
    )
    report = spatial_submission.validate_spatial_report(
        report_bytes,
        "Test Model",
        computed,
    )
    run_metadata = spatial_submission.validate_run_manifest(
        json.dumps(run_manifest).encode("utf-8"),
        submission,
        report_bytes,
        "Test Model",
        records,
        manifest_path,
    )
    score = spatial_submission.build_spatial_task_score(
        report,
        "Test Model",
        {"organization": "Example Lab"},
        run_metadata,
    )

    assert len(records) == len(SPATIAL_DATASET_KEYS) * len(EVAL_CONDITIONS)
    assert score.accuracy == 1.0
    assert score.macro_accuracy == 1.0
    assert score.total_samples == len(SPATIAL_DATASET_KEYS)
    assert score.grading["server_ground_truth_evaluation"] is False
    assert score.metadata["public_evidence"]["available"] is True


def test_spatial_contract_can_be_replayed_from_retained_bytes(tmp_path):
    (
        manifest_path,
        _ground_truth,
        template_path,
        questions_path,
        submission,
        run_manifest,
        _records,
        report_bytes,
    ) = _official_fixture(tmp_path)
    manifest_bytes = manifest_path.read_bytes()
    template_bytes = template_path.read_bytes()
    questions_bytes = questions_path.read_bytes()

    records, computed, _manifest = spatial_submission.parse_spatial_evidence(
        submission,
        manifest_bytes,
        template_bytes,
        questions_bytes,
    )
    run_metadata = spatial_submission.validate_run_manifest(
        json.dumps(run_manifest).encode("utf-8"),
        submission,
        report_bytes,
        "Test Model",
        records,
        manifest_bytes,
    )

    assert run_metadata["benchmark_manifest_sha256"] == _sha(manifest_bytes)
    assert computed["summary"]["main_noncot"] == 1.0


def test_spatial_report_rejects_score_that_does_not_match_public_evidence(tmp_path):
    (
        manifest_path,
        _ground_truth,
        template,
        questions,
        submission,
        _run_manifest,
        _records,
        report_bytes,
    ) = _official_fixture(tmp_path)
    _records, computed, _manifest = spatial_submission.parse_spatial_evidence(
        submission,
        manifest_path,
        template,
        questions,
    )
    report = json.loads(report_bytes)
    report["summary"]["main_noncot"] = 0.5

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.validate_spatial_report(
            json.dumps(report).encode("utf-8"),
            "Test Model",
            computed,
        )

    assert captured.value.code == "spatial_report_evidence_mismatch"


def _spatial_zip(
    submission: bytes = b'{"question_id":"q1","condition":"main_noncot","answer":"A"}\n',
    manifest: bytes = b'{"schema_version":"ms-vista-spatial-run/v1"}\n',
    report: bytes = b'{"schema_version":"ms-vista-spatial-report/v2"}\n',
    *,
    extra: tuple[str, bytes] | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("submission.jsonl", submission)
        package.writestr("run_manifest.json", manifest)
        package.writestr("leaderboard.json", report)
        if extra:
            package.writestr(*extra)
    return output.getvalue()


def test_spatial_archive_reads_exact_members_without_extracting(tmp_path):
    submission = b'{"question_id":"q1","condition":"main_noncot","answer":"A"}\n'
    manifest = b'{"schema_version":"ms-vista-spatial-run/v1"}\n'
    report = b'{"schema_version":"ms-vista-spatial-report/v2"}\n'

    extracted_submission, extracted_manifest, extracted_report = (
        spatial_submission.read_spatial_submission_archive(
            _spatial_zip(submission, manifest, report)
        )
    )

    assert extracted_submission == submission
    assert extracted_manifest == manifest
    assert extracted_report == report
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"not a zip", "invalid_spatial_submission_archive"),
        (
            _spatial_zip(extra=("unexpected.txt", b"no")),
            "invalid_spatial_archive_contents",
        ),
        (
            _spatial_zip(extra=("nested/submission.jsonl", b"no")),
            "invalid_spatial_archive_contents",
        ),
    ],
)
def test_spatial_archive_rejects_invalid_packages(payload, expected_code):
    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(payload)

    assert captured.value.code == expected_code
    assert captured.value.details["field"] == "file"


def test_spatial_archive_rejects_symlinks():
    output = io.BytesIO()
    link = zipfile.ZipInfo("submission.jsonl")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr(link, "target")
        package.writestr("run_manifest.json", "{}")
        package.writestr("leaderboard.json", "{}")

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(output.getvalue())

    assert captured.value.code == "unsafe_spatial_archive_member"


def test_spatial_archive_rejects_oversized_uncompressed_members(monkeypatch):
    monkeypatch.setattr(spatial_submission, "MAX_SPATIAL_SUBMISSION_BYTES", 8)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("submission.jsonl", b"123456789")
        package.writestr("run_manifest.json", b"{}")
        package.writestr("leaderboard.json", b"{}")

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(output.getvalue())

    assert captured.value.code == "spatial_archive_member_too_large"


def test_spatial_archive_rejects_duplicate_members():
    output = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as package:
            package.writestr("submission.jsonl", b"first")
            package.writestr("submission.jsonl", b"second")
            package.writestr("run_manifest.json", b"{}")
            package.writestr("leaderboard.json", b"{}")

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(output.getvalue())

    assert captured.value.code == "duplicate_spatial_archive_members"


def test_spatial_archive_rejects_suspicious_compression_ratio():
    payload = _spatial_zip(submission=b"0" * 20_000)

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(payload)

    assert captured.value.code == "unsafe_spatial_archive_ratio"


def test_spatial_archive_rejects_encrypted_flag():
    payload = bytearray(_spatial_zip())
    local_header = payload.find(b"PK\x03\x04")
    central_header = payload.find(b"PK\x01\x02")
    assert local_header >= 0 and central_header >= 0
    payload[local_header + 6 : local_header + 8] = (
        int.from_bytes(payload[local_header + 6 : local_header + 8], "little") | 1
    ).to_bytes(2, "little")
    payload[central_header + 8 : central_header + 10] = (
        int.from_bytes(payload[central_header + 8 : central_header + 10], "little") | 1
    ).to_bytes(2, "little")

    with pytest.raises(SubmissionValidationError) as captured:
        spatial_submission.read_spatial_submission_archive(bytes(payload))

    assert captured.value.code == "encrypted_spatial_archive"
