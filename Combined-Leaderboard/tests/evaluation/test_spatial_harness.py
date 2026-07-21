import base64
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from judge_track3 import (
    _aggregate_report,
    _clear_stale_judge_artifacts,
    _load_inference_manifest,
    _write_submission_package,
    explicit_abstention_letter,
    resolve_judge_endpoint_model,
)
from run_track3_vllm import (
    _clear_stale_run_artifacts,
    build_records,
    resolve_model_names,
)
from loaders.prepare_custom import Loader, convert_records
from spatial_contract import DATASETS, HARNESS_VERSION, REQUIRED_CONDITIONS


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_dataset(root: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / f"{name}.tsv", sep="\t", index=False)


def test_selected_rows_resolve_image_references_from_full_tsv(tmp_path):
    encoded = _image_b64()
    _write_dataset(
        tmp_path,
        "BLINK",
        [
            {"index": "0", "image": encoded, "question": "Q0", "A": "x", "B": "y", "answer": "A"},
            {"index": "1", "image": "0", "question": "Q1", "A": "x", "B": "y", "answer": "B"},
        ],
    )

    records = build_records(tmp_path, "BLINK", "noimage", selected_indices={"1"})

    assert len(records) == 1
    assert records[0]["images"] == [encoded]
    assert records[0]["gray"] is True


def test_main_circular_records_rotate_answers_and_share_group(tmp_path):
    _write_dataset(
        tmp_path,
        "SpatialBench",
        [{"index": "7", "image": _image_b64(), "question": "Q", "A": "left", "B": "right", "answer": "A"}],
    )

    records = build_records(tmp_path, "SpatialBench", "main")

    assert [record["question_id"] for record in records] == [
        "SpatialBench:7:r0",
        "SpatialBench:7:r1",
    ]
    assert {record["evaluation_group"] for record in records} == {"SpatialBench:7"}
    assert [record["gt"] for record in records] == ["A", "B"]


def test_no_image_plus_appends_the_abstention_option(tmp_path):
    _write_dataset(
        tmp_path,
        "BLINK",
        [{"index": "1", "image": _image_b64(), "question": "Q", "A": "left", "B": "right", "answer": "A"}],
    )

    record = build_records(tmp_path, "BLINK", "noimgpp", selected_indices={"1"})[0]

    assert record["cannot_label"] == "C"
    assert record["gt"] == "C"
    assert record["options"]["C"] == "Cannot determine from the image"
    assert explicit_abstention_letter({**record, "mode": "noimgpp", "output": "Cannot determine"}) == "C"
    assert explicit_abstention_letter(
        {**record, "mode": "noimgpp", "output": "I cannot determine from the image."}
    ) == "C"


def test_report_uses_all_or_nothing_circular_group_states():
    report = _aggregate_report(
        {
            ("SpatialBench", "main_noncot", "SpatialBench:1"): {"all_correct": False, "variants": 4},
            ("SpatialBench", "main_noncot", "SpatialBench:2"): {"all_correct": True, "variants": 4},
        },
        ["SpatialBench"],
    )

    result = report["datasets"][0]["experiments"]["main"]["noncot"]
    assert result == {"correct": 1, "total": 2, "accuracy": 0.5}


def test_rerun_cleanup_removes_stale_uploads_but_preserves_unrelated_files(tmp_path):
    stale = [
        "inference_manifest.json",
        "submission.jsonl",
        "run_manifest.json",
        "spatial_reasoning_submission.zip",
        "pred_main_noncot.jsonl",
        "judged_noimgpp_cot.jsonl",
    ]
    for name in stale:
        (tmp_path / name).write_text("old", encoding="utf-8")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    _clear_stale_run_artifacts(tmp_path)

    assert all(not (tmp_path / name).exists() for name in stale)
    assert unrelated.read_text(encoding="utf-8") == "keep"

    (tmp_path / "submission.jsonl").write_text("old", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("old", encoding="utf-8")
    _clear_stale_judge_artifacts(tmp_path)
    assert not (tmp_path / "submission.jsonl").exists()
    assert not (tmp_path / "run_manifest.json").exists()
    assert unrelated.exists()


def test_judge_creates_one_exact_upload_package(tmp_path):
    submission = tmp_path / "submission.jsonl"
    manifest = tmp_path / "run_manifest.json"
    report = tmp_path / "leaderboard.json"
    submission.write_bytes(b'{"question_id":"q1","condition":"main_noncot","answer":"A"}\n')
    manifest.write_bytes(b'{"schema_version":"ms-vista-spatial-run/v1"}\n')
    report.write_bytes(b'{"schema_version":"ms-vista-spatial-report/v2"}\n')

    package_path = _write_submission_package(tmp_path, submission, manifest, report)

    assert package_path.name == "spatial_reasoning_submission.zip"
    with zipfile.ZipFile(package_path) as package:
        assert package.namelist() == [
            "submission.jsonl",
            "run_manifest.json",
            "leaderboard.json",
        ]
        assert package.read("submission.jsonl") == submission.read_bytes()
        assert package.read("run_manifest.json") == manifest.read_bytes()
        assert package.read("leaderboard.json") == report.read_bytes()


def test_judge_rejects_an_inference_manifest_from_an_old_harness(tmp_path):
    manifest = {
        "schema_version": "ms-vista-spatial-inference/v1",
        "harness_version": "0.9.0",
        "debug": False,
        "datasets": list(DATASETS),
        "conditions": list(REQUIRED_CONDITIONS),
        "error_counts": {condition: 0 for condition in REQUIRED_CONDITIONS},
        "benchmark_manifest_sha256": "a" * 64,
    }
    (tmp_path / "inference_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="current harness"):
        _load_inference_manifest(tmp_path)

    manifest["harness_version"] = HARNESS_VERSION
    manifest["datasets"] = list(DATASETS[:-1])
    (tmp_path / "inference_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="all 13 official datasets"):
        _load_inference_manifest(tmp_path)


def test_endpoint_and_public_model_names_are_independent_with_legacy_support():
    assert resolve_model_names("served/qwen", "Qwen Public", "") == (
        "served/qwen",
        "Qwen Public",
    )
    assert resolve_model_names("served/qwen", "", "") == (
        "served/qwen",
        "served/qwen",
    )
    assert resolve_model_names("", "", "legacy-name") == (
        "legacy-name",
        "legacy-name",
    )
    assert resolve_judge_endpoint_model("judge-served", "") == "judge-served"
    assert resolve_judge_endpoint_model("", "legacy-judge") == "legacy-judge"
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_model_names("served", "Public", "legacy")
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_judge_endpoint_model("served", "legacy")


def test_custom_conversion_stops_on_unexpected_record_errors_and_skips():
    class FailingLoader(Loader):
        name = "Failing"

        def to_intermediate(self, _record):
            raise KeyError("missing answer")

    with pytest.raises(RuntimeError, match="source record 0.*missing answer"):
        convert_records(FailingLoader(), [{}])

    class ChangedLoader(Loader):
        name = "Changed"
        expected_skips = {}

        def to_intermediate(self, record):
            return {
                "image": _image_b64(),
                "question": "Q",
                "options": record.get("options"),
                "answer": "A",
            }

    with pytest.raises(RuntimeError, match="exclusion counts changed"):
        convert_records(
            ChangedLoader(),
            [{"options": ["yes", "no"]}, {"options": None}],
        )
