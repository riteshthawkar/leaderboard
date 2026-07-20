import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spatial_harness import prepare_data
from spatial_harness.loaders.prepare_custom import (
    mindcube_rows,
    mmvp_rows,
    omnispatial_rows,
    realworldqa_rows,
    sat_real_rows,
    spatialbench_rows,
)
from spatial_harness.prepare_data import (
    EXPECTED_COUNTS,
    MINDCUBE_TSV,
    PINNED_TSV_DATASETS,
    _atomic_manifest,
    _load_manifest,
    normalize_hosted_frame,
    validate_frame,
    verify_bundle,
)


def test_vsr_hosted_rows_are_converted_to_mcq():
    frame = pd.DataFrame(
        [{"index": 1, "image": "a" * 80, "question": "Is it left?", "answer": "yes"}]
    )
    result = normalize_hosted_frame(frame, "VSR_MCQ")
    assert result.loc[0, "A"] == "Yes"
    assert result.loc[0, "B"] == "No"
    assert result.loc[0, "answer"] == "A"
    assert result.loc[0, "answer_type"] == "mcq"


def test_mmvp_uses_official_lowercase_option_markers(tmp_path: Path):
    image_root = tmp_path / "MMVP Images"
    image_root.mkdir()
    Image.new("RGB", (2, 2)).save(image_root / "1.jpg")
    (tmp_path / "Questions.csv").write_text(
        "Index,Question,Options,Correct Answer\n"
        '1,Open or closed?,"(a) Open (b) Closed",(b)\n',
        encoding="utf-8",
    )
    rows = mmvp_rows(tmp_path)
    assert rows[0]["A"] == "Open"
    assert rows[0]["B"] == "Closed"
    assert rows[0]["answer"] == "B"
    assert rows[0]["answer_type"] == "mcq"


def test_realworldqa_routes_option_and_direct_questions():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    image = {"bytes": buffer.getvalue(), "path": None}
    frame = pd.DataFrame(
        [
            {
                "image": image,
                "question": (
                    "Which side?\nA. Left\nB. Right\n"
                    "Please answer directly with only the letter of the correct option."
                ),
                "answer": "B",
            },
            {
                "image": image,
                "question": (
                    "How many?\nPlease answer directly with a single word or number."
                ),
                "answer": "3",
            },
        ]
    )
    rows = realworldqa_rows(frame)
    assert [row["answer_type"] for row in rows] == ["mcq", "vqa"]
    assert rows[0]["question"] == "Which side?"
    assert rows[0]["answer"] == "B"
    assert rows[1]["question"] == "How many?"
    assert rows[1]["answer"] == "3"


def test_mindcube_parses_candidates_and_absolute_multi_image_paths(tmp_path: Path):
    first = tmp_path / "data" / "scene" / "1.jpg"
    second = tmp_path / "data" / "scene" / "2.jpg"
    first.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(first)
    Image.new("RGB", (2, 2)).save(second)
    frame = pd.DataFrame(
        [
            {
                "index": "q1",
                "question": "<image><image>\nWhat is left? A. Door B. Wall",
                "answer": "B",
                "image_path": "['/data/scene/1.jpg', '/data/scene/2.jpg']",
                "candidates": "['A. Door', 'B. Wall']",
                "category": "rotation",
            }
        ]
    )
    rows = mindcube_rows(frame, tmp_path)
    assert rows[0]["question"] == "What is left?"
    assert rows[0]["A"] == "Door"
    assert rows[0]["answer"] == "B"
    assert json.loads(rows[0]["image"]) == [str(first.resolve()), str(second.resolve())]


def test_omnispatial_keeps_vqa_and_mcq_rows(tmp_path: Path):
    root = tmp_path / "OmniSpatial-test"
    (root / "Dynamic_Reasoning").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(root / "Dynamic_Reasoning" / "1.png")
    (root / "data.json").write_text(
        """[
          {"id":"1_0","question":"Where?","options":["left","right"],"answer":0,"task_type":"Dynamic_Reasoning","sub_task_type":"motion"},
          {"id":"1_1","question":"How many?","options":[],"answer":"3","task_type":"Dynamic_Reasoning","sub_task_type":"count"}
        ]""",
        encoding="utf-8",
    )
    rows = omnispatial_rows(tmp_path)
    assert [row["answer_type"] for row in rows] == ["mcq", "vqa"]
    assert rows[1]["answer"] == "3"


def test_spatialbench_keeps_mixed_rows_and_multiple_images(tmp_path: Path):
    (tmp_path / "size").mkdir()
    for name in ("first.png", "second.png"):
        Image.new("RGB", (2, 2)).save(tmp_path / "size" / name)
    (tmp_path / "size.json").write_text(
        json.dumps(
            [
                {
                    "image": ["size/first.png", "size/second.png"],
                    "question": "Which is larger? (A) first (B) second",
                    "answer": "first",
                },
                {
                    "image": "size/first.png",
                    "question": "How many objects?",
                    "answer": 3,
                },
            ]
        ),
        encoding="utf-8",
    )
    rows, skipped = spatialbench_rows(tmp_path)
    assert skipped == {}
    assert [row["answer_type"] for row in rows] == ["mcq", "vqa"]
    assert len(json.loads(rows[0]["image"])) == 2


def test_spatialbench_download_uses_https_and_restores_xet(monkeypatch, tmp_path: Path):
    previous = prepare_data.hf_constants.HF_HUB_DISABLE_XET

    def fake_snapshot_download(*args, **kwargs):
        assert prepare_data.hf_constants.HF_HUB_DISABLE_XET is True
        return tmp_path

    monkeypatch.setattr(prepare_data, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(prepare_data, "spatialbench_rows", lambda _snapshot: ([], {}))

    rows, metadata = prepare_data.build_spatialbench(tmp_path, "test-token")

    assert rows == []
    assert metadata["transport"] == "https"
    assert prepare_data.hf_constants.HF_HUB_DISABLE_XET is previous


def test_sat_real_preserves_multiple_images():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    frame = pd.DataFrame(
        [
            {
                "image_bytes": np.array([buffer.getvalue(), buffer.getvalue()], dtype=object),
                "question": "Which direction?",
                "answers": np.array(["left", "right"], dtype=object),
                "correct_answer": 1,
            }
        ]
    )
    rows = sat_real_rows(frame)
    assert rows[0]["answer"] == "B"
    assert rows[0]["answer_type"] == "mcq"
    assert rows[0]["image"].startswith("[")


def test_subset_manifest_preserves_existing_dataset_provenance(tmp_path: Path):
    lmudata = tmp_path / "LMUData"
    lmudata.mkdir()
    (lmudata / "BLINK.tsv").write_text("index\tanswer_type\n0\tmcq\n")
    path = lmudata / "track3_data_manifest.json"
    manifest = {
        "schema_version": 2,
        "vlmevalkit_commit": "7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f",
        "datasets": {
            "BLINK": {
                "source": "VLMEvalKit",
                "registry_name": "BLINK",
                "rows": 1,
                "answer_types": {"mcq": 1},
                "tsv_sha256": "abc",
            }
        },
    }
    _atomic_manifest(path, manifest)
    loaded = _load_manifest(path, lmudata)
    loaded["datasets"]["SAT-Real"] = {
        "source": "test",
        "rows": 150,
        "answer_types": {"mcq": 150},
        "tsv_sha256": "def",
    }
    _atomic_manifest(path, loaded)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["dataset_count"] == 2
    assert result["datasets"]["BLINK"]["registry_name"] == "BLINK"


def test_frame_validation_rejects_invalid_mcq_label():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "image": "a" * 80,
                "question": "Which?",
                "A": "left",
                "B": "right",
                "answer": "C",
                "answer_type": "mcq",
            }
        ]
    )
    with pytest.raises(ValueError, match="answer 'C'"):
        validate_frame(frame, "test")


def test_large_hosted_tsvs_are_revision_pinned():
    assert PINNED_TSV_DATASETS == {
        "VStarBench": (
            "xjtupanda/VStar_Bench",
            "VStarBench.tsv",
            "b1312e6fbf89ace5a50cc5b0214010aeb0f69bdc",
        ),
        "MMSIBench_wo_circular": (
            "lmms-lab-si/EASI-Leaderboard-Data",
            "MMSIBench_wo_circular.tsv",
            "52bb1dd2bc5cd58a291b3a780dab08d19c0ccf0d",
        ),
    }


def test_mindcube_uses_paper_1k_split():
    assert MINDCUBE_TSV == "MindCubeBench_tiny_raw_qa.tsv"
    assert EXPECTED_COUNTS["MindCube"] == 1040


def test_bundle_verification_rejects_tampered_tsv(tmp_path: Path):
    lmudata = tmp_path / "LMUData"
    lmudata.mkdir()
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "image": "a" * 80,
                "question": "Which?",
                "A": "left",
                "B": "right",
                "answer": "A",
                "answer_type": "mcq",
            }
        ]
    )
    destination = lmudata / "BLINK.tsv"
    frame.to_csv(destination, sep="\t", index=False)
    manifest = {
        "schema_version": 2,
        "vlmevalkit_commit": "7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f",
        "datasets": {
            "BLINK": {
                "rows": 1,
                "answer_types": {"mcq": 1},
                "tsv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        },
    }
    _atomic_manifest(lmudata / "track3_data_manifest.json", manifest)
    verify_bundle(lmudata, ("BLINK",))
    destination.write_text(destination.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_bundle(lmudata, ("BLINK",))