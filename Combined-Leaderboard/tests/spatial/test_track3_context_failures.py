import json
from pathlib import Path

import pytest

from spatial_harness.finalize_context_failures import finalize_context_failures
from spatial_harness.run_track3_vllm import (
    DATASETS,
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
)
from spatial_harness.submission_contract import (
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, failed_sample: str = "MMSIBench_wo_circular:140"):
    dataset, index = failed_sample.split(":", 1)
    run_config = {
        "schema_version": 5,
        "harness_contract": HARNESS_CONTRACT,
        "model": "example/model",
        "model_revision": "abc",
        "datasets": list(DATASETS),
        "modes": list(MODES),
        "prompt_modes": list(PROMPT_MODES),
        "server_metadata": {"max_model_len": 32768},
    }
    _write_json(tmp_path / "run_config.json", run_config)
    for mode in MODES:
        for prompt_mode in PROMPT_MODES:
            fails = not (mode == "main" and prompt_mode == "noncot")
            row = {
                "dataset": dataset,
                "index": index,
                "group": index,
                "answer_type": "mcq",
                "options": {"A": "left", "B": "right"},
                "gt": "A",
                "cannot_label": None,
                "mode": mode,
                "pmode": prompt_mode,
                "output": "" if fails else "A",
            }
            if fails:
                row["error"] = (
                    "BadRequestError: Input length (33432) exceeds model's "
                    "maximum context length (32768)."
                )
            _write_jsonl(
                tmp_path / f"pred_{mode}_{prompt_mode}.jsonl",
                [row],
            )


def test_finalizer_marks_only_approved_context_failures(tmp_path):
    _fixture(tmp_path)

    report = finalize_context_failures(
        tmp_path,
        tmp_path / "inference_failure_resolution.json",
        {"MMSIBench_wo_circular:140"},
        5,
    )

    assert report["condition_rows"] == 5
    assert report["unique_samples"] == 1
    assert report["disposition"] == INFERENCE_FAILURE_DISPOSITION
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["terminal_inference_failures"]["policy"] == INFERENCE_FAILURE_POLICY
    for path in tmp_path.glob("pred_*.jsonl"):
        row = json.loads(path.read_text())
        if path.name == "pred_main_noncot.jsonl":
            assert "terminal_failure" not in row
        else:
            assert row["finish_reason"] == "inference_error"
            assert row["terminal_failure"]["schema_version"] == (
                INFERENCE_FAILURE_SCHEMA_VERSION
            )


def test_finalizer_rejects_unapproved_failure_without_writing(tmp_path):
    _fixture(tmp_path, failed_sample="MMSIBench_wo_circular:999")
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.glob("pred_*.jsonl")
    }

    with pytest.raises(ValueError, match="not explicitly approved"):
        finalize_context_failures(
            tmp_path,
            tmp_path / "inference_failure_resolution.json",
            {"MMSIBench_wo_circular:140"},
            5,
        )

    assert before == {
        path.name: path.read_bytes()
        for path in tmp_path.glob("pred_*.jsonl")
    }
