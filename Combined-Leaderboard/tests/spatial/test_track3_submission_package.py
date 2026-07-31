import hashlib
import json
import zipfile
from pathlib import Path

import spatial_submission

from spatial_harness.judge_track3 import (
    PAPER_JUDGE_MODEL,
    PAPER_JUDGE_REVISION,
)
from spatial_harness.package_submission import (
    LONG_VQA_COMMITMENT_PREFIX,
    _public_vqa_answer,
    package_submission,
)
from spatial_harness.run_track3_vllm import (
    DATASETS,
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
)
from spatial_harness.submission_contract import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    EVALUATION_POLICY,
    HARNESS_VERSION,
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_JUDGE_METHOD,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
    JUDGE_DECODING,
    REQUIRED_CONDITIONS,
    condition_for,
    public_question_id,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "run"
    contract_dir = tmp_path / "contract"
    input_dir.mkdir()
    contract_dir.mkdir()
    run_config = {
        "schema_version": 5,
        "harness_contract": HARNESS_CONTRACT,
        "model": "example/model",
        "model_revision": "abc123",
        "datasets": list(DATASETS),
        "modes": list(MODES),
        "prompt_modes": list(PROMPT_MODES),
        "limit": 0,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens_noncot": 0,
        "max_tokens_cot": 0,
        "chat_template_kwargs": {"noncot": {}, "cot": {}},
        "server_metadata": {"dtype": "bfloat16", "quantization": None},
    }
    _write_json(input_dir / "run_config.json", run_config)

    judged_rows = []
    template_rows = []
    question_conditions = {}
    condition_counts = {condition: 0 for condition in REQUIRED_CONDITIONS}
    for mode in MODES:
        for prompt_mode in PROMPT_MODES:
            condition = condition_for(mode, prompt_mode)
            prediction_rows = []
            for dataset in DATASETS:
                mcq_index = "mcq"
                mcq = {
                    "dataset": dataset,
                    "index": mcq_index,
                    "group": "mcq",
                    "answer_type": "mcq",
                    "options": {"A": "left", "B": "right", "C": "Cannot determine"},
                    "gt": "A",
                    "cannot_label": "C" if mode == "noimgpp" else None,
                    "mode": mode,
                    "pmode": prompt_mode,
                    "output": "C" if mode == "noimgpp" else "A",
                    "judged": "C" if mode == "noimgpp" else "A",
                    "judge_method": "paper_llm_judge",
                    "judge_attempts": 1,
                }
                prediction_rows.append(mcq)
                judged_rows.append(dict(mcq))
                template_rows.append(
                    {
                        "question_id": public_question_id(dataset, mcq_index),
                        "condition": condition,
                        "answer": "",
                    }
                )
                question_conditions.setdefault(
                    (dataset, mcq_index, "mcq"),
                    set(),
                ).add(condition)
                condition_counts[condition] += 1
                if mode != "noimgpp":
                    vqa_index = "vqa"
                    vqa = {
                        "dataset": dataset,
                        "index": vqa_index,
                        "group": "vqa",
                        "answer_type": "vqa",
                        "options": {},
                        "gt": "three",
                        "cannot_label": None,
                        "mode": mode,
                        "pmode": prompt_mode,
                        "output": "<think>count</think><answer>three</answer>",
                        "judged": "1",
                        "judge_method": "paper_llm_judge",
                        "judge_attempts": 1,
                    }
                    prediction_rows.append(vqa)
                    judged_rows.append(dict(vqa))
                    template_rows.append(
                        {
                            "question_id": public_question_id(dataset, vqa_index),
                            "condition": condition,
                            "answer": "",
                        }
                    )
                    question_conditions.setdefault(
                        (dataset, vqa_index, "vqa"),
                        set(),
                    ).add(condition)
                    condition_counts[condition] += 1
            _write_jsonl(
                input_dir / f"pred_{mode}_{prompt_mode}.jsonl",
                prediction_rows,
            )
    _write_jsonl(input_dir / "judged.jsonl", judged_rows)
    _write_jsonl(contract_dir / "submission_template.jsonl", template_rows)
    question_rows = [
        {
            "question_id": public_question_id(dataset, index),
            "dataset_key": dataset,
            "type": "2D",
            "answer_type": answer_type,
            "evaluation_group": public_question_id(dataset, index),
            "conditions": sorted(
                conditions,
                key=list(REQUIRED_CONDITIONS).index,
            ),
        }
        for (dataset, index, answer_type), conditions in sorted(
            question_conditions.items()
        )
    ]
    _write_jsonl(contract_dir / "questions.jsonl", question_rows)
    dataset_condition_counts = {
        dataset: {
            condition: sum(
                1
                for row in template_rows
                if row["condition"] == condition
                and row["question_id"].startswith(f"{dataset}:")
            )
            for condition in REQUIRED_CONDITIONS
        }
        for dataset in DATASETS
    }
    manifest = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "task_id": "spatial",
        "benchmark_version": "test-paper-v5",
        "harness_contract": HARNESS_CONTRACT,
        "harness_version": HARNESS_VERSION,
        "demo": False,
        "datasets": list(DATASETS),
        "dataset_count": len(DATASETS),
        "required_conditions": list(REQUIRED_CONDITIONS),
        "primary_condition": "main_noncot",
        "evaluation_policy": EVALUATION_POLICY,
        "condition_counts": condition_counts,
        "condition_group_counts": dict(condition_counts),
        "dataset_condition_counts": dataset_condition_counts,
        "dataset_condition_group_counts": dataset_condition_counts,
        "unique_question_ids": len(question_rows),
        "dataset_files": {
            dataset: {
                "filename": f"{dataset}.tsv",
                "size_bytes": 100,
                "sha256": "a" * 64,
                "rows": 2,
                "answer_types": {"mcq": 1, "vqa": 1},
                "source": {"revision": "test"},
            }
            for dataset in DATASETS
        },
        "data_manifest": {
            "filename": "track3_data_manifest.json",
            "sha256": "b" * 64,
        },
        "prompts": {
            "noncot": {"source": "test", "sha256": "c" * 64},
            "cot": {"source": "test", "sha256": "d" * 64},
        },
        "decoding": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1.0,
            "seed": 0,
            "completion_budget": "server_context_remainder",
        },
        "judge": {
            "model": PAPER_JUDGE_MODEL,
            "revision": PAPER_JUDGE_REVISION,
            "system_prompt_sha256": {
                "mcq": "e" * 64,
                "vqa": "f" * 64,
            },
            "decoding": JUDGE_DECODING,
        },
        "artifacts": {
            "questions": {
                "filename": "questions.jsonl",
                "rows": len(question_rows),
                "sha256": _sha(contract_dir / "questions.jsonl"),
            },
            "submission_template": {
                "filename": "submission_template.jsonl",
                "rows": len(template_rows),
                "sha256": _sha(contract_dir / "submission_template.jsonl"),
            },
        },
    }
    _write_json(contract_dir / "manifest.json", manifest)
    return input_dir, contract_dir


def test_public_vqa_answer_uses_only_the_final_answer_tag():
    assert (
        _public_vqa_answer(
            "<think>First guess: two.</think><answer>three</answer>"
        )
        == "three"
    )
    assert _public_vqa_answer("three") == "three"


def test_public_vqa_answer_commits_oversized_evidence_without_truncating():
    answer = "long answer " * 300
    normalized = answer.strip()
    commitment = _public_vqa_answer(answer)

    assert commitment.startswith(LONG_VQA_COMMITMENT_PREFIX)
    assert commitment.endswith(
        f";UTF8_BYTES:{len(normalized.encode('utf-8'))};CHARS:{len(normalized)}"
    )
    assert answer not in commitment
    assert len(commitment) < 2_048


def test_package_contains_only_public_evidence_and_three_archive_members(tmp_path):
    input_dir, contract_dir = _fixture(tmp_path)
    output_dir = tmp_path / "package"

    archive_path = package_submission(
        input_dir,
        contract_dir,
        output_dir,
        "Example Model",
    )

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "submission.jsonl",
            "run_manifest.json",
            "leaderboard.json",
        }
    evidence = [
        json.loads(line)
        for line in (output_dir / "submission.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert evidence
    assert all("gt" not in row and "output" not in row for row in evidence)
    assert all(
        set(row)
        == {
            "dataset",
            "question_id",
            "evaluation_group",
            "answer_type",
            "condition",
            "answer",
            "correct",
            "judge_method",
            "judge_attempts",
        }
        for row in evidence
    )
    vqa = next(row for row in evidence if row["answer_type"] == "vqa")
    assert vqa["answer"] == "three"
    assert vqa["judge_method"] == "paper_vqa_llm_judge"
    no_image_plus = [
        row
        for row in evidence
        if row["condition"].startswith("no_image_plus")
    ]
    assert no_image_plus
    assert all(row["answer_type"] == "mcq" for row in no_image_plus)

    report = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert report["model"]["name"] == "Example Model"
    assert report["summary"]["main_noncot"] == 1.0
    assert report["summary"]["cot_delta"] == 0.0

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["public_evidence_policy"]["vqa"].endswith(
        "sha256_commitment_above_2048_chars"
    )
    assert run_manifest["public_evidence_policy"]["oversized_vqa_commitment"] == (
        "sha256_utf8_with_byte_and_character_counts"
    )
    assert run_manifest["public_evidence_policy"]["reasoning_traces_included"] is False
    assert run_manifest["public_evidence_policy"]["private_ground_truth_included"] is False
    assert run_manifest["error_counts"] == {
        "inference": 0,
        "judge": 0,
        "missing_outputs": 0,
    }


def test_package_preserves_explicit_inference_failure_as_incorrect(tmp_path):
    input_dir, contract_dir = _fixture(tmp_path)
    marker = {
        "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
        "policy": INFERENCE_FAILURE_POLICY,
        "category": "input_context_exceeded",
        "disposition": INFERENCE_FAILURE_DISPOSITION,
        "input_length": 33432,
        "max_model_len": 32768,
    }
    prediction_path = input_dir / "pred_main_cot.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text().splitlines()
    ]
    failed_prediction = next(
        row
        for row in predictions
        if row["dataset"] == "BLINK" and row["index"] == "mcq"
    )
    failed_prediction.update(
        {
            "output": "",
            "error": (
                "BadRequestError: Input length (33432) exceeds model's "
                "maximum context length (32768)."
            ),
            "finish_reason": "inference_error",
            "terminal_failure": marker,
        }
    )
    _write_jsonl(prediction_path, predictions)

    judged_path = input_dir / "judged.jsonl"
    judged = [json.loads(line) for line in judged_path.read_text().splitlines()]
    failed_judged = next(
        row
        for row in judged
        if row["dataset"] == "BLINK"
        and row["index"] == "mcq"
        and row["mode"] == "main"
        and row["pmode"] == "cot"
    )
    failed_judged.update(
        {
            "output": "",
            "error": failed_prediction["error"],
            "finish_reason": "inference_error",
            "terminal_failure": marker,
            "judged": "0",
            "judge_method": INFERENCE_FAILURE_JUDGE_METHOD,
            "judge_attempts": 0,
        }
    )
    _write_jsonl(judged_path, judged)
    config_path = input_dir / "run_config.json"
    config = json.loads(config_path.read_text())
    config["terminal_inference_failures"] = {
        "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
        "policy": INFERENCE_FAILURE_POLICY,
        "disposition": INFERENCE_FAILURE_DISPOSITION,
        "eligible_category": "input_context_exceeded",
        "configured_max_model_len": 32768,
        "unique_samples": 1,
        "condition_rows": 1,
        "sample_ids": ["BLINK:mcq"],
        "condition_counts": {"main_cot": 1},
        "finalized_at": "2026-07-29T00:00:00+00:00",
    }
    _write_json(config_path, config)

    output_dir = tmp_path / "package"
    archive_path = package_submission(input_dir, contract_dir, output_dir)

    evidence = [
        json.loads(line)
        for line in (output_dir / "submission.jsonl").read_text().splitlines()
    ]
    failed = next(
        row
        for row in evidence
        if row["dataset"] == "BLINK"
        and row["question_id"] == "BLINK:mcq"
        and row["condition"] == "main_cot"
    )
    assert failed["answer"] == "0"
    assert failed["correct"] is False
    assert failed["judge_method"] == INFERENCE_FAILURE_JUDGE_METHOD
    assert failed["judge_attempts"] == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["error_counts"] == {
        "inference": 1,
        "judge": 0,
        "missing_outputs": 1,
    }
    assert manifest["inference_failure_policy"]["condition_rows"] == 1

    submission_bytes, manifest_bytes, report_bytes = (
        spatial_submission.read_spatial_submission_archive(
            archive_path.read_bytes()
        )
    )
    records, computed_report, _benchmark_manifest = (
        spatial_submission.parse_spatial_evidence(
            submission_bytes,
            contract_dir / "manifest.json",
            contract_dir / "submission_template.jsonl",
            contract_dir / "questions.jsonl",
        )
    )
    spatial_submission.validate_spatial_report(
        report_bytes,
        "example/model",
        computed_report,
    )
    metadata = spatial_submission.validate_run_manifest(
        manifest_bytes,
        submission_bytes,
        report_bytes,
        "example/model",
        records,
        contract_dir / "manifest.json",
    )
    assert metadata["harness_contract"] == HARNESS_CONTRACT


def test_backend_accepts_the_paper_aligned_public_package(tmp_path):
    input_dir, contract_dir = _fixture(tmp_path)
    output_dir = tmp_path / "package"
    archive_path = package_submission(
        input_dir,
        contract_dir,
        output_dir,
        "Example Model",
    )
    submission_bytes, manifest_bytes, report_bytes = (
        spatial_submission.read_spatial_submission_archive(
            archive_path.read_bytes()
        )
    )
    records, computed_report, benchmark_manifest = (
        spatial_submission.parse_spatial_evidence(
            submission_bytes,
            contract_dir / "manifest.json",
            contract_dir / "submission_template.jsonl",
            contract_dir / "questions.jsonl",
        )
    )
    assert benchmark_manifest["benchmark_version"] == "test-paper-v5"
    assert {row["answer_type"] for row in records} == {"mcq", "vqa"}
    spatial_submission.validate_spatial_report(
        report_bytes,
        "Example Model",
        computed_report,
    )
    metadata = spatial_submission.validate_run_manifest(
        manifest_bytes,
        submission_bytes,
        report_bytes,
        "Example Model",
        records,
        contract_dir / "manifest.json",
    )
    assert metadata["harness_contract"] == HARNESS_CONTRACT
    assert metadata["judge_revision"] == PAPER_JUDGE_REVISION
