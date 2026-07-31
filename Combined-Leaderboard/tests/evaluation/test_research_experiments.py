import json
import hashlib
import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analysis.research.audit_inference_quality import audit
from analysis.research.compare_causal_runs import compare
from analysis.research.offline_complementarity import assign_folds
from analysis.research.analyze_qwen35_thinking_ablation import (
    CONDITIONS as QWEN35_CONDITIONS,
    factorial_statistics,
)
from analysis.research import score_analysis_queue, score_condition_matrix
from evaluation.common.vllm_runner import ANSWER_EXTRACTION_METHOD
from evaluation.research.generate_causal_transformations import build_bundle
from evaluation.research.generate_perception_curriculum import build as build_curriculum
from evaluation.research.finalize_persistent_extractor_failures import (
    POLICY as PERSISTENT_EXTRACTOR_POLICY,
    UNPARSEABLE_ERROR,
    finalize_persistent_failures,
)
from evaluation.research.prepare_state_interventions import prepare as prepare_states
from evaluation.research.prepare_visual_fidelity import prepare as prepare_fidelity
from evaluation.research.record_experiment_run import automatic_source_inputs
from evaluation.research.run_state_extractor_vllm import validate_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_GPU_CAUSAL_SCRIPT = (
    PROJECT_ROOT / "evaluation/research/run_single_gpu_causal_analysis.sh"
)
SINGLE_GPU_ANALYSIS_SCRIPT = (
    PROJECT_ROOT / "evaluation/research/run_single_gpu_analysis_queue.sh"
)
CONDITION_MATRIX_SCRIPT = (
    PROJECT_ROOT / "evaluation/research/run_visual_condition_matrix.sh"
)
QWEN35_ABLATION_SCRIPT = (
    PROJECT_ROOT / "evaluation/research/run_qwen35_thinking_ablation.sh"
)


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_fold_assignment_is_stratified_and_complete():
    groups = {
        "a": np.asarray([0, 1, 2, 3, 4]),
        "b": np.asarray([5, 6, 7, 8, 9]),
    }
    assignments = assign_folds(groups, 5, np.random.default_rng(7))
    assert sorted(assignments.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert np.all(assignments >= 0)


def test_causal_bundle_changes_only_the_target_answer_contract(tmp_path):
    manifest = build_bundle(tmp_path, pairs=2, seed=17)
    truth = read_jsonl(tmp_path / "private_ground_truth.jsonl")
    by_pair = {}
    for row in truth:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = row["answer"]
    assert manifest["item_count"] == 6
    for answers in by_pair.values():
        assert answers["base"] == answers["nuisance"]
        assert answers["base"] != answers["causal"]
    assert all(
        row["base_operation"] != row["causal_operation"]
        for row in read_jsonl(tmp_path / "pairs.jsonl")
    )
    assert len(read_jsonl(tmp_path / "questions.jsonl")) == 6
    assert len(list((tmp_path / "images").glob("*.png"))) == 6


def test_state_interventions_are_answer_blind_and_mismatched(tmp_path):
    questions = PROJECT_ROOT / "tasks/minds_eye/questions.jsonl"
    manifest = prepare_states(questions, tmp_path, seed=19)
    assert manifest["item_count"] == 199
    items = read_jsonl(tmp_path / "items.jsonl")
    assert all(
        row["oracle_abstraction"] != row["mismatched_abstraction"]
        for row in items
    )
    assert all("answer" not in row["oracle_abstraction"].casefold() for row in items)


def test_state_validator_rejects_answer_leakage():
    validate_state(
        "dynamic_isomorphism",
        {"frames": [{"visible_objects": [{"position": [1, 2]}]}]},
    )
    with pytest.raises(ValueError, match="answer-bearing"):
        validate_state(
            "dynamic_isomorphism",
            {"frames": [{"visible_objects": []}], "correct_answer": "B"},
        )
    with pytest.raises(ValueError, match="answer-bearing"):
        validate_state(
            "slippage",
            {"panels": [{"description": "The outlier is panel C."}]},
        )


def test_fidelity_preparation_uses_matched_ids_and_dimensions(tmp_path):
    dataset_root = tmp_path / "dataset"
    question_root = tmp_path / "questions"
    question_root.mkdir()
    rows_by_track = {
        "do_you_see_me": [
            {
                "question_id": "dysm_test_0",
                "track": "t1_2d",
                "task": "shape_discrimination",
                "difficulty": "easy",
                "question": "Count.",
                "answer_type": "integer",
                "source_subset": "dysm_2d_v1",
                "image": "images/test.png",
                "image_url": "",
            }
        ],
        "minds_eye": [
            {
                "question_id": "minds_test_0",
                "track": "t2",
                "task": "mrt",
                "difficulty": None,
                "question": "Choose.",
                "answer_type": "mcq_letter",
                "source_subset": "minds_eye_fresh_v1",
                "image": "images/test.png",
                "image_url": "",
            }
        ],
    }
    for track, rows in rows_by_track.items():
        path = question_root / f"{track}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        subset = rows[0]["source_subset"]
        image_path = dataset_root / subset / "images/test.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        array = np.indices((63, 63)).sum(axis=0) % 2
        pixels = np.repeat((array * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        Image.fromarray(pixels).save(image_path)

    output = tmp_path / "output"
    manifest = prepare_fidelity(
        output=output,
        dysm_questions=question_root / "do_you_see_me.jsonl",
        minds_eye_questions=question_root / "minds_eye.jsonl",
        dysm_per_stratum=1,
        minds_eye_per_task=1,
        seed=23,
        dataset_root=dataset_root,
        workers=2,
    )
    assert manifest["item_count"] == 2
    for track in rows_by_track:
        identifiers = []
        sizes = []
        for variant in ("native", "downsample_50", "downsample_25"):
            questions = read_jsonl(output / track / variant / "questions.jsonl")
            identifiers.append([row["question_id"] for row in questions])
            with Image.open(output / track / variant / questions[0]["image"]) as image:
                sizes.append(image.size)
        assert identifiers[0] == identifiers[1] == identifiers[2]
        assert sizes == [(63, 63), (63, 63), (63, 63)]


def test_curriculum_conditions_are_prompt_and_answer_matched(tmp_path):
    manifest = build_curriculum(tmp_path, train_count=5, validation_count=5, seed=29)
    assert manifest["item_count"] == 20
    for split in ("train", "validation"):
        curriculum = json.loads(
            (tmp_path / f"annotations/perception_curriculum_{split}.json").read_text()
        )
        control = json.loads(
            (tmp_path / f"annotations/recognition_control_{split}.json").read_text()
        )
        assert len(curriculum) == len(control) == 5
        for target, baseline in zip(curriculum, control, strict=True):
            assert target["conversations"] == baseline["conversations"]
            assert (tmp_path / target["image"]).is_file()
            assert (tmp_path / baseline["image"]).is_file()


def test_single_gpu_causal_run_replaces_temporary_extraction_before_scoring():
    script = SINGLE_GPU_CAUSAL_SCRIPT.read_text(encoding="utf-8")
    assert 'GPU_ID="${GPU_ID:-2}"' in script
    assert "--dtype bfloat16" in script
    assert "--extract-existing-diagnostics" in script
    assert 'EXTRACTOR_MODEL="Qwen/Qwen3-8B"' in script
    assert "run_fixed_extraction" in script
    assert "score_causal_transformations.py" in script


def test_single_gpu_analysis_queue_uses_full_precision_and_deferred_extraction():
    script = SINGLE_GPU_ANALYSIS_SCRIPT.read_text(encoding="utf-8")
    assert 'GPU_ID="${GPU_ID:-2}"' in script
    assert "--dtype bfloat16" in script
    assert "--defer-extraction" in script
    assert "--extract-existing-diagnostics" in script
    assert "--max-tokens" not in script
    assert 'TARGET_SLUGS="${TARGET_SLUGS:-qwen3-vl-8b,internvl35-8b}"' in script
    assert 'START_PHASE="${START_PHASE:-inference}"' in script
    assert 'if [[ "$START_PHASE" == "inference" ]]' in script
    assert "inference | extraction | scoring" in script
    assert 'if [[ "$START_PHASE" != "scoring" ]]' in script
    assert '"resuming_from_fixed_extraction"' in script
    assert "audit_inference_quality.py" in script
    assert "finalize_persistent_extractor_failures.py" in script
    assert 'EXTRACTOR_ATTEMPTS="${EXTRACTOR_ATTEMPTS:-3}"' in script
    assert '|| return 0' in script
    assert "compare_causal_runs.py" in script


def test_persistent_unparseable_extractor_output_is_audited_as_unresolved(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    submission = tmp_path / "submission.jsonl"
    report = tmp_path / "quality/persistent_extractor_resolution.json"
    raw_output = "repeated output without a final answer"
    source_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    diagnostics.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "output": raw_output,
                "finish_reason": "length",
                "answer_extraction_method": ANSWER_EXTRACTION_METHOD,
                "extractor_model": "Qwen/Qwen3-8B",
                "extractor_revision": "revision",
                "extractor_output": "still unparseable",
                "extractor_finish_reason": "length",
                "extractor_status": "failed",
                "extractor_error": UNPARSEABLE_ERROR,
                "extractor_source_output_sha256": source_hash,
                "extracted_answer": "UNRESOLVED",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    submission.write_text(
        json.dumps({"question_id": "q1", "condition": "standard", "answer": "old"})
        + "\n",
        encoding="utf-8",
    )

    result = finalize_persistent_failures(
        diagnostics_path=diagnostics,
        submission_path=submission,
        report_path=report,
        extractor_model="Qwen/Qwen3-8B",
        extractor_revision="revision",
        attempts=3,
    )

    finalized = read_jsonl(diagnostics)[0]
    assert finalized["extractor_status"] == "unresolved"
    assert finalized["extracted_answer"] == "UNRESOLVED"
    assert finalized["extractor_error"] == UNPARSEABLE_ERROR
    assert finalized["extractor_terminal_resolution"] == {
        "policy": PERSISTENT_EXTRACTOR_POLICY,
        "attempts": 3,
        "disposition": "unresolved_scored_incorrect",
        "original_status": "failed",
        "original_error": UNPARSEABLE_ERROR,
    }
    assert read_jsonl(submission)[0]["answer"] == "UNRESOLVED"
    assert result["finalized_count"] == 1
    assert json.loads(report.read_text(encoding="utf-8"))["question_ids"] == ["q1"]


def test_persistent_extractor_finalizer_rejects_transport_failures(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    submission = tmp_path / "submission.jsonl"
    report = tmp_path / "report.json"
    diagnostics.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "output": "candidate",
                "finish_reason": "stop",
                "answer_extraction_method": ANSWER_EXTRACTION_METHOD,
                "extractor_model": "Qwen/Qwen3-8B",
                "extractor_revision": "revision",
                "extractor_status": "failed",
                "extractor_error": "HTTPError: timeout",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    submission.write_text(
        json.dumps({"question_id": "q1", "condition": "standard", "answer": "old"})
        + "\n",
        encoding="utf-8",
    )
    before = diagnostics.read_bytes()

    with pytest.raises(ValueError, match="not a completed unparseable"):
        finalize_persistent_failures(
            diagnostics_path=diagnostics,
            submission_path=submission,
            report_path=report,
            extractor_model="Qwen/Qwen3-8B",
            extractor_revision="revision",
            attempts=3,
        )

    assert diagnostics.read_bytes() == before
    assert not report.exists()


def test_condition_matrix_defaults_to_remaining_context_budget():
    script = CONDITION_MATRIX_SCRIPT.read_text(encoding="utf-8")
    assert 'MAX_TOKENS="${MAX_TOKENS:-0}"' in script
    assert 'max_token_args=(--max-tokens "${MAX_TOKENS}")' in script
    assert '"${max_token_args[@]}"' in script


def test_run_provenance_hashes_exact_harness_and_prompt_sources():
    inputs = automatic_source_inputs(
        experiment="controlled_visual_analysis",
        parameters={"track": "minds_eye", "prompt_mode": "cot"},
    )

    assert set(inputs) == {
        "source_analysis_plan",
        "source_manifest_writer",
        "source_orchestration",
        "source_prompt_template",
        "source_track_runner",
        "source_visual_pipeline",
        "source_vllm_runner",
    }
    assert inputs["source_prompt_template"].name == "cot.txt"
    assert all(path.is_file() for path in inputs.values())


def test_qwen35_ablation_has_a_separate_frozen_plan_and_full_precision_contract():
    script = QWEN35_ABLATION_SCRIPT.read_text(encoding="utf-8")
    assert "--dtype bfloat16" in script
    assert "--defer-extraction" in script
    assert "--extract-existing-diagnostics" in script
    assert "--max-tokens" not in script
    assert '{"enable_thinking":false}' in script
    assert '{"enable_thinking":true}' in script
    inputs = automatic_source_inputs(
        experiment="qwen35_thinking_ablation",
        parameters={"track": "minds_eye", "prompt_mode": "cot"},
    )
    assert (
        inputs["source_analysis_plan"].name
        == "PRESPECIFIED_QWEN35_THINKING_PLAN.md"
    )
    assert inputs["source_orchestration"] == QWEN35_ABLATION_SCRIPT


def test_qwen35_factorial_statistics_preserve_paired_interaction():
    metadata = [
        {"capability": "first"} for _ in range(4)
    ] + [{"capability": "second"} for _ in range(4)]
    arrays = {
        "direct_disabled": np.asarray([0, 0, 0, 0, 0, 0, 0, 0]),
        "direct_enabled": np.asarray([1, 1, 0, 0, 1, 1, 0, 0]),
        "cot_disabled": np.asarray([1, 0, 0, 0, 1, 0, 0, 0]),
        "cot_enabled": np.asarray([1, 1, 1, 1, 1, 1, 1, 0]),
    }

    rows = factorial_statistics(
        arrays,
        metadata,
        bootstrap_replicates=100,
        randomization_replicates=100,
        seed=31,
    )

    assert len(rows) == 5
    assert set(arrays) == set(QWEN35_CONDITIONS)
    by_endpoint = {row["endpoint"]: row for row in rows}
    assert by_endpoint["thinking_effect_direct_prompt"]["effect"] == pytest.approx(
        0.5
    )
    assert by_endpoint["thinking_effect_reasoning_prompt"][
        "effect"
    ] == pytest.approx(0.625)
    assert by_endpoint["reasoning_prompt_effect_thinking_disabled"][
        "effect"
    ] == pytest.approx(0.25)
    assert by_endpoint["reasoning_prompt_effect_thinking_enabled"][
        "effect"
    ] == pytest.approx(0.375)
    assert by_endpoint["prompt_by_thinking_interaction"][
        "effect"
    ] == pytest.approx(0.125)
    assert all(0 <= row["holm_adjusted_p"] <= 1 for row in rows)


def test_inference_quality_gate_accepts_length_as_model_outcome(tmp_path):
    questions = tmp_path / "questions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    submission = tmp_path / "submission.jsonl"
    extractor_model = "test/extractor"
    extractor_revision = "abc123"
    raw_outputs = {"q1": "Final answer: A", "q2": "unfinished reasoning"}
    questions.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "question": "Choose.",
                    "answer_type": "mcq_letter",
                }
            )
            for question_id in raw_outputs
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "output": raw_output,
                    "finish_reason": "stop" if question_id == "q1" else "length",
                    "completion_tokens": 10 if question_id == "q1" else 32700,
                    "answer_extraction_method": ANSWER_EXTRACTION_METHOD,
                    "extractor_model": extractor_model,
                    "extractor_revision": extractor_revision,
                    "extractor_source_output_sha256": hashlib.sha256(
                        raw_output.encode("utf-8")
                    ).hexdigest(),
                    "extractor_status": (
                        "resolved" if question_id == "q1" else "unresolved"
                    ),
                    "extracted_answer": "A" if question_id == "q1" else "UNRESOLVED",
                }
            )
            for question_id, raw_output in raw_outputs.items()
        )
        + "\n",
        encoding="utf-8",
    )
    submission.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "condition": "standard",
                    "answer": "A" if question_id == "q1" else "UNRESOLVED",
                }
            )
            for question_id in raw_outputs
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit(
        questions_path=questions,
        diagnostics_path=diagnostics,
        submission_path=submission,
        output=tmp_path / "audit",
        extractor_model=extractor_model,
        extractor_revision=extractor_revision,
    )

    assert summary["quality_gate_passed"] is True
    assert summary["finish_reason_counts"] == {"length": 1, "stop": 1}
    assert summary["unresolved_by_finish_reason"] == {"length": 1}
    assert summary["inputs"]["questions"]["sha256"] == hashlib.sha256(
        questions.read_bytes()
    ).hexdigest()
    assert summary["inputs"]["diagnostics"]["bytes"] == diagnostics.stat().st_size


def test_causal_run_comparison_applies_holm_correction(tmp_path):
    fieldnames = [
        "pair_id",
        "base_operation",
        "causal_operation",
        "base_answer",
        "causal_answer",
        "nuisance_answer",
        "base_correct",
        "causal_correct",
        "nuisance_correct",
        "causal_answer_flip",
        "nuisance_answer_invariant",
        "causal_pair_success",
        "nuisance_pair_success",
        "full_triplet_success",
    ]
    paths = {}
    for name, values in (
        ("direct", (0, 0)),
        ("cot", (1, 0)),
    ):
        path = tmp_path / f"{name}.csv"
        paths[name] = path
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow(
                    {
                        "pair_id": f"p{index}",
                        "base_operation": "rotate",
                        "causal_operation": "flip",
                        "base_answer": "A",
                        "causal_answer": "B",
                        "nuisance_answer": "A",
                        **{
                            metric: value
                            for metric in fieldnames[6:]
                        },
                    }
                )
    summary = compare(
        baseline_name="direct",
        baseline_path=paths["direct"],
        candidate_name="cot",
        candidate_path=paths["cot"],
        output=tmp_path / "comparison",
        bootstrap_replicates=100,
        seed=31,
    )

    assert len(summary["comparisons"]) == 8
    assert all(
        row["holm_adjusted_p"] >= row["mcnemar_exact_p"]
        for row in summary["comparisons"]
    )


def test_condition_matrix_reports_corrected_subgroup_effects(tmp_path, monkeypatch):
    class FakeScorer:
        primary_condition = "standard"
        ground_truth = {
            "q1": {"dimension": "2D", "capability": "shape"},
            "q2": {"dimension": "2D", "capability": "shape"},
            "q3": {"dimension": "3D", "capability": "space"},
            "q4": {"dimension": "3D", "capability": "space"},
        }

        def __init__(self, track):
            assert track == "do_you_see_me"

        def _grade_condition(self, question_id, answer, condition):
            assert condition == "standard"
            return answer == "1"

    monkeypatch.setattr(score_condition_matrix, "TaskScorer", FakeScorer)
    paths = {}
    for name, answers in (
        ("native", ("1", "0", "1", "0")),
        ("degraded", ("1", "1", "0", "0")),
    ):
        path = tmp_path / f"{name}.jsonl"
        paths[name] = path
        path.write_text(
            "\n".join(
                json.dumps({"question_id": f"q{index}", "answer": answer})
                for index, answer in enumerate(answers, start=1)
            )
            + "\n",
            encoding="utf-8",
        )

    summary = score_condition_matrix.score_conditions(
        track="do_you_see_me",
        condition_paths=list(paths.items()),
        baseline_name="native",
        output=tmp_path / "score",
        bootstrap_replicates=100,
        seed=37,
    )

    assert len(summary["subgroup_comparisons"]) == 2
    assert all(
        row["holm_adjusted_p"] >= row["mcnemar_exact_p"]
        for row in summary["subgroup_comparisons"]
    )
    assert all(
        0 < row["paired_randomization_p"] <= 1
        for row in summary["comparisons"]
    )
    item_rows = list(
        csv.DictReader(
            (tmp_path / "score/item_results.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert len(item_rows) == 4
    assert set(item_rows[0]) == {
        "question_id",
        "subgroup",
        "native",
        "degraded",
    }
    assert (tmp_path / "score/subgroup_paired_comparisons.csv").is_file()


def test_cross_model_interaction_preserves_benchmark_aggregation():
    contrast = np.asarray([1.0, 1.0, -1.0, -1.0])
    metadata = [
        {"dimension": "2D", "capability": "shape"},
        {"dimension": "2D", "capability": "shape"},
        {"dimension": "3D", "capability": "space"},
        {"dimension": "3D", "capability": "space"},
    ]

    result = score_analysis_queue.interaction_statistics(
        contrast=contrast,
        track="do_you_see_me",
        metadata=metadata,
        bootstrap_replicates=100,
        randomization_replicates=100,
        seed=41,
    )

    assert result["effect"] == pytest.approx(0.0)
    assert result["interval_low"] <= 0 <= result["interval_high"]
    assert 0 < result["raw_p"] <= 1
