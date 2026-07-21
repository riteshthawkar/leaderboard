import hashlib
import json
import os
from pathlib import Path

import pytest

from evaluation.finalize_visual_results import (
    CURRENT_PIPELINE_REVISION,
    FinalizationError,
    build_canonical_results,
    canonicalize_track_rows,
    discover_candidates,
    prune_cache,
    prune_source_runs,
    read_json,
    read_jsonl,
    sha256,
    validate_submission,
    verify_canonical_results,
)
from evaluation.common.visual_pipeline import (
    INVALID_FORMAT_ANSWER,
    MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
    MANDATORY_EXTRACTOR_MODEL_ID,
    MANDATORY_EXTRACTOR_MODEL_REVISION,
    MANDATORY_EXTRACTOR_PROMPT_SHA256,
)


TRACK_IDS = {
    "do_you_see_me": ["d1", "d2"],
    "minds_eye": ["m1", "m2"],
}


def _extracted_diagnostic(
    question_id: str,
    output: str,
    answer: str | None = "A",
    *,
    answer_type: str = "text",
    status: str = "resolved",
) -> dict:
    row = {
        "question_id": question_id,
        "answer_type": answer_type,
        "output": output,
        "answer_extraction_method": MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
        "extractor_status": status,
        "extractor_model": MANDATORY_EXTRACTOR_MODEL_ID,
        "extractor_model_revision": MANDATORY_EXTRACTOR_MODEL_REVISION,
        "extractor_quantization": "unquantized",
        "extractor_runtime": "vllm 0.25.1",
        "extractor_prompt_sha256": MANDATORY_EXTRACTOR_PROMPT_SHA256,
        "extractor_ground_truth_access": False,
        "extractor_image_access": False,
        "extractor_source_diagnostics": "diagnostics.jsonl",
        "extractor_source_output_sha256": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
    }
    if answer is not None:
        row["extracted_answer"] = answer
        row["extractor_output"] = f"<answer>{answer}</answer>"
    else:
        row["extractor_output"] = "<answer>UNRESOLVED</answer>"
        row["extractor_error"] = "The extractor returned UNRESOLVED."
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_questions(project_root: Path) -> None:
    for track, question_ids in TRACK_IDS.items():
        _write_jsonl(
            project_root / "tasks" / track / "questions.jsonl",
            [
                {"question_id": question_id, "answer_type": "text"}
                for question_id in question_ids
            ],
        )


def _make_candidate(
    results_root: Path,
    relative_dir: str,
    model_id: str,
    revision: str,
    track: str,
    modified_at: float,
    *,
    loading: str = "unquantized",
    complete_answer: str = "A",
) -> Path:
    directory = results_root / relative_dir
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": model_id,
        "model_revision": revision,
        "weight_loading": loading,
        "compute_dtype": "bfloat16",
        "pipeline_revision": CURRENT_PIPELINE_REVISION,
        "serving_engine": {"name": "vllm", "version": "test"},
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "request_concurrency": 1,
        "max_model_len": 32768,
        "generation": {track: {"temperature": 0.0}},
    }
    (directory / ".run_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    question_ids = TRACK_IDS[track]
    submission = directory / f"{track}_submission.jsonl"
    _write_jsonl(
        submission,
        [
            {
                "question_id": question_id,
                "condition": "standard",
                "answer": complete_answer,
            }
            for question_id in question_ids
        ],
    )
    _write_jsonl(
        directory / f"{track}.diagnostics.jsonl",
        [
            {
                **_extracted_diagnostic(
                    question_id,
                    f"<answer>{complete_answer}</answer>",
                    complete_answer,
                ),
                "finish_reason": "stop",
            }
            for question_id in question_ids
        ],
    )
    os.utime(submission, (modified_at, modified_at))
    return directory


def test_builds_canonical_results_from_newest_complete_bf16_tracks(tmp_path):
    project_root = tmp_path / "project"
    results_root = project_root / "evaluation" / "results"
    output_root = results_root / "final"
    _write_questions(project_root)

    _make_candidate(
        results_root,
        "visual_suite_old/model-a",
        "org/model-a",
        "revision-a",
        "do_you_see_me",
        100,
    )
    newest_dys = _make_candidate(
        results_root,
        "visual_suite_new/model-a",
        "org/model-a",
        "revision-a",
        "do_you_see_me",
        300,
    )
    minds_eye = _make_candidate(
        results_root,
        "visual_suite_me/model-a",
        "org/model-a",
        "revision-a",
        "minds_eye",
        200,
    )
    _make_candidate(
        results_root,
        "visual_suite_quantized/model-a",
        "org/model-a",
        "revision-a",
        "do_you_see_me",
        400,
        loading="bnb4",
    )
    _make_candidate(
        results_root,
        "visual_suite_incomplete/model-b",
        "org/model-b",
        "revision-b",
        "do_you_see_me",
        500,
    )

    index = build_canonical_results(results_root, output_root, project_root)

    assert index["model_count"] == 1
    model_dir = output_root / "model-a"
    manifest_path = model_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "org/model-a"
    assert manifest["selection_policy"]["canonical_answer_parser"] == (
        "llm-extractor-output-contract-parser-v9"
    )
    assert manifest["selection_policy"]["answer_acceptance"] == (
        "mandatory-independent-llm-extraction"
    )
    assert manifest["selection_policy"]["invalid_format_submission_value"] == (
        INVALID_FORMAT_ANSWER
    )
    assert manifest["selection_policy"]["letter_disambiguation_max_length"] == 9
    assert manifest["tracks"]["do_you_see_me"]["source_run"] == str(
        newest_dys.relative_to(results_root)
    )
    assert manifest["tracks"]["minds_eye"]["source_run"] == str(
        minds_eye.relative_to(results_root)
    )
    for track in TRACK_IDS:
        assert manifest["tracks"][track]["row_count"] == 2
        assert manifest["tracks"][track]["strict_answer_count"] == 2
        assert manifest["tracks"][track]["invalid_format_count"] == 0
        submission = model_dir / f"{track}_submission.jsonl"
        assert (
            manifest["tracks"][track]["artifacts"][submission.name]["sha256"]
            == sha256(submission)
        )
    assert index["models"][0]["manifest_sha256"] == sha256(manifest_path)


def test_finalizer_replaces_legacy_long_raw_fallback_with_bounded_marker(tmp_path):
    project_root = tmp_path / "project"
    results_root = project_root / "evaluation" / "results"
    output_root = results_root / "final"
    _write_questions(project_root)
    for track in TRACK_IDS:
        directory = _make_candidate(
            results_root,
            f"visual_suite/model-a-{track}",
            "org/model-a",
            "revision-a",
            track,
            100,
        )
        if track != "minds_eye":
            continue
        raw_output = "<think>" + ("unfinished reasoning " * 2_000)
        assert len(raw_output) > 30_000
        _write_jsonl(
            directory / f"{track}_submission.jsonl",
            [
                {"question_id": "m1", "condition": "standard", "answer": "A"},
                {
                    "question_id": "m2",
                    "condition": "standard",
                    "answer": INVALID_FORMAT_ANSWER,
                },
            ],
        )
        _write_jsonl(
            directory / f"{track}.diagnostics.jsonl",
            [
                _extracted_diagnostic("m1", "<answer>A</answer>", "A"),
                {
                    **_extracted_diagnostic(
                        "m2", raw_output, None, status="unresolved"
                    ),
                    "submission_status": "invalid_format",
                    "format_failure_reason": (
                        "unresolved_or_unsupported_after_independent_extraction"
                    ),
                    "raw_output_characters": len(raw_output),
                    "raw_output_bytes": len(raw_output.encode("utf-8")),
                    "raw_output_sha256": hashlib.sha256(
                        raw_output.encode("utf-8")
                    ).hexdigest(),
                },
            ],
        )

    index = build_canonical_results(results_root, output_root, project_root)

    model_dir = output_root / index["models"][0]["slug"]
    submissions = read_jsonl(model_dir / "minds_eye_submission.jsonl")
    diagnostics = read_jsonl(model_dir / "minds_eye.diagnostics.jsonl")
    assert submissions[1]["answer"] == INVALID_FORMAT_ANSWER
    assert len(submissions[1]["answer"]) < 100
    assert diagnostics[1]["output"] == raw_output
    assert diagnostics[1]["submission_status"] == "invalid_format"
    assert diagnostics[1]["raw_output_characters"] == len(raw_output)
    assert diagnostics[1]["raw_output_sha256"] == hashlib.sha256(
        raw_output.encode("utf-8")
    ).hexdigest()
    manifest = read_json(model_dir / "final_manifest.json")
    assert manifest["schema_version"] == 2
    assert manifest["tracks"]["minds_eye"]["invalid_format_count"] == 1
    assert verify_canonical_results(output_root, project_root)["model_count"] == 1


@pytest.mark.parametrize(
    ("answer_type", "task", "output", "legacy_answer"),
    (
        ("text", "form_constancy", "No.", "No"),
        (
            "integer",
            "visual_spatial",
            "There is one square at position (row 2, column 2).",
            "1",
        ),
        (
            "integer",
            "shape_discrimination",
            "There are two cylinders in total.",
            "2",
        ),
        (
            "integer",
            "joint_shape_color",
            "The final answer is \\boxed{3}.",
            "3",
        ),
    ),
)
def test_canonicalizer_refuses_deterministic_legacy_answers(
    answer_type, task, output, legacy_answer
):
    with pytest.raises(FinalizationError, match="mandatory independent extractor"):
        canonicalize_track_rows(
            [
                {
                    "question_id": "q1",
                    "condition": "standard",
                    "answer": legacy_answer,
                }
            ],
            [{"question_id": "q1", "answer_type": answer_type, "output": output}],
            {
                "q1": {
                    "question_id": "q1",
                    "answer_type": answer_type,
                    "task": task,
                }
            },
        )


def test_new_staging_model_merges_with_canonical_baseline_after_prune(tmp_path):
    project_root = tmp_path / "project"
    results_root = project_root / "evaluation" / "results"
    output_root = results_root / "final"
    _write_questions(project_root)
    for track in TRACK_IDS:
        _make_candidate(
            results_root,
            f"visual_suite_model_a/model-a-{track}",
            "org/model-a",
            "revision-a",
            track,
            100,
        )
    first_index = build_canonical_results(results_root, output_root, project_root)
    assert first_index["model_count"] == 1
    prune_source_runs(results_root, output_root)

    for track in TRACK_IDS:
        _make_candidate(
            results_root,
            f"visual_suite_model_b/model-b-{track}",
            "org/model-b",
            "revision-b",
            track,
            200,
        )
    second_index = build_canonical_results(results_root, output_root, project_root)

    assert second_index["model_count"] == 2
    assert {record["model_id"] for record in second_index["models"]} == {
        "org/model-a",
        "org/model-b",
    }
    assert verify_canonical_results(output_root, project_root)["model_count"] == 2


def test_prune_removes_only_visual_suite_source_roots(tmp_path):
    results_root = tmp_path / "results"
    output_root = results_root / "final"
    output_root.mkdir(parents=True)
    (results_root / ".cache").mkdir()
    (results_root / "notes").mkdir()
    (results_root / "visual_suite_old").mkdir()
    (results_root / "visual_suite_failed").mkdir()

    removed = prune_source_runs(results_root, output_root)

    assert removed == ["visual_suite_failed", "visual_suite_old"]
    assert output_root.is_dir()
    assert (results_root / ".cache").is_dir()
    assert (results_root / "notes").is_dir()


def test_prune_refuses_live_evaluation_marker(tmp_path):
    results_root = tmp_path / "results"
    output_root = results_root / "final"
    output_root.mkdir(parents=True)
    active_root = results_root / "visual_suite_active"
    active_root.mkdir()
    (active_root / ".active-run.json").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )

    with pytest.raises(FinalizationError, match="active evaluation"):
        prune_source_runs(results_root, output_root)

    assert active_root.is_dir()


def test_live_run_is_excluded_and_blocks_cache_pruning(tmp_path):
    project_root = tmp_path / "project"
    results_root = project_root / "evaluation" / "results"
    output_root = results_root / "final"
    output_root.mkdir(parents=True)
    (results_root / ".cache").mkdir()
    _write_questions(project_root)
    active_source = _make_candidate(
        results_root,
        "visual_suite_active/model-a",
        "org/model-a",
        "revision-a",
        "do_you_see_me",
        100,
    )
    (active_source.parent / ".active-run.json").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )

    assert discover_candidates(results_root, output_root, project_root) == []
    with pytest.raises(FinalizationError, match="active runs"):
        prune_cache(results_root)
    assert (results_root / ".cache").is_dir()


def test_cache_pruning_is_explicit(tmp_path):
    results_root = tmp_path / "results"
    cache = results_root / ".cache"
    cache.mkdir(parents=True)
    (cache / "object").write_text("cached", encoding="utf-8")

    assert prune_cache(results_root) is True
    assert not cache.exists()
    assert prune_cache(results_root) is False


def test_submission_validation_rejects_duplicate_ids(tmp_path):
    submission = tmp_path / "submission.jsonl"
    _write_jsonl(
        submission,
        [
            {"question_id": "d1", "condition": "standard", "answer": "A"},
            {"question_id": "d1", "condition": "standard", "answer": "B"},
        ],
    )

    with pytest.raises(FinalizationError, match="duplicate"):
        validate_submission(submission, TRACK_IDS["do_you_see_me"])


def test_verify_rejects_tampered_canonical_artifact(tmp_path):
    project_root = tmp_path / "project"
    results_root = project_root / "evaluation" / "results"
    output_root = results_root / "final"
    _write_questions(project_root)
    for track in TRACK_IDS:
        _make_candidate(
            results_root,
            f"visual_suite_source/model-a-{track}",
            "org/model-a",
            "revision-a",
            track,
            100,
        )
    build_canonical_results(results_root, output_root, project_root)
    submission = next(output_root.glob("*/do_you_see_me_submission.jsonl"))
    submission.write_text(submission.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(FinalizationError, match="hash mismatch"):
        verify_canonical_results(output_root, project_root)
