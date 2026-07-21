import asyncio
import hashlib
import json

import pytest

from evaluation.common.visual_pipeline import INVALID_FORMAT_ANSWER
from evaluation.common.vllm_runner import LOCAL_ANSWER_EXTRACTION_METHOD
from evaluation.finalize_visual_results import (
    FinalizationError,
    read_jsonl,
    sha256,
)
from evaluation.recover_visual_answers import (
    LOCAL_EXTRACTOR_PROMPT_SHA256,
    ExtractionItem,
    _checkpoint_contract,
    _extract,
    _load_checkpoint,
    _stage_results,
    _verify_staged_results,
    _write_checkpoint,
)


MODEL_LABEL = "mlx-community/Qwen3.5-2B-8bit@revision"


def _item(*, output: str = "Final answer: B", question: str = "Choose one"):
    return ExtractionItem(
        slug="model-a",
        track="do_you_see_me",
        question={
            "question_id": "q1",
            "question": question,
            "answer_type": "mcq_letter",
            "task": "shape",
        },
        record={"question_id": "q1", "output": output},
    )


def _contract(*, max_tokens: int = 64):
    return _checkpoint_contract(
        model_label=MODEL_LABEL,
        model_revision="revision",
        quantization="8-bit MLX",
        runtime="mlx-lm 0.31.3 / MLX 0.32.0",
        max_tokens=max_tokens,
    )


def _result(item: ExtractionItem):
    return {
        **item.record,
        "answer_extraction_method": LOCAL_ANSWER_EXTRACTION_METHOD,
        "extractor_model": MODEL_LABEL,
        "extractor_model_revision": "revision",
        "extractor_quantization": "8-bit MLX",
        "extractor_runtime": "mlx-lm 0.31.3 / MLX 0.32.0",
        "extractor_prompt_sha256": LOCAL_EXTRACTOR_PROMPT_SHA256,
        "extractor_ground_truth_access": False,
        "extractor_image_access": False,
        "extractor_source_diagnostics": (
            "model-a/do_you_see_me.diagnostics.jsonl"
        ),
        "extractor_source_output_sha256": hashlib.sha256(
            str(item.record.get("output") or "").encode("utf-8")
        ).hexdigest(),
        "extractor_output": "<answer>B</answer>",
        "extracted_answer": "B",
    }


def test_checkpoint_round_trip_rehydrates_current_source_record(tmp_path):
    item = _item()
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [item], {item.key: _result(item)}, _contract())

    rows = read_jsonl(path)
    assert len(rows) == 1
    assert "output" not in rows[0]["result"]

    loaded = _load_checkpoint(path, [item], _contract())
    assert loaded[item.key]["output"] == "Final answer: B"
    assert loaded[item.key]["extracted_answer"] == "B"


@pytest.mark.parametrize(
    ("item", "contract"),
    [
        (_item(output="Final answer: C"), _contract()),
        (_item(question="A changed question"), _contract()),
        (_item(), _contract(max_tokens=32)),
    ],
)
def test_checkpoint_rejects_stale_source_or_extractor_contract(
    tmp_path, item, contract
):
    source = _item()
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [source], {source.key: _result(source)}, _contract())

    assert _load_checkpoint(path, [item], contract) == {}


def test_checkpoint_does_not_persist_transport_failures(tmp_path):
    item = _item()
    path = tmp_path / "checkpoint.jsonl"
    transport_failure = {
        **item.record,
        "answer_extraction_method": LOCAL_ANSWER_EXTRACTION_METHOD,
        "extractor_error": "APIConnectionError: server unavailable",
    }

    _write_checkpoint(
        path, [item], {item.key: transport_failure}, _contract()
    )

    assert read_jsonl(path) == []


def test_checkpoint_rejects_duplicate_keys(tmp_path):
    item = _item()
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [item], {item.key: _result(item)}, _contract())
    row = path.read_text(encoding="utf-8")
    path.write_text(row + row, encoding="utf-8")

    with pytest.raises(FinalizationError, match="duplicate key"):
        _load_checkpoint(path, [item], _contract())


def test_extract_returns_a_complete_valid_checkpoint_without_server(tmp_path):
    item = _item()
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [item], {item.key: _result(item)}, _contract())

    loaded = asyncio.run(
        _extract(
            [item],
            endpoint="http://127.0.0.1:1/v1",
            api_model="unused",
            model_label=MODEL_LABEL,
            model_revision="revision",
            quantization="8-bit MLX",
            runtime="mlx-lm 0.31.3 / MLX 0.32.0",
            concurrency=1,
            max_tokens=64,
            request_timeout=1,
            label="test",
            checkpoint_path=path,
        )
    )

    assert loaded[item.key]["extracted_answer"] == "B"


def test_checkpoint_reassesses_cached_extractor_output(tmp_path):
    item = ExtractionItem(
        slug="model-a",
        track="do_you_see_me",
        question={
            "question_id": "q1",
            "question": "Read the letters",
            "answer_type": "text",
            "task": "letter_disambiguation",
        },
        record={"question_id": "q1", "output": "B, A, C"},
    )
    result = _result(item)
    result.pop("extracted_answer")
    result["extractor_output"] = "<answer>B A C</answer>"
    result["extractor_error"] = (
        "The extractor did not return a parseable answer block."
    )
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [item], {item.key: result}, _contract())

    loaded = _load_checkpoint(path, [item], _contract())

    assert loaded[item.key]["extracted_answer"] == "BAC"
    assert "extractor_error" not in loaded[item.key]


def test_checkpoint_file_is_jsonl_not_raw_response_archive(tmp_path):
    item = _item(output="x" * 10_000)
    path = tmp_path / "checkpoint.jsonl"
    _write_checkpoint(path, [item], {item.key: _result(item)}, _contract())

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["source_output_sha256"]
    assert "x" * 100 not in path.read_text(encoding="utf-8")


def test_staged_verifier_checks_every_changed_row_and_source_hash(tmp_path):
    final_root = tmp_path / "final"
    staging_root = tmp_path / "staging"
    item = _item()
    source_submission = {
        "question_id": "q1",
        "condition": "standard",
        "answer": INVALID_FORMAT_ANSWER,
    }
    source_diagnostic = {
        **item.record,
        "submission_status": "invalid_format",
        "format_failure_reason": "test",
        "raw_output_sha256": "source-metadata",
    }
    item = ExtractionItem(
        slug=item.slug,
        track=item.track,
        question=item.question,
        record=source_diagnostic,
    )
    source_dir = final_root / item.slug
    source_dir.mkdir(parents=True)
    submission_path = source_dir / f"{item.track}_submission.jsonl"
    diagnostics_path = source_dir / f"{item.track}.diagnostics.jsonl"
    submission_path.write_text(
        json.dumps(source_submission) + "\n", encoding="utf-8"
    )
    diagnostics_path.write_text(
        json.dumps(source_diagnostic) + "\n", encoding="utf-8"
    )
    bundles = {
        item.slug: {
            item.track: {
                "submissions": [source_submission],
                "diagnostics": [source_diagnostic],
                "submission_sha256": sha256(submission_path),
                "diagnostics_sha256": sha256(diagnostics_path),
            }
        }
    }
    recovered = {item.key: _result(item)}
    _stage_results(staging_root, bundles, [item], recovered)

    verification = _verify_staged_results(
        final_root=final_root,
        staging_root=staging_root,
        bundles=bundles,
        unresolved=[item],
        recovered=recovered,
        contract=_contract(),
    )

    assert verification == {
        "status": "passed",
        "verified_tracks": 1,
        "verified_candidates": 1,
        "verified_recoveries": 1,
        "verified_invalid": 0,
        "canonical_sources_unchanged": True,
    }

    staged_submission_path = (
        staging_root / item.slug / f"{item.track}_submission.jsonl"
    )
    staged_submission_path.write_text(
        json.dumps({**source_submission, "answer": "C"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FinalizationError, match="Staged submission mismatch"):
        _verify_staged_results(
            final_root=final_root,
            staging_root=staging_root,
            bundles=bundles,
            unresolved=[item],
            recovered=recovered,
            contract=_contract(),
        )


def test_staging_demotes_an_unresolved_unsupported_legacy_extraction(tmp_path):
    final_root = tmp_path / "final"
    staging_root = tmp_path / "staging"
    base = _item(output="Option A differs from option B without a conclusion.")
    source_diagnostic = {
        **base.record,
        "answer_extraction_method": "same-served-model-text-only-v1",
        "extractor_model": "evaluated/model",
        "extractor_output": "<answer>B</answer>",
        "extracted_answer": "B",
    }
    item = ExtractionItem(
        slug=base.slug,
        track=base.track,
        question=base.question,
        record=source_diagnostic,
        source_status="unsupported_legacy_extraction",
    )
    source_submission = {
        "question_id": "q1",
        "condition": "standard",
        "answer": "B",
    }
    source_dir = final_root / item.slug
    source_dir.mkdir(parents=True)
    submission_path = source_dir / f"{item.track}_submission.jsonl"
    diagnostics_path = source_dir / f"{item.track}.diagnostics.jsonl"
    submission_path.write_text(
        json.dumps(source_submission) + "\n", encoding="utf-8"
    )
    diagnostics_path.write_text(
        json.dumps(source_diagnostic) + "\n", encoding="utf-8"
    )
    bundles = {
        item.slug: {
            item.track: {
                "submissions": [source_submission],
                "diagnostics": [source_diagnostic],
                "submission_sha256": sha256(submission_path),
                "diagnostics_sha256": sha256(diagnostics_path),
            }
        }
    }
    unresolved_result = _result(item)
    unresolved_result.pop("extracted_answer")
    unresolved_result["extractor_output"] = "<answer>UNRESOLVED</answer>"
    unresolved_result["extractor_error"] = "The extractor returned UNRESOLVED."
    recovered = {item.key: unresolved_result}

    _stage_results(staging_root, bundles, [item], recovered)
    verification = _verify_staged_results(
        final_root=final_root,
        staging_root=staging_root,
        bundles=bundles,
        unresolved=[item],
        recovered=recovered,
        contract=_contract(),
    )

    assert verification["verified_invalid"] == 1
    assert read_jsonl(
        staging_root / item.slug / f"{item.track}_submission.jsonl"
    )[0]["answer"] == INVALID_FORMAT_ANSWER
    staged_diagnostic = read_jsonl(
        staging_root / item.slug / f"{item.track}.diagnostics.jsonl"
    )[0]
    assert staged_diagnostic["submission_status"] == "invalid_format"
    assert staged_diagnostic["format_failure_reason"] == (
        "unsupported_legacy_extraction_and_local_extractor_unresolved"
    )
