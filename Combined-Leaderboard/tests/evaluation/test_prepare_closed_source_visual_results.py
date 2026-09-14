import json
from pathlib import Path

import pytest

from evaluation.closed_source_catalog import CLOSED_SOURCE_MODEL_CATALOG
from evaluation.prepare_closed_source_visual_results import (
    ClosedSourcePreparationError,
    PENDING_ANSWER,
    prepare_closed_source_results,
)
from evaluation.build_production_visual_results import execution_profile_for_variant


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    questions = {
        "do_you_see_me": {
            "question_id": "t1_2d_shape_discrimination_easy_0000",
            "task": "shape_discrimination",
            "answer_type": "integer",
            "question": "Count.",
        },
        "minds_eye": {
            "question_id": "t2_analogies_0000",
            "task": "analogies",
            "answer_type": "mcq_letter",
            "question": "Choose.",
        },
    }
    for track, row in questions.items():
        _write_jsonl(project / "tasks" / track / "questions.jsonl", [row])

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = []
    for source_model in CLOSED_SOURCE_MODEL_CATALOG:
        filename = f"results_{source_model}__main_noncot.jsonl"
        manifest.append({"model": source_model, "file": filename, "questions": 2})
        _write_jsonl(
            bundle / filename,
            [
                {
                    "question_id": questions["do_you_see_me"]["question_id"],
                    "subset": "dysm_2d_v1",
                    "task": "shape_discrimination",
                    "answer_type": "integer",
                    "output": "There are 2.",
                },
                {
                    "question_id": questions["minds_eye"]["question_id"],
                    "subset": "minds_eye_fresh_v1",
                    "task": "analogies",
                    "answer_type": "mcq_letter",
                    "output": "The answer is B.",
                },
            ],
        )
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return project, bundle


def test_prepare_closed_source_results_preserves_outputs_and_contract(tmp_path):
    project, bundle = _fixture(tmp_path)
    output = tmp_path / "source"

    result = prepare_closed_source_results(project, bundle, output)

    assert result["model_count"] == len(CLOSED_SOURCE_MODEL_CATALOG)
    assert result["response_count"] == 2 * len(CLOSED_SOURCE_MODEL_CATALOG)
    variant = result["variants"][0]
    model_dir = output / variant["variant_id"]
    diagnostic = json.loads(
        (model_dir / "do_you_see_me.diagnostics.jsonl").read_text().splitlines()[0]
    )
    submission = json.loads(
        (model_dir / "do_you_see_me_submission.jsonl").read_text().splitlines()[0]
    )
    assert diagnostic["output"] == "There are 2."
    assert diagnostic["provider_output_retained_verbatim"] is True
    assert submission["answer"] == PENDING_ANSWER
    assert variant["access"] == "closed"
    assert variant["weight_loading"] == "provider_managed"
    assert execution_profile_for_variant(output, variant) == {
        "weight_loading": "provider_managed",
        "compute_dtype": "provider_managed",
        "access": "closed",
        "selection_precision": "provider-api-retained-output",
        "final_pipeline_revision": "provider-api-gold-blind-evidence-extraction-v1",
    }


def test_prepare_closed_source_results_rejects_task_drift(tmp_path):
    project, bundle = _fixture(tmp_path)
    first = next(bundle.glob("results_*.jsonl"))
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    rows[0]["task"] = "wrong"
    _write_jsonl(first, rows)

    with pytest.raises(ClosedSourcePreparationError, match="task mismatch"):
        prepare_closed_source_results(project, bundle, tmp_path / "source")


def test_prepare_closed_source_results_can_select_one_trusted_model(tmp_path):
    project, bundle = _fixture(tmp_path)
    source_model = "gemini-3.5-flash-lite"
    output = tmp_path / "source"

    result = prepare_closed_source_results(
        project,
        bundle,
        output,
        source_models={source_model},
    )

    assert result["model_count"] == 1
    assert result["response_count"] == 2
    variant = result["variants"][0]
    assert variant["source_model"] == source_model
    run_config = json.loads(
        (
            output
            / variant["variant_id"]
            / "do_you_see_me.run_config.json"
        ).read_text()
    )
    assert run_config["generation"]["do_you_see_me"] == {
        "prompt_mode": "noncot",
        "source": "retained_provider_api_output",
        "source_declared_temperature": 0,
        "temperature": "provider_default",
        "temperature_behavior": "provider_deprecated_ignored",
        "thinking_level": "provider_default_not_verified_from_request_log",
    }
    assert run_config["provider_version_pinning"] == (
        "stable_alias_without_immutable_backend_revision"
    )


def test_prepare_closed_source_results_rejects_unknown_selection(tmp_path):
    project, bundle = _fixture(tmp_path)

    with pytest.raises(ClosedSourcePreparationError, match="Unknown trusted models"):
        prepare_closed_source_results(
            project,
            bundle,
            tmp_path / "source",
            source_models={"not-in-the-catalog"},
        )
