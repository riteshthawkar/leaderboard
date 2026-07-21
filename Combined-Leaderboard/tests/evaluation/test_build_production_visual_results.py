import hashlib
import json
from pathlib import Path

import pytest

from evaluation.build_production_visual_results import (
    ProductionBuildError,
    build_production_results,
    load_completed_audit,
)
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    classify_extractor_output,
)
from evaluation.finalize_visual_results import read_json, sha256, verify_canonical_results


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audit_row(candidate: dict, extractor_payload: dict, contract: str) -> dict:
    extractor_output = json.dumps(extractor_payload, separators=(",", ":"))
    classified = classify_extractor_output(candidate, extractor_output)
    return {
        **candidate,
        **classified,
        "method": METHOD,
        "extractor_model": DEFAULT_EXTRACTOR_MODEL,
        "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
        "extractor_contract_sha256": contract,
        "extractor_output": extractor_output,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "finish_reason": "stop",
        "completion_tokens": 12,
    }


def test_builds_verified_v12_tree_from_complete_evidence_audit(tmp_path):
    project_root = tmp_path / "project"
    source_root = project_root / "evaluation" / "results" / "final-extracted-v11"
    output_root = project_root / "evaluation" / "results" / "final-extracted-v12"
    audit_path = tmp_path / "evidence-audit.jsonl"
    slug = "model-a"
    revision = "a" * 40
    contract = "c" * 64
    track_cases = {
        "do_you_see_me": {
            "question_id": "t1_2d_letter_disambiguation_easy_0000",
            "answer_type": "text",
            "task": "letter_disambiguation",
            "response": "The letters visible are E T O N.",
            "source_answer": "E T O N",
            "extractor": {
                "verdict": "COMMITTED",
                "answer": "E T O N",
                "evidence": "The letters visible are E T O N",
            },
        },
        "minds_eye": {
            "question_id": "t2_analogies_0000",
            "answer_type": "mcq_letter",
            "task": "analogies",
            "response": "The final answer is G.",
            "source_answer": "G",
            "extractor": {
                "verdict": "COMMITTED",
                "answer": "G",
                "evidence": "The final answer is G",
            },
        },
    }

    variant_tracks = {}
    audit_rows = []
    source_dir = source_root / slug
    for track, case in track_cases.items():
        question = {
            "question_id": case["question_id"],
            "question": "Choose the final answer.",
            "answer_type": case["answer_type"],
            "task": case["task"],
        }
        _write_jsonl(project_root / "tasks" / track / "questions.jsonl", [question])
        diagnostics_path = source_dir / f"{track}.diagnostics.jsonl"
        submission_path = source_dir / f"{track}_submission.jsonl"
        run_config_path = source_dir / f"{track}.run_config.json"
        diagnostic = {
            "question_id": case["question_id"],
            "answer_type": case["answer_type"],
            "output": case["response"],
            "finish_reason": "stop",
            "completion_tokens": 20,
            "extracted_answer": case["source_answer"],
        }
        _write_jsonl(diagnostics_path, [diagnostic])
        _write_jsonl(
            submission_path,
            [
                {
                    "question_id": case["question_id"],
                    "condition": "standard",
                    "answer": case["source_answer"],
                }
            ],
        )
        _write_json(
            run_config_path,
            {
                "schema_version": 11,
                "model_id": "org/model-a",
                "model_revision": revision,
                "weight_loading": "unquantized",
                "compute_dtype": "bfloat16",
                "pipeline_revision": "unquantized-bf16-mandatory-extraction-v11",
                "generation": {track: {"temperature": 0.0}},
                "serving_engine": {"name": "vllm", "version": "test"},
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "request_concurrency": 1,
                "max_model_len": 32768,
            },
        )
        variant_tracks[track] = {
            "relative_dir": slug,
            "diagnostics": diagnostics_path.name,
            "submission": submission_path.name,
            "diagnostics_sha256": sha256(diagnostics_path),
            "submission_sha256": sha256(submission_path),
            "source_run_config_sha256": sha256(run_config_path),
        }
        response_hash = hashlib.sha256(case["response"].encode("utf-8")).hexdigest()
        candidate = {
            "model_slug": slug,
            "source_relative_dir": slug,
            "track": track,
            "question_id": case["question_id"],
            "answer_type": case["answer_type"],
            "task": case["task"],
            "category": "prior_model_extractor",
            "question": question["question"],
            "response": case["response"],
            "response_finish_reason": "stop",
            "response_completion_tokens": 20,
            "response_sha256": response_hash,
            "current_submission_answer": case["source_answer"],
        }
        audit_rows.append(_audit_row(candidate, case["extractor"], contract))

    _write_json(
        source_root / "index.json",
        {
            "schema_version": 1,
            "variant_count": 1,
            "variants": [
                {
                    "variant_id": slug,
                    "model_id": "org/model-a",
                    "model_revision": revision,
                    "mode": "nonthinking",
                    "tracks": variant_tracks,
                }
            ],
        },
    )
    _write_jsonl(audit_path, audit_rows)

    index = build_production_results(
        project_root,
        source_root,
        audit_path,
        output_root,
        excluded_variants=set(),
    )

    assert index["model_count"] == 1
    assert set(index) >= {"models", "model_count", "evidence_extraction"}
    assert verify_canonical_results(output_root, project_root) == {
        "model_count": 1,
        "verified_models": ["org/model-a"],
    }
    dys_submission = json.loads(
        (output_root / slug / "do_you_see_me_submission.jsonl").read_text()
    )
    minds_eye_submission = json.loads(
        (output_root / slug / "minds_eye_submission.jsonl").read_text()
    )
    assert dys_submission["answer"] == "ETON"
    assert minds_eye_submission["answer"] == "__INVALID_FORMAT__"

    manifest = read_json(output_root / slug / "final_manifest.json")
    assert manifest["tracks"]["do_you_see_me"]["strict_answer_count"] == 1
    assert manifest["tracks"]["do_you_see_me"]["invalid_format_count"] == 0
    assert manifest["tracks"]["minds_eye"]["strict_answer_count"] == 0
    assert manifest["tracks"]["minds_eye"]["invalid_commitment_count"] == 1
    assert manifest["tracks"]["minds_eye"]["invalid_format_count"] == 1
    assert manifest["evidence_extraction"]["extractor_model"] == DEFAULT_EXTRACTOR_MODEL
    assert manifest["evidence_extraction"]["ground_truth_loaded"] is False


def test_completed_audit_rejects_spoofed_terminal_fallback_method(tmp_path):
    response = "Reasoning. \\boxed{4}"
    response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    candidate = {
        "model_slug": "model-a",
        "track": "do_you_see_me",
        "question_id": "q1",
        "answer_type": "mcq_index_1_4",
        "task": "visual_closure",
        "response": response,
        "response_sha256": response_hash,
    }
    audit_path = tmp_path / "audit.jsonl"
    _write_jsonl(
        audit_path,
        [
            {
                **candidate,
                "method": METHOD,
                "extractor_model": DEFAULT_EXTRACTOR_MODEL,
                "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
                "extractor_contract_sha256": "c" * 64,
                "ground_truth_loaded": False,
                "ground_truth_supplied_to_extractor": False,
                "extractor_output": "{",
                "extractor_verdict": "COMMITTED",
                "answer": "4",
                "evidence": "\\boxed{4}",
                "status": "committed",
                "extractor_attempts": [{"status": "invalid_extractor_output"}],
                "terminal_fallback_method": "untrusted-fallback",
                "terminal_fallback_from_status": "invalid_extractor_output",
            }
        ],
    )

    with pytest.raises(ProductionBuildError, match="fallback method mismatch"):
        load_completed_audit(audit_path, [candidate])
