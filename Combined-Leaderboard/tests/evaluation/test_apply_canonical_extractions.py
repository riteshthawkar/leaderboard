import hashlib

import pytest

from evaluation.apply_canonical_extractions import (
    ApplyExtractionError,
    apply_result,
)
from evaluation.extract_canonical_answers import METHOD


GROUND_TRUTH_SHA256 = "a" * 64


def test_apply_result_updates_answer_with_evidence_and_preserves_prior_attempt():
    output = "I considered B, but the final answer is C."
    diagnostic = {
        "question_id": "q1",
        "output": output,
        "answer_extraction_method": "old-extractor",
        "extractor_output": "UNRESOLVED",
        "extractor_error": "ambiguous",
    }
    submission = {"question_id": "q1", "condition": "standard", "answer": "B"}
    audit = {
        "method": METHOD,
        "answer_type": "mcq_letter",
        "status": "other_committed",
        "verdict": "OTHER_COMMITTED",
        "answer": "C",
        "evidence": "the final answer is C",
        "extractor_model": "Qwen/Qwen3.6-27B",
        "extractor_output": '{"answer":"C"}',
        "finish_reason": "stop",
        "completion_tokens": 12,
        "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "ground_truth_sha256": GROUND_TRUTH_SHA256,
        "ground_truth_supplied": True,
    }

    assert (
        apply_result(
            diagnostic,
            submission,
            audit,
            gold_answer="B",
            ground_truth_sha256=GROUND_TRUTH_SHA256,
        )
        == "other_committed"
    )
    assert submission["answer"] == "C"
    assert diagnostic["extracted_answer"] == "C"
    assert diagnostic["extractor_evidence"] == "the final answer is C"
    assert diagnostic["extractor_commitment_verdict"] == "OTHER_COMMITTED"
    assert diagnostic["ground_truth_supplied"] is True
    assert diagnostic["extractor_attempts"][-1]["extractor_error"] == "ambiguous"


def test_apply_result_revalidates_evidence_and_hash():
    output = "A and B are discussed without a final choice."
    diagnostic = {"question_id": "q1", "output": output}
    submission = {"question_id": "q1", "condition": "standard", "answer": output}
    audit = {
        "method": METHOD,
        "answer_type": "mcq_letter",
        "status": "gold_committed",
        "verdict": "GOLD_COMMITTED",
        "answer": "B",
        "evidence": "final choice B",
        "extractor_model": "Qwen/Qwen3.6-27B",
        "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "ground_truth_sha256": GROUND_TRUTH_SHA256,
        "ground_truth_supplied": True,
    }
    assert (
        apply_result(
            diagnostic,
            submission,
            audit,
            gold_answer="B",
            ground_truth_sha256=GROUND_TRUTH_SHA256,
        )
        is None
    )

    audit["response_sha256"] = "wrong"
    with pytest.raises(ApplyExtractionError, match="Response hash changed"):
        apply_result(
            diagnostic,
            submission,
            audit,
            gold_answer="B",
            ground_truth_sha256=GROUND_TRUTH_SHA256,
        )


def test_apply_result_rejects_verdict_leakage():
    output = "The final answer is C."
    diagnostic = {"question_id": "q1", "output": output}
    submission = {"question_id": "q1", "condition": "standard", "answer": "C"}
    audit = {
        "method": METHOD,
        "answer_type": "mcq_letter",
        "status": "gold_committed",
        "verdict": "GOLD_COMMITTED",
        "answer": "C",
        "evidence": "The final answer is C",
        "extractor_model": "Qwen/Qwen3.6-27B",
        "response_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "ground_truth_sha256": GROUND_TRUTH_SHA256,
        "ground_truth_supplied": True,
    }
    with pytest.raises(ApplyExtractionError, match="verdict disagrees"):
        apply_result(
            diagnostic,
            submission,
            audit,
            gold_answer="B",
            ground_truth_sha256=GROUND_TRUTH_SHA256,
        )