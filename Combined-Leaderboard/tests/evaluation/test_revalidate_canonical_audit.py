import hashlib

import pytest

from evaluation.extract_canonical_answers import METHOD
from evaluation.revalidate_canonical_audit import RevalidationError, revalidate_row


def test_revalidate_row_recomputes_verdict_without_an_llm_request():
    response = "The final answer is B."
    candidate = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "category": "heuristic_local_parse",
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "gold_answer": "B",
        "current_submission_answer": "A",
    }
    source = {
        key: candidate[key]
        for key in (
            "model_slug",
            "track",
            "question_id",
            "answer_type",
            "category",
            "response_sha256",
        )
    } | {
        "method": "older-method",
        "extractor_contract_sha256": "c" * 64,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "extractor_model": "Qwen/Qwen3.6-27B",
        "extractor_output": (
            '{"verdict":"COMMITTED","answer":"B",'
            '"evidence":"The final answer is B"}'
        ),
        "finish_reason": "stop",
        "completion_tokens": 20,
    }

    result = revalidate_row(candidate, source, "g" * 64, "a" * 64)

    assert result["method"] == METHOD
    assert result["status"] == "gold_committed"
    assert result["verdict"] == "GOLD_COMMITTED"
    assert result["extractor_verdict"] == "COMMITTED"
    assert result["extractor_contract_sha256"] == "c" * 64
    assert result["ground_truth_supplied_to_extractor"] is False
    assert result["ground_truth_loaded_by_extractor_process"] is False
    assert result["revalidated_without_llm_request"] is True
    assert result["revalidated_from_method"] == "older-method"
    assert result["revalidated_from_audit_sha256"] == "a" * 64


def test_revalidate_row_rejects_changed_response():
    candidate = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "category": "unresolved_raw",
        "response": "B",
        "response_sha256": "current",
        "gold_answer": "B",
        "current_submission_answer": "",
    }
    source = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "response_sha256": "old",
        "extractor_contract_sha256": "c" * 64,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
    }

    with pytest.raises(RevalidationError, match="Response hash changed"):
        revalidate_row(candidate, source, "g" * 64, "a" * 64)


def test_revalidate_row_rejects_gold_exposed_source():
    response = "The final answer is B."
    candidate = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "category": "heuristic_local_parse",
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "gold_answer": "B",
        "current_submission_answer": "A",
    }
    source = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "response_sha256": candidate["response_sha256"],
        "extractor_contract_sha256": "c" * 64,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": True,
    }

    with pytest.raises(RevalidationError, match="not gold-blind"):
        revalidate_row(candidate, source, "g" * 64, "a" * 64)


def test_revalidate_row_rejects_source_process_that_loaded_gold():
    response = "The final answer is B."
    candidate = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "category": "heuristic_local_parse",
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "gold_answer": "B",
        "current_submission_answer": "A",
    }
    source = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "response_sha256": candidate["response_sha256"],
        "extractor_contract_sha256": "c" * 64,
        "ground_truth_loaded": True,
        "ground_truth_supplied_to_extractor": False,
    }

    with pytest.raises(RevalidationError, match="process loaded ground truth"):
        revalidate_row(candidate, source, "g" * 64, "a" * 64)
