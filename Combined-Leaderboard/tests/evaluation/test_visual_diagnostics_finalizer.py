import json

import pytest

from evaluation.common.finalize_visual_diagnostics import (
    INVALID_MODEL_RESPONSE,
    finalize_diagnostics,
)
from evaluation.common.visual_pipeline import EvaluationPipelineError, final_answer
from utils.answer_extractor import AnswerComparator


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_final_answer_rejects_accidental_choice_letters_and_unclosed_reasoning():
    assert final_answer("reasoning without a final choice", "mcq_letter") == ""
    assert final_answer("<think>There are 7 candidates", "integer") == ""


def test_finalizer_preserves_answers_and_marks_unparseable_outputs_incorrect(tmp_path):
    questions = tmp_path / "questions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    output = tmp_path / "submission.jsonl"
    _write_jsonl(
        questions,
        [
            {"question_id": "q1", "answer_type": "integer"},
            {"question_id": "q2", "answer_type": "integer"},
            {"question_id": "q3", "answer_type": "mcq_letter"},
        ],
    )
    _write_jsonl(
        diagnostics,
        [
            {"question_id": "q1", "output": "<answer>12</answer>"},
            {"question_id": "q2", "output": "Many"},
            {"question_id": "q3", "output": "reasoning without a final choice"},
        ],
    )

    report = finalize_diagnostics(diagnostics, questions, output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["unparseable_count"] == 2
    assert [row["answer"] for row in rows] == [
        "12",
        INVALID_MODEL_RESPONSE,
        INVALID_MODEL_RESPONSE,
    ]
    assert all(row["condition"] == "standard" for row in rows)
    assert AnswerComparator.compare_answers("12", INVALID_MODEL_RESPONSE)[0] is False
    assert AnswerComparator.compare_answers("A", INVALID_MODEL_RESPONSE)[0] is False


def test_finalizer_refuses_inference_errors_and_incomplete_coverage(tmp_path):
    questions = tmp_path / "questions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    output = tmp_path / "submission.jsonl"
    _write_jsonl(
        questions,
        [
            {"question_id": "q1", "answer_type": "integer"},
            {"question_id": "q2", "answer_type": "integer"},
        ],
    )
    _write_jsonl(
        diagnostics,
        [{"question_id": "q1", "output": None, "error": "request timed out"}],
    )

    with pytest.raises(EvaluationPipelineError, match="coverage is invalid"):
        finalize_diagnostics(diagnostics, questions, output)

    _write_jsonl(
        diagnostics,
        [
            {"question_id": "q1", "output": None, "error": "request timed out"},
            {"question_id": "q2", "output": "2"},
        ],
    )
    with pytest.raises(EvaluationPipelineError, match="inference errors"):
        finalize_diagnostics(diagnostics, questions, output)
    assert not output.exists()
