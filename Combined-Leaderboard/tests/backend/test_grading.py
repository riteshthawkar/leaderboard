"""
Grading / scoring tests for the Combined Leaderboard.

Covers the pieces that are easy to get wrong and expensive to debug in
production:

  * the pure grading helpers (``normalize`` / ``deterministic_match`` /
    ``is_cannot_determine``);
  * ``LLMGrader.grade`` with the OpenAI call **mocked** – the LLM extract and
    LLM judge paths, plus graceful fallback to deterministic matching when no
    key is configured or the network call fails;
  * malformed submission uploads (bad JSONL, retired extensions, duplicate
    sample/question ids, incomplete coverage);
  * the spatial diagnostics math (CoT delta, shortcut score, hallucination
    resistance).

These tests are fully offline and hermetic: no API key and no dataset files
are required. Run them with ``python -m pytest tests/backend/test_grading.py``.
"""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from scoring.llm_grader import (  # noqa: E402
    LLMGrader,
    normalize,
    deterministic_match,
    is_cannot_determine,
    _format_options,
)
import scoring.task_scorer as task_scorer_module  # noqa: E402
from scoring.task_scorer import (  # noqa: E402
    SubmissionValidationError,
    TaskScorer,
    _strict_visual_match,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """Minimal stand-in for a ``requests`` Response from the chat endpoint."""

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _enabled_grader(grading):
    """Build an LLMGrader and force it into the 'reachable OpenAI' state so the
    mocked network call is exercised regardless of the local environment."""
    grader = LLMGrader(grading)
    grader.api_key = "test-key"
    grader._backend = "openai"
    grader._disabled = False
    return grader


def _expect_error(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"Expected {exc.__name__} was not raised")


def _write_temp(content, suffix):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False,
                                    encoding="utf-8", newline="")
    f.write(content)
    f.close()
    return Path(f.name)


# --------------------------------------------------------------------------- #
# 1. Pure grading helpers
# --------------------------------------------------------------------------- #
def test_normalize():
    print("Testing normalize()...")
    assert normalize("The answer is: (C).") == "c"
    assert normalize("  Option B  ") == "b"
    assert normalize(None) == ""
    assert normalize("Answer - 42") == "42"
    print("\u2713 normalize tests passed")


def test_deterministic_match():
    print("Testing deterministic_match()...")
    # exact / MCQ letter equivalence
    assert deterministic_match("(C)", "C") is True
    assert deterministic_match("C", "c") is True
    # numeric equality on free-response answers
    assert deterministic_match("The answer is 42 units", "42") is True
    assert deterministic_match("3.0", "3") is True
    # clear mismatches
    assert deterministic_match("B", "A") is False
    assert deterministic_match("", "A") is False
    print("\u2713 deterministic_match tests passed")


def test_is_cannot_determine():
    print("Testing is_cannot_determine()...")
    assert is_cannot_determine("Cannot determine") is True
    assert is_cannot_determine("unknown") is True
    assert is_cannot_determine("Cannot be determined") is True
    assert is_cannot_determine("A") is False
    print("\u2713 is_cannot_determine tests passed")


@pytest.mark.parametrize(
    ("prediction", "gold", "answer_type", "task", "valid_format", "correct"),
    [
        ("3", "3", "integer", "shape_discrimination", True, True),
        ("The answer is 3", "3", "integer", "shape_discrimination", False, False),
        ("2", "2", "mcq_index_1_4", "visual_closure", True, True),
        ("(2)", "2", "mcq_index_1_4", "visual_closure", False, False),
        ("a", "A", "mcq_letter", "analogies", True, True),
        ("Option A", "A", "mcq_letter", "analogies", False, False),
        ("yes", "Yes", "text", "form_constancy", True, True),
        ("Yes.", "Yes", "text", "form_constancy", False, False),
        ("PARK", "PARK", "text", "letter_disambiguation", True, True),
        ("p a r k", "A", "text", "letter_disambiguation", False, False),
        ("ABCDEFGHIJ", "A", "text", "letter_disambiguation", False, False),
    ],
)
def test_visual_answers_require_the_declared_final_answer_format(
    prediction,
    gold,
    answer_type,
    task,
    valid_format,
    correct,
):
    assert _strict_visual_match(prediction, gold, answer_type, task) == (
        valid_format,
        correct,
    )


def test_format_options():
    print("Testing _format_options()...")
    assert _format_options(["cat", "dog"]) == "A. cat\nB. dog"
    assert _format_options({"A": "cat", "B": "dog"}) == "A. cat\nB. dog"
    assert _format_options(None) == ""
    print("\u2713 _format_options tests passed")


# --------------------------------------------------------------------------- #
# 2. LLMGrader with the OpenAI call mocked
# --------------------------------------------------------------------------- #
def test_llm_extract_path():
    print("Testing LLM extract path (mocked OpenAI)...")
    grader = _enabled_grader({"method": "extract", "answer_types": ["mcq"]})

    # The extractor returns "C"; ground truth is "C" -> correct via llm_extract.
    with patch("scoring.llm_grader.requests.post",
               return_value=_FakeResponse("C")):
        ok, method = grader.grade("I believe the answer is C.", "C",
                                  answer_type="mcq")
    assert method == "llm_extract", method
    assert ok is True

    # The extractor returns the wrong letter -> incorrect, still llm_extract.
    with patch("scoring.llm_grader.requests.post",
               return_value=_FakeResponse("B")):
        ok, method = grader.grade("... final answer B ...", "C",
                                  answer_type="mcq")
    assert method == "llm_extract", method
    assert ok is False
    print("\u2713 LLM extract path tests passed")


def test_llm_judge_path():
    print("Testing LLM judge path (mocked OpenAI)...")
    grader = _enabled_grader({"method": "judge", "answer_types": ["mcq"]})

    with patch("scoring.llm_grader.requests.post",
               return_value=_FakeResponse("correct")):
        ok, method = grader.grade("The object is on the left, so B.", "B",
                                  question="Where is it?", options=["left", "right"])
    assert method == "llm_judge", method
    assert ok is True

    with patch("scoring.llm_grader.requests.post",
               return_value=_FakeResponse("incorrect")):
        ok, method = grader.grade("It's on the right, so A.", "B",
                                  question="Where is it?", options=["left", "right"])
    assert method == "llm_judge", method
    assert ok is False
    print("\u2713 LLM judge path tests passed")


def test_deterministic_fallback_no_key():
    print("Testing deterministic fallback (no key)...")
    grader = LLMGrader({"method": "extract", "answer_types": ["mcq"]})
    grader._backend = ""  # simulate no OPENAI_API_KEY
    assert grader.enabled is False
    ok, method = grader.grade("A", "A")
    assert method == "deterministic", method
    assert ok is True
    ok, method = grader.grade("A", "B")
    assert method == "deterministic" and ok is False
    print("\u2713 deterministic fallback (no key) tests passed")


def test_fallback_on_network_failure():
    print("Testing deterministic fallback on network failure...")
    grader = _enabled_grader({"method": "judge", "answer_types": ["mcq"]})
    # The OpenAI call raises -> _chat disables the backend and returns None,
    # so grade() must fall back to deterministic matching rather than error out.
    with patch("scoring.llm_grader.requests.post",
               side_effect=RuntimeError("connection reset")):
        ok, method = grader.grade("A", "A", question="q", options=["x", "y"])
    assert method == "deterministic", method
    assert ok is True
    assert grader._disabled is True
    print("\u2713 network-failure fallback tests passed")


# --------------------------------------------------------------------------- #
# 3. Malformed uploads
# --------------------------------------------------------------------------- #
def test_jsonl_ground_truth_sources():
    print("Testing JSONL ground truth source loading...")
    gt_2d = _write_temp(
        '{"question_id":"t1_2d_shape_discrimination_easy_0000","answer":"1"}\n',
        ".jsonl",
    )
    gt_3d = _write_temp(
        '{"question_id":"t1_3d_form_constancy_hard_0001","answer":"Yes"}\n',
        ".jsonl",
    )
    try:
        scorer = TaskScorer("do_you_see_me")
        scorer.ground_truth_files = [gt_2d, gt_3d]
        scorer._gt = None
        gt = scorer.ground_truth
        assert len(gt) == 2
        assert gt["t1_2d_shape_discrimination_easy_0000"]["capability"] == "shape_discrimination"
        assert gt["t1_2d_shape_discrimination_easy_0000"]["dimension"] == "2D"
        assert gt["t1_2d_shape_discrimination_easy_0000"]["difficulty"] == "easy"
        assert gt["t1_3d_form_constancy_hard_0001"]["capability"] == "form_constancy"
        assert gt["t1_3d_form_constancy_hard_0001"]["dimension"] == "3D"
    finally:
        gt_2d.unlink(missing_ok=True)
        gt_3d.unlink(missing_ok=True)
    print("\u2713 JSONL ground truth source tests passed")


def test_hf_ground_truth_fallback():
    print("Testing HF ground truth fallback...")

    class _HFResponse:
        status_code = 200

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1024):
            yield self.content

    calls = []

    def fake_get(url, headers=None, stream=False, timeout=None):
        calls.append({
            "url": url,
            "headers": headers or {},
            "stream": stream,
            "timeout": timeout,
        })
        return _HFResponse(
            b'{"question_id":"t1_2d_shape_discrimination_easy_0000","answer":"1"}\n'
        )

    old_values = {
        "GROUND_TRUTHS_SOURCE": task_scorer_module.GROUND_TRUTHS_SOURCE,
        "GROUND_TRUTHS_HF_REPO": task_scorer_module.GROUND_TRUTHS_HF_REPO,
        "GROUND_TRUTHS_HF_REPO_TYPE": task_scorer_module.GROUND_TRUTHS_HF_REPO_TYPE,
        "GROUND_TRUTHS_HF_REVISION": task_scorer_module.GROUND_TRUTHS_HF_REVISION,
        "GROUND_TRUTHS_HF_CACHE_DIR": task_scorer_module.GROUND_TRUTHS_HF_CACHE_DIR,
        "GROUND_TRUTHS_HF_FORCE_REFRESH": task_scorer_module.GROUND_TRUTHS_HF_FORCE_REFRESH,
        "HF_TOKEN": task_scorer_module.HF_TOKEN,
        "GROUND_TRUTHS_DIR": task_scorer_module.GROUND_TRUTHS_DIR,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "missing_gt"
        cache = Path(tmp) / "cache"
        try:
            task_scorer_module.GROUND_TRUTHS_SOURCE = "hf"
            task_scorer_module.GROUND_TRUTHS_HF_REPO = "owner/private-gt"
            task_scorer_module.GROUND_TRUTHS_HF_REPO_TYPE = "dataset"
            task_scorer_module.GROUND_TRUTHS_HF_REVISION = "main"
            task_scorer_module.GROUND_TRUTHS_HF_CACHE_DIR = cache
            task_scorer_module.GROUND_TRUTHS_HF_FORCE_REFRESH = False
            task_scorer_module.HF_TOKEN = "test-token"
            task_scorer_module.GROUND_TRUTHS_DIR = root
            scorer = TaskScorer("do_you_see_me")
            scorer.ground_truth_files = [root / "dysm_2d_v1" / "ground_truth.jsonl"]
            scorer._gt = None
            with patch("scoring.task_scorer.requests.get", side_effect=fake_get):
                gt = scorer.ground_truth
            assert len(gt) == 1
            assert calls, "HF download was not attempted"
            assert calls[0]["headers"]["Authorization"] == "Bearer test-token"
            assert "test-token" not in calls[0]["url"]
            cached_file = cache / "dysm_2d_v1" / "ground_truth.jsonl"
            assert cached_file.exists()
            if os.name != "nt":
                assert cached_file.parent.stat().st_mode & 0o777 == 0o700
                assert cached_file.stat().st_mode & 0o777 == 0o600
        finally:
            for name, value in old_values.items():
                setattr(task_scorer_module, name, value)
    print("\u2713 HF ground truth fallback tests passed")


def test_malformed_uploads():
    print("Testing malformed uploads...")
    scorer = TaskScorer("do_you_see_me")

    # Invalid JSONL content.
    bad_json = _write_temp("{not valid json", ".jsonl")
    try:
        _expect_error(lambda: scorer.parse_submission(bad_json))
    finally:
        bad_json.unlink(missing_ok=True)

    # Retired / unsupported extensions.
    bad_ext = _write_temp("{}", ".json")
    try:
        _expect_error(lambda: scorer.parse_submission(bad_ext), ValueError)
    finally:
        bad_ext.unlink(missing_ok=True)

    # Empty JSONL.
    empty = _write_temp("", ".jsonl")
    try:
        _expect_error(lambda: scorer.parse_submission(empty), ValueError)
    finally:
        empty.unlink(missing_ok=True)

    # Sanity: well-formed JSONL parses into 'standard' predictions.
    good_jsonl = _write_temp(
        '{"question_id":"s1","answer":"A"}\n'
        '{"question_id":"s2","condition":"standard","prediction":"B"}\n',
        ".jsonl",
    )
    try:
        preds, _ = scorer.parse_submission(good_jsonl)
        assert preds["standard"] == {"s1": "A", "s2": "B"}
    finally:
        good_jsonl.unlink(missing_ok=True)
    print("\u2713 malformed upload tests passed")


def test_jsonl_submission_records():
    print("Testing JSONL submission record preservation...")
    scorer = TaskScorer("minds_eye")
    text = (
        '{"question_id":"q1","answer":"A"}\n'
        '\n'
        '{"question_id":"q2","condition":"standard","prediction":"B"}\n'
    )
    preds, meta, records = scorer.parse_submission_text_with_records(text)
    assert meta == {}
    assert preds["standard"] == {"q1": "A", "q2": "B"}
    assert records == [
        {
            "row_index": 1,
            "line_number": 1,
            "question_id": "q1",
            "condition": "standard",
            "answer": "A",
        },
        {
            "row_index": 2,
            "line_number": 3,
            "question_id": "q2",
            "condition": "standard",
            "answer": "B",
        },
    ]
    print("\u2713 JSONL submission record tests passed")


def test_jsonl_coverage_validation():
    print("Testing JSONL coverage validation...")
    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {
        "s1": {"format": "mcq_letter"},
        "s2": {"format": "mcq_letter"},
    }
    scorer._gt = {
        "s1": {"answer": "A", "group": "g"},
        "s2": {"answer": "B", "group": "g"},
    }

    full = _write_temp(
        '{"question_id":"s1","answer":"Option A"}\n'
        '{"question_id":"s2","answer":"B"}\n',
        ".jsonl",
    )
    try:
        score = scorer.score(full, "model")
        assert score.total_samples == 2
        assert score.correct_samples == 1
        assert score.grading["method_counts"] == {
            "invalid_format": 1,
            "jsonl_exact": 1,
        }
        assert score.metadata == {"submission_format": "jsonl"}
    finally:
        full.unlink(missing_ok=True)

    partial = _write_temp('{"question_id":"s1","answer":"A"}\n', ".jsonl")
    try:
        _expect_error(lambda: scorer.score(partial, "model"), ValueError)
    finally:
        partial.unlink(missing_ok=True)

    unknown = _write_temp(
        '{"question_id":"s1","answer":"A"}\n'
        '{"question_id":"s3","answer":"B"}\n',
        ".jsonl",
    )
    try:
        _expect_error(lambda: scorer.score(unknown, "model"), ValueError)
    finally:
        unknown.unlink(missing_ok=True)

    duplicate = _write_temp(
        '{"question_id":"s1","answer":"A"}\n'
        '{"question_id":"s1","answer":"A"}\n'
        '{"question_id":"s2","answer":"B"}\n',
        ".jsonl",
    )
    try:
        _expect_error(lambda: scorer.score(duplicate, "model"), ValueError)
    finally:
        duplicate.unlink(missing_ok=True)
    print("\u2713 JSONL coverage validation tests passed")


@pytest.mark.parametrize(
    ("text", "expected_code"),
    [
        ("", "empty_submission_file"),
        ("{not json", "invalid_jsonl_syntax"),
        ("[]", "jsonl_row_not_object"),
        ('{"answer":"A"}', "missing_question_id"),
        ('{"question_id":"s1","sample_id":"s2","answer":"A"}', "conflicting_question_ids"),
        ('{"question_id":"s1"}', "missing_answer_field"),
        ('{"question_id":"s1","answer":"A","prediction":"B"}', "multiple_answer_fields"),
        ('{"question_id":"s1","answer":{"choice":"A"}}', "invalid_answer_type"),
        ('{"question_id":"s1","answer":NaN}', "invalid_answer_value"),
        ('{"question_id":"s1","answer":"A","condition":"other"}', "invalid_submission_condition"),
        ('{"question_id":"s1","answer":"A","condition":"cot"}', "condition_not_supported_for_task"),
        (
            '{"question_id":"s1","answer":"A"}\n'
            '{"question_id":"s1","answer":"B"}',
            "duplicate_sample_output",
        ),
    ],
)
def test_submission_validation_codes_are_specific(text, expected_code):
    scorer = TaskScorer("do_you_see_me")

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.parse_submission_text_with_records(text)

    assert captured.value.code == expected_code
    assert str(captured.value)


def test_blank_answers_are_reported_together():
    scorer = TaskScorer("do_you_see_me")
    text = (
        '{"question_id":"s1","answer":""}\n'
        '{"question_id":"s2","answer":null}\n'
        '{"question_id":"s3","answer":"A"}\n'
    )

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.parse_submission_text_with_records(text)

    error = captured.value
    assert error.code == "empty_sample_outputs"
    assert error.details["count"] == 2
    assert [item["line_number"] for item in error.details["examples"]] == [1, 2]


def test_submission_parser_rejects_oversized_lines(monkeypatch):
    scorer = TaskScorer("do_you_see_me")
    monkeypatch.setattr(task_scorer_module, "MAX_SUBMISSION_LINE_CHARS", 20)

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.parse_submission_text_with_records(
            '{"question_id":"s1","answer":"A"}'
        )

    assert captured.value.code == "submission_line_too_long"
    assert captured.value.details["line_number"] == 1
    assert captured.value.details["max_characters"] == 20


def test_submission_scores_oversized_malformed_answer_as_incorrect():
    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {"s1": {"format": "mcq_letter"}}
    scorer._gt = {
        "s1": {"answer": "A", "group": "g", "task": "analogies"},
    }
    raw_output = "A" * 20_000

    score = scorer.score_submission_text(
        '{"question_id":"s1","answer":"' + raw_output + '"}\n',
        "model",
    )

    assert score.correct_samples == 0
    assert score.grading["method_counts"] == {"invalid_format": 1}


def test_submission_parser_rejects_more_rows_than_the_benchmark():
    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {}
    scorer._gt = {"s1": {"answer": "A", "group": "g"}}

    text = "\n".join(
        f'{{"question_id":"s{index}","answer":"A"}}'
        for index in range(1, 103)
    )

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.parse_submission_text_with_records(text)

    assert captured.value.code == "too_many_submission_rows"
    assert captured.value.details["max_rows"] == 1
    assert captured.value.details["rows_seen"] == 102


def test_missing_spatial_conditions_have_specific_guidance():
    scorer = TaskScorer("spatial")

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.parse_submission_text_with_records(
            '{"question_id":"s1","answer":"A","condition":"main_cot"}\n'
        )

    assert captured.value.code == "missing_required_conditions"
    assert captured.value.details["conditions_present"] == ["main_cot"]
    assert "main_noncot" in captured.value.details["missing_conditions"]
    assert "no_image_plus_cot" in captured.value.details["missing_conditions"]


@pytest.mark.parametrize(
    ("text", "expected_code", "expected_details"),
    [
        (
            '{"question_id":"s1","answer":"A"}\n',
            "missing_sample_outputs",
            {"count": 1, "question_ids": ["s2"]},
        ),
        (
            '{"question_id":"s1","answer":"A"}\n'
            '{"question_id":"s2","answer":"B"}\n'
            '{"question_id":"s3","answer":"C"}\n',
            "unknown_sample_ids",
            {"count": 1, "question_ids": ["s3"]},
        ),
        (
            '{"question_id":"s1","answer":"A"}\n'
            '{"question_id":"s3","answer":"C"}\n',
            "sample_id_coverage_mismatch",
            {"missing_count": 1, "unknown_count": 1},
        ),
    ],
)
def test_submission_coverage_errors_include_counts_and_examples(text, expected_code, expected_details):
    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {}
    scorer._gt = {
        "s1": {"answer": "A", "group": "g"},
        "s2": {"answer": "B", "group": "g"},
    }
    predictions, _meta = scorer.parse_submission_text(text)

    with pytest.raises(SubmissionValidationError) as captured:
        scorer.score_predictions(predictions, "model")

    error = captured.value
    assert error.code == expected_code
    for key, value in expected_details.items():
        assert error.details[key] == value


def test_paper_aligned_analysis_aggregates():
    perception = TaskScorer("do_you_see_me")
    perception._questions = {}
    perception._gt = {
        "p1": {"answer": "A", "group": "shape", "capability": "shape", "dimension": "2D", "difficulty": "easy"},
        "p2": {"answer": "A", "group": "shape", "capability": "shape", "dimension": "2D", "difficulty": "medium"},
        "p3": {"answer": "A", "group": "shape", "capability": "shape", "dimension": "2D", "difficulty": "hard"},
        "p4": {"answer": "A", "group": "closure", "capability": "closure", "dimension": "3D", "difficulty": "easy"},
        "p5": {"answer": "A", "group": "closure", "capability": "closure", "dimension": "3D", "difficulty": "medium"},
        "p6": {"answer": "A", "group": "closure", "capability": "closure", "dimension": "3D", "difficulty": "hard"},
    }
    perception_score = perception.score_predictions(
        {
            "standard": {
                "p1": "A",
                "p2": "A",
                "p3": "B",
                "p4": "A",
                "p5": "B",
                "p6": "B",
            }
        },
        "perception-model",
    )
    assert perception_score.submitted_at.utcoffset().total_seconds() == 0
    assert perception_score.analysis["dimension"]["2D"].accuracy == pytest.approx(2 / 3)
    assert perception_score.analysis["dimension"]["3D"].accuracy == pytest.approx(1 / 3)
    assert perception_score.analysis["difficulty"]["easy"].accuracy == 1.0
    assert perception_score.analysis["difficulty"]["medium"].accuracy == 0.5
    assert perception_score.analysis["difficulty"]["hard"].accuracy == 0.0

    cognition = TaskScorer("minds_eye")
    cognition._questions = {}
    cognition._gt = {
        "c1": {"answer": "A", "group": "analogical_reasoning", "capability": "analogical_reasoning"},
        "c2": {"answer": "A", "group": "hierarchical_reasoning", "capability": "hierarchical_reasoning"},
        "c3": {"answer": "A", "group": "dynamic_reasoning", "capability": "dynamic_reasoning"},
        "c4": {"answer": "A", "group": "symmetry_analysis", "capability": "symmetry_analysis"},
        "c5": {"answer": "A", "group": "mental_rotation", "capability": "mental_rotation"},
        "c6": {"answer": "A", "group": "mental_composition", "capability": "mental_composition"},
    }
    cognition_score = cognition.score_predictions(
        {
            "standard": {
                "c1": "A",
                "c2": "A",
                "c3": "A",
                "c4": "B",
                "c5": "B",
                "c6": "B",
            }
        },
        "cognition-model",
    )
    assert cognition_score.analysis["art"]["abstraction"].accuracy == 1.0
    assert cognition_score.analysis["art"]["relation"].accuracy == 0.5
    assert cognition_score.analysis["art"]["transformation"].accuracy == 0.0
    assert cognition_score.to_dict()["analysis"]["art"]["relation"]["total_samples"] == 2


def test_dysm_headline_score_balances_dimensions_instead_of_samples():
    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {}
    scorer._gt = {
        **{
            f"p{index}": {
                "answer": "A",
                "group": "shape",
                "capability": "shape",
                "dimension": "2D",
                "difficulty": "easy",
            }
            for index in range(1, 5)
        },
        "p5": {
            "answer": "A",
            "group": "closure",
            "capability": "closure",
            "dimension": "3D",
            "difficulty": "easy",
        },
    }

    score = scorer.score_predictions(
        {"standard": {"p1": "A", "p2": "A", "p3": "A", "p4": "A", "p5": "B"}},
        "dimension-balanced-model",
    )

    assert score.accuracy == pytest.approx(0.8)
    assert score.macro_accuracy == pytest.approx(0.5)
    assert score.score_method == "dimension_balanced_task_macro"
    assert set(score.analysis["task_variant"]) == {"2D:shape", "3D:closure"}
    assert score.task_spread == score.accuracy_std
    serialized = score.to_dict()
    assert serialized["accuracy"] == 0.8
    assert serialized["micro_accuracy"] == 0.8
    assert serialized["macro_accuracy"] == 0.5
    assert serialized["score_method"] == "dimension_balanced_task_macro"


def test_minds_eye_random_baseline_uses_task_option_counts():
    scorer = TaskScorer("minds_eye")
    scorer._questions = {}
    task_names = [
        "analogies",
        "hierarchical_isomorphism",
        "dynamic_isomorphism",
        "slippage",
        "symmetric_structures",
        "mrt",
        "paper_folding",
        "mental_composition",
    ]
    scorer._gt = {
        f"q{index}": {"answer": "A", "task": task, "group": task}
        for index, task in enumerate(task_names)
    }

    assert scorer._random_baseline() == pytest.approx(11 / 48)


# --------------------------------------------------------------------------- #
# 4. Spatial diagnostics math
# --------------------------------------------------------------------------- #
def test_spatial_diagnostics_math():
    print("Testing spatial diagnostics math...")
    scorer = TaskScorer("spatial")
    # Inject a tiny controlled GT so the arithmetic is exact and no dataset
    # files are needed.
    scorer._questions = {
        sid: {"options": {"A": "left", "B": "right", "C": "above", "D": "below"}}
        for sid in ("s1", "s2", "s3", "s4")
    }
    scorer._gt = {
        "s1": {"answer": "A", "condition_answers": {"no_image_plus_noncot": "E", "no_image_plus_cot": "E"}, "group": "g", "dataset": "g", "evaluation_group": "s1"},
        "s2": {"answer": "B", "condition_answers": {"no_image_plus_noncot": "E", "no_image_plus_cot": "E"}, "group": "g", "dataset": "g", "evaluation_group": "s2"},
        "s3": {"answer": "A", "condition_answers": {"no_image_plus_noncot": "E", "no_image_plus_cot": "E"}, "group": "g", "dataset": "g", "evaluation_group": "s3"},
        "s4": {"answer": "C", "condition_answers": {"no_image_plus_noncot": "E", "no_image_plus_cot": "E"}, "group": "g", "dataset": "g", "evaluation_group": "s4"},
    }

    predictions = {
        "main_noncot": {"s1": "A", "s2": "X", "s3": "A", "s4": "X"},
        "main_cot": {"s1": "A", "s2": "B", "s3": "A", "s4": "X"},
        "no_image_noncot": {"s1": "A", "s2": "X", "s3": "X", "s4": "X"},
        "no_image_cot": {"s1": "A", "s2": "B", "s3": "X", "s4": "X"},
        "no_image_plus_noncot": {"s1": "cannot determine", "s2": "E",
                                     "s3": "unknown", "s4": "B"},
        "no_image_plus_cot": {"s1": "E", "s2": "E", "s3": "A", "s4": "B"},
    }

    diag = scorer._compute_diagnostics(predictions)
    assert diag is not None
    assert abs(diag.standard_accuracy - 0.50) < 1e-9, diag.standard_accuracy
    assert abs(diag.cot_accuracy - 0.75) < 1e-9, diag.cot_accuracy
    assert abs(diag.cot_delta - 0.25) < 1e-9, diag.cot_delta
    assert abs(diag.shortcut_score - 0.25) < 1e-9, diag.shortcut_score
    assert abs(diag.shortcut_score_cot - 0.50) < 1e-9, diag.shortcut_score_cot
    assert abs(diag.hallucination_resistance - 0.75) < 1e-9, diag.hallucination_resistance
    assert abs(diag.hallucination_resistance_cot - 0.50) < 1e-9, diag.hallucination_resistance_cot
    print("\u2713 spatial diagnostics math tests passed")


def test_spatial_circular_variants_score_as_one_all_or_nothing_group():
    scorer = TaskScorer("spatial")
    scorer._gt = {
        "d:1:r0": {"answer": "A", "dataset": "D", "evaluation_group": "d:1"},
        "d:1:r1": {"answer": "B", "dataset": "D", "evaluation_group": "d:1"},
        "d:2": {"answer": "C", "dataset": "D", "evaluation_group": "d:2"},
    }

    correct, total, groups = scorer._spatial_condition_result(
        "main_noncot",
        {"d:1:r0": "A", "d:1:r1": "X", "d:2": "C"},
    )

    assert (correct, total) == (1, 2)
    assert groups == {"D": [1, 2]}


def test_shared_task_scorer_keeps_concurrent_grading_provenance_isolated(monkeypatch):
    scorer = TaskScorer("do_you_see_me")
    scorer._gt = {
        "sample-1": {
            "answer": "A",
            "capability": "shape_discrimination",
            "group": "shape_discrimination",
            "dimension": "2D",
            "difficulty": "easy",
        },
    }
    scorer._questions = {}
    started = threading.Barrier(2)
    graded = threading.Barrier(2)
    original_grade = scorer._grade

    def synchronized_grade(prediction, answer, sample_id):
        started.wait(timeout=5)
        result = original_grade(prediction, answer, sample_id)
        graded.wait(timeout=5)
        return result

    monkeypatch.setattr(scorer, "_grade", synchronized_grade)

    with ThreadPoolExecutor(max_workers=2) as executor:
        scores = list(executor.map(
            lambda answer: scorer.score_predictions(
                {"standard": {"sample-1": answer}},
                model_name=f"Concurrent {answer}",
            ),
            ("A", "B"),
        ))

    assert [score.correct_samples for score in scores] == [1, 0]
    assert all(
        score.grading["method_counts"] == {"jsonl_exact": 1}
        for score in scores
    )
