"""
Grading / scoring tests for the Combined Leaderboard.

Covers the pieces that are easy to get wrong and expensive to debug in
production:

  * the pure grading helpers (``normalize`` / ``deterministic_match`` /
    ``is_cannot_determine``);
  * ``LLMGrader.grade`` with the OpenAI call **mocked** – the LLM extract and
    LLM judge paths, plus graceful fallback to deterministic matching when no
    key is configured or the network call fails;
  * malformed submission uploads (bad JSON, missing CSV columns, unsupported
    extension, empty predictions);
  * the spatial diagnostics math (CoT delta, shortcut score, hallucination
    resistance).

These tests are fully offline and hermetic – no API key and no dataset files
are required. Run directly (``python test_grading.py``) or via pytest.
"""

import sys
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add backend to path (same convention as test_system.py)
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from scoring.llm_grader import (  # noqa: E402
    LLMGrader,
    normalize,
    deterministic_match,
    is_cannot_determine,
    _format_options,
)
from scoring.task_scorer import TaskScorer  # noqa: E402


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
def test_malformed_uploads():
    print("Testing malformed uploads...")
    scorer = TaskScorer("do_you_see_me")

    # Invalid JSON content.
    bad_json = _write_temp("{not valid json", ".json")
    try:
        _expect_error(lambda: scorer.parse_submission(bad_json))
    finally:
        bad_json.unlink(missing_ok=True)

    # CSV missing the required columns.
    bad_csv = _write_temp("foo,bar\n1,2\n", ".csv")
    try:
        _expect_error(lambda: scorer.parse_submission(bad_csv), ValueError)
    finally:
        bad_csv.unlink(missing_ok=True)

    # Unsupported extension.
    bad_ext = _write_temp("whatever", ".txt")
    try:
        _expect_error(lambda: scorer.parse_submission(bad_ext), ValueError)
    finally:
        bad_ext.unlink(missing_ok=True)

    # Well-formed JSON but no predictions at all.
    empty = _write_temp(json.dumps({"predictions": {}}), ".json")
    try:
        _expect_error(lambda: scorer.parse_submission(empty), ValueError)
    finally:
        empty.unlink(missing_ok=True)

    # Sanity: a well-formed CSV parses into 'standard' predictions.
    good_csv = _write_temp("sample_id,prediction\ns1,A\ns2,B\n", ".csv")
    try:
        preds, _ = scorer.parse_submission(good_csv)
        assert preds["standard"] == {"s1": "A", "s2": "B"}
    finally:
        good_csv.unlink(missing_ok=True)
    print("\u2713 malformed upload tests passed")


# --------------------------------------------------------------------------- #
# 4. Spatial diagnostics math
# --------------------------------------------------------------------------- #
def test_spatial_diagnostics_math():
    print("Testing spatial diagnostics math...")
    scorer = TaskScorer("spatial")
    # Force offline deterministic grading and inject a tiny controlled GT so the
    # arithmetic is exact and no dataset files are needed.
    scorer.grader._backend = ""
    scorer._questions = {}
    scorer._gt = {
        "s1": {"answer": "A", "group": "g"},
        "s2": {"answer": "B", "group": "g"},
        "s3": {"answer": "A", "group": "g"},
        "s4": {"answer": "C", "group": "g"},
    }

    predictions = {
        # standard: s1,s3 correct  -> 2/4 = 0.50
        "standard": {"s1": "A", "s2": "X", "s3": "A", "s4": "X"},
        # cot: s1,s2,s3 correct    -> 3/4 = 0.75  (delta +0.25)
        "cot": {"s1": "A", "s2": "B", "s3": "A", "s4": "X"},
        # no_image: only s1 correct-> 1/4 = 0.25  (shortcut score)
        "no_image": {"s1": "A", "s2": "X", "s3": "X", "s4": "X"},
        # no_image_plus: 2 of 4 correctly abstain -> 0.50 (hallucination resist.)
        "no_image_plus": {"s1": "cannot determine", "s2": "A",
                          "s3": "unknown", "s4": "B"},
    }

    diag = scorer._compute_diagnostics(predictions)
    assert diag is not None
    assert abs(diag.standard_accuracy - 0.50) < 1e-9, diag.standard_accuracy
    assert abs(diag.cot_accuracy - 0.75) < 1e-9, diag.cot_accuracy
    assert abs(diag.cot_delta - 0.25) < 1e-9, diag.cot_delta
    assert abs(diag.shortcut_score - 0.25) < 1e-9, diag.shortcut_score
    assert abs(diag.hallucination_resistance - 0.50) < 1e-9, diag.hallucination_resistance
    print("\u2713 spatial diagnostics math tests passed")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_all_tests():
    print("\n" + "=" * 50)
    print("Combined Leaderboard - Grading / Scoring Tests")
    print("=" * 50 + "\n")

    tests = [
        test_normalize,
        test_deterministic_match,
        test_is_cannot_determine,
        test_format_options,
        test_llm_extract_path,
        test_llm_judge_path,
        test_deterministic_fallback_no_key,
        test_fallback_on_network_failure,
        test_malformed_uploads,
        test_spatial_diagnostics_math,
    ]

    try:
        for t in tests:
            t()
        print("\n" + "=" * 50)
        print("\u2713 All grading tests passed!")
        print("=" * 50 + "\n")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\n\u2717 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
