import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("v2_review", Path(__file__).resolve().parents[2] / "scripts/review_track3_v2.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def archive_with_rows(rows):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("rows.jsonl", "\n".join(json.dumps(row) for row in rows))
    return zipfile.ZipFile(stream)


def prediction():
    return {"dataset": "BLINK", "index": "1", "group": "1", "answer_type": "mcq", "mode": "main", "pmode": "cot", "options": {"A": "one", "B": "two"}, "output": "A"}


def test_checks_every_prediction_for_answer_key_leaks():
    first = prediction()
    second = {**first, "index": "2", "gt": "A"}
    with archive_with_rows([first, second]) as archive, pytest.raises(ValueError, match="answer keys"):
        review.load_rows(archive, "rows.jsonl", "main_cot", verdict=False)


@pytest.mark.parametrize("change", [{}, {"mode": "noimg"}, {"answer_type": "other"}])
def test_duplicate_and_inconsistent_rows_are_rejected(change):
    with archive_with_rows([prediction(), {**prediction(), **change}]) as archive, pytest.raises(ValueError):
        review.load_rows(archive, "rows.jsonl", "main_cot", verdict=False)


def test_null_outputs_remain_visible_for_missing_evidence_diagnostics():
    with archive_with_rows([{**prediction(), "output": None}]) as archive:
        assert review.load_rows(archive, "rows.jsonl", "main_cot", verdict=False)[("BLINK", "1")]["output"] is None


def test_credit_uses_submitted_verdict_not_any_reference_answers():
    assert review.claimed_credit({"answer_type": "vqa", "judged": "1"})
    assert not review.claimed_credit({"answer_type": "vqa", "judged": "0"})
    assert review.claimed_credit({"answer_type": "mcq", "judged": "D", "cannot_label": "D", "gt": "A", "mode": "noimgpp"})
    assert not review.claimed_credit({"answer_type": "mcq", "judged": "D", "cannot_label": "D", "gt": "A", "mode": "main"})


@pytest.mark.parametrize("name", ["../metadata.json", "/metadata.json", "..\\metadata.json"])
def test_unsafe_archive_paths_are_rejected_without_extraction(name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, "{}")
    with zipfile.ZipFile(stream) as archive, pytest.raises(ValueError, match="unsafe path"):
        review.check_archive(archive)
