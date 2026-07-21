"""Regression tests for compatibility services still used by the API."""

import json
from pathlib import Path

from data_handlers.ground_truth import GroundTruthManager
from data_handlers.submission import SubmissionParser
from leaderboard_manager import LeaderboardManager
from scoring.engine import ScoringEngine
from utils.answer_extractor import AnswerComparator, OptionExtractor


def test_option_extraction():
    cases = [
        ("The answer is (C)", "C"),
        ("Option B is correct", "B"),
        ("(a)", "A"),
        ("I think it's D", "D"),
    ]

    for text, expected in cases:
        assert OptionExtractor.extract_option(text) == expected


def test_answer_comparison():
    cases = [
        ("A", "Option A", True),
        ("(c)", "C", True),
        ("1", "1", True),
        ("cat", "Cat", True),
        ("B", "A", False),
    ]

    for ground_truth, prediction, expected in cases:
        is_correct, _reason = AnswerComparator.compare_answers(ground_truth, prediction)
        assert is_correct is expected


def test_ground_truth_task_discovery_has_stable_shape():
    tasks = GroundTruthManager().list_available_tasks()

    assert set(tasks) == {"do_you_see_me", "minds_eye"}
    assert all(isinstance(task_names, list) for task_names in tasks.values())


def test_legacy_submission_parsers(tmp_path: Path):
    csv_file = tmp_path / "submission.csv"
    csv_file.write_text(
        "image_name,task_name,prediction\n"
        "img_0.png,task_1,A\n"
        "img_1.png,task_1,B\n",
        encoding="utf-8",
    )
    csv_predictions = SubmissionParser.parse_csv(csv_file)
    assert len(csv_predictions["task_1"]) == 2

    json_file = tmp_path / "submission.json"
    json_file.write_text(
        json.dumps({"task_1": {"img_0.png": "A", "img_1.png": "B"}}),
        encoding="utf-8",
    )
    json_predictions = SubmissionParser.parse_json(json_file)
    assert len(json_predictions["task_1"]) == 2


def test_scoring_engine_initializes_without_remote_grader():
    assert isinstance(ScoringEngine(use_ollama=False), ScoringEngine)


def test_leaderboard_manager_starts_empty(tmp_path: Path):
    manager = LeaderboardManager(leaderboard_file=tmp_path / "leaderboard.json")

    assert manager.get_leaderboard() == []
    assert manager.get_statistics()["total_submissions"] == 0
