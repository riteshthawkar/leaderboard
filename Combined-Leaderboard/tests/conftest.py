"""Shared isolation and import paths for the repository test suite."""

import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="ms-vista-tests-"))
TEST_DATA_DIR = TEST_RUNTIME_ROOT / "data"
TEST_GROUND_TRUTHS_DIR = TEST_RUNTIME_ROOT / "ground-truths"


def _synthetic_answer(question: dict):
    """Return a valid-format placeholder without using a benchmark answer."""
    answer_type = str(question.get("answer_type") or "").strip()
    if answer_type in {"integer", "mcq_index_1_4"}:
        return 1
    if answer_type == "mcq_letter":
        return "A"
    return "test"


def _write_synthetic_visual_ground_truth() -> None:
    destinations = {
        "dysm_2d_v1": TEST_GROUND_TRUTHS_DIR / "dysm_2d_v1" / "ground_truth.jsonl",
        "dysm_3d_v1": TEST_GROUND_TRUTHS_DIR / "dysm_3d_v1" / "ground_truth.jsonl",
        "minds_eye_fresh_v1": (
            TEST_GROUND_TRUTHS_DIR
            / "minds_eye_fresh_v1"
            / "ground_truth.jsonl"
        ),
    }
    rows_by_subset = {subset: [] for subset in destinations}

    for task_id in ("do_you_see_me", "minds_eye"):
        questions_file = PROJECT_ROOT / "tasks" / task_id / "questions.jsonl"
        with questions_file.open("r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                if not raw_line.strip():
                    continue
                question = json.loads(raw_line)
                subset = str(question["source_subset"])
                rows_by_subset[subset].append(
                    {
                        "question_id": str(question["question_id"]),
                        "answer": _synthetic_answer(question),
                    }
                )

    for subset, output_file in destinations.items():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file_handle:
            for row in rows_by_subset[subset]:
                file_handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _configure_isolated_runtime() -> None:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{TEST_DATA_DIR / 'leaderboard.db'}"
    test_environment = {
        "LEADERBOARD_DATA_DIR": str(TEST_DATA_DIR),
        "RESULTS_DIR": str(TEST_DATA_DIR / "results"),
        "AUTO_BACKUP_DIR": str(TEST_DATA_DIR / "backups"),
        "AUTO_BACKUP_ENABLED": "false",
        "DATABASE_URL": database_url,
        "AUTH_DATABASE_URL": database_url,
        "SUBMISSION_DATABASE_URL": database_url,
        "LEADERBOARD_LOG_DIR": str(TEST_DATA_DIR / "logs"),
        "GROUND_TRUTHS_DIR": str(TEST_GROUND_TRUTHS_DIR),
        "GROUND_TRUTHS_SOURCE": "local",
        "SECRET_KEY": "test-only-secret-key-that-is-never-used-outside-pytest",
    }
    os.environ.update(test_environment)
    _write_synthetic_visual_ground_truth()


atexit.register(shutil.rmtree, TEST_RUNTIME_ROOT, ignore_errors=True)
_configure_isolated_runtime()

for import_root in (
    PROJECT_ROOT,
    PROJECT_ROOT / "backend",
    PROJECT_ROOT / "scripts",
):
    resolved = str(import_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
