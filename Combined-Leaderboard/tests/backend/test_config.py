from pathlib import Path

import pytest

from backend.config import (
    _assert_unique_dotenv_assignments,
    _duplicate_dotenv_keys,
    _validate_backup_configuration,
    _validate_sqlite_concurrency,
)


def test_duplicate_dotenv_keys_reports_names_without_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SECRET_KEY=first-secret\n"
        "export API_BASE_URL=https://api.example.com\n"
        "SECRET_KEY=second-secret\n",
        encoding="utf-8",
    )

    assert _duplicate_dotenv_keys(env_file) == ["SECRET_KEY"]
    with pytest.raises(RuntimeError) as captured:
        _assert_unique_dotenv_assignments(env_file)

    assert "SECRET_KEY" in str(captured.value)
    assert "first-secret" not in str(captured.value)
    assert "second-secret" not in str(captured.value)


def test_duplicate_dotenv_keys_ignores_comments_and_missing_files(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# SECRET_KEY=commented-example\n"
        "SECRET_KEY=active-value\n",
        encoding="utf-8",
    )

    assert _duplicate_dotenv_keys(env_file) == []
    assert _duplicate_dotenv_keys(tmp_path / "missing.env") == []


def test_public_sqlite_requires_one_worker():
    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=1"):
        _validate_sqlite_concurrency(
            True,
            ("sqlite:////data/leaderboard.db",),
            2,
        )

    _validate_sqlite_concurrency(True, ("sqlite:////data/leaderboard.db",), 1)
    _validate_sqlite_concurrency(True, ("postgresql://db/app",), 2)


def test_public_backup_requires_a_separate_mirror(tmp_path: Path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    with pytest.raises(RuntimeError, match="AUTO_BACKUP_MIRROR_DIR"):
        _validate_backup_configuration(True, True, backup_dir, None, True)
    with pytest.raises(RuntimeError, match="different filesystem"):
        mirror_dir = tmp_path / "mirror"
        mirror_dir.mkdir()
        _validate_backup_configuration(True, True, backup_dir, mirror_dir, True)
    with pytest.raises(RuntimeError, match="AUTO_BACKUP_ENABLED"):
        _validate_backup_configuration(True, False, backup_dir, None, False)
