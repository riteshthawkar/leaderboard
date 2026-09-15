import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seed_published_results.py"
SPEC = importlib.util.spec_from_file_location("published_seed", SCRIPT)
seed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed)


def database(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE registered_models(id TEXT PRIMARY KEY, owner_email TEXT);
        CREATE TABLE spatial_benchmark_contracts(manifest_sha256 TEXT PRIMARY KEY);
        CREATE TABLE submissions(
            id INTEGER PRIMARY KEY, model_id TEXT, task_id TEXT, model_name TEXT,
            user_email TEXT, status TEXT, moderation_status TEXT, created_at TEXT,
            row_count INTEGER, score_submission_id TEXT, latest_score_json TEXT,
            model_meta_json TEXT, spatial_contract_sha256 TEXT
        );
        CREATE TABLE submission_answers(
            id INTEGER PRIMARY KEY, submission_id INTEGER REFERENCES submissions(id),
            raw_answer_text TEXT, answer_sha256 TEXT
        );
        CREATE TABLE submission_artifacts(
            id INTEGER PRIMARY KEY, submission_id INTEGER REFERENCES submissions(id),
            content BLOB, size_bytes INTEGER, sha256 TEXT
        );
        CREATE TABLE users(id INTEGER PRIMARY KEY, credential TEXT);
    """)
    connection.commit()
    return connection


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.sqlite"
    cache_path = tmp_path / "source.json"
    with database(path) as connection:
        connection.execute("INSERT INTO users VALUES (1,'must-not-export')")
        connection.executemany("INSERT INTO registered_models VALUES (?,?)", [
            ("model", "importer@example.invalid"), ("unpublished", "someone@example.invalid"),
        ])
        for number, status in [(1, "visible"), (2, "visible"), (3, "deleted")]:
            score = {
                "submission_id": str(number), "model_id": "model", "task_id": "minds_eye",
                "model_name": "Example", "model_meta": {}, "accuracy": 50,
            }
            connection.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                number, "model", "minds_eye", "Example", "importer@example.invalid",
                "scored", status, f"2026-01-0{number}", 1, str(number), json.dumps(score), "{}", None,
            ))
            answer = f"answer-{number}"
            connection.execute("INSERT INTO submission_answers VALUES (?,?,?,?)", (
                number, number, answer, hashlib.sha256(answer.encode()).hexdigest(),
            ))
            artifact = answer.encode()
            connection.execute("INSERT INTO submission_artifacts VALUES (?,?,?,?,?)", (
                number, number, artifact, len(artifact), hashlib.sha256(artifact).hexdigest(),
            ))
            if number == 2:
                cache_path.write_text(json.dumps({"models": {
                    "model": {"model_id": "model", "tasks": {"minds_eye": score}},
                }}))
    return path, cache_path


def bundle(tmp_path, source):
    destination = tmp_path / "bundle"
    seed.export_bundle(*source, destination)
    return destination


def test_export_selects_latest_published_evidence_without_accounts(tmp_path, source):
    exported = bundle(tmp_path, source)
    with sqlite3.connect(exported / "published.sqlite") as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == set(seed.TABLES)
        assert connection.execute("SELECT id FROM submissions").fetchall() == [(2,)]
        assert connection.execute("SELECT id FROM registered_models").fetchall() == [("model",)]
        assert connection.execute("SELECT raw_answer_text FROM submission_answers").fetchone()[0] == "answer-2"
    assert json.loads((exported / "manifest.json").read_text())["authentication_included"] is False
    assert (exported / "published.sqlite").stat().st_mode & 0o777 == 0o600


def test_dry_run_and_apply_preserve_live_accounts_and_support_retry(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    cache = tmp_path / "target.json"
    with database(target) as connection:
        connection.execute("INSERT INTO users VALUES (1,'new-admin-secret')")
    cache.write_text('{"models": {}}')
    backup = tmp_path / "before"
    report = seed.apply_bundle(exported, target, cache, backup)
    assert report["mode"] == "dry-run"
    assert not backup.exists()
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
    seed.apply_bundle(exported, target, cache, backup, apply=True)
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT credential FROM users").fetchone()[0] == "new-admin-secret"
        assert connection.execute("SELECT id FROM submissions").fetchall() == [(2,)]
    with sqlite3.connect(backup / "before.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
        assert connection.execute("SELECT credential FROM users").fetchone()[0] == "new-admin-secret"
    assert cache.read_bytes() == source[1].read_bytes()
    retry = seed.apply_bundle(exported, target, cache, tmp_path / "retry", apply=True)
    assert retry["already_imported"] is True


def test_conflicting_live_results_are_not_overwritten(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    with database(target) as connection:
        connection.execute("INSERT INTO registered_models VALUES ('new-user-model','new-owner')")
    with pytest.raises(ValueError, match="Target already contains results"):
        seed.apply_bundle(exported, target, tmp_path / "cache.json", tmp_path / "backup", apply=True)
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT id FROM registered_models").fetchall() == [("new-user-model",)]


def test_apply_maps_reordered_columns_by_name(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    with sqlite3.connect(exported / "published.sqlite") as original, sqlite3.connect(target) as connection:
        for table in seed.TABLES:
            info = original.execute(f'PRAGMA table_info("{table}")').fetchall()
            definitions = [f'"{row[1]}" {row[2]}' + (" PRIMARY KEY" if row[5] else "") for row in reversed(info)]
            connection.execute(f'CREATE TABLE "{table}" ({",".join(definitions)})')
    report = seed.apply_bundle(exported, target, tmp_path / "cache.json", tmp_path / "before", apply=True)
    with sqlite3.connect(target) as connection:
        assert seed.table_profile(connection) == report["tables"]
        assert connection.execute("SELECT raw_answer_text FROM submission_answers").fetchone()[0] == "answer-2"


def test_changed_column_schema_is_rejected(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    with database(target) as connection:
        connection.execute("ALTER TABLE submissions ADD COLUMN new_required_field TEXT")
    with pytest.raises(ValueError, match="Schema mismatch: submissions"):
        seed.apply_bundle(exported, target, tmp_path / "cache.json", tmp_path / "before", apply=True)
    assert not (tmp_path / "before").exists()


def test_tampered_bundle_is_rejected_before_writes(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    database(target).close()
    (exported / "leaderboard_store.json").write_text('{"models":{}}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        seed.apply_bundle(exported, target, tmp_path / "cache.json", tmp_path / "backup", apply=True)
    assert not (tmp_path / "backup").exists()


def test_cache_score_mismatch_fails_export(tmp_path, source):
    cache = json.loads(source[1].read_text())
    cache["models"]["model"]["tasks"]["minds_eye"]["accuracy"] = 99
    source[1].write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="cache and stored scores disagree"):
        bundle(tmp_path, source)


def test_changed_answer_content_fails_export(tmp_path, source):
    with sqlite3.connect(source[0]) as connection:
        connection.execute("UPDATE submission_answers SET raw_answer_text='changed' WHERE id=2")
    with pytest.raises(ValueError, match="Answer content hash mismatch"):
        bundle(tmp_path, source)


def test_failed_insert_rolls_back_all_result_tables(tmp_path, source):
    exported = bundle(tmp_path, source)
    target = tmp_path / "target.sqlite"
    cache = tmp_path / "cache.json"
    cache.write_text('{"models": {}}')
    with database(target) as connection:
        connection.execute("INSERT INTO users VALUES (1,'keep-admin')")
        connection.executescript("""CREATE TRIGGER block_seed BEFORE INSERT ON submission_answers
            BEGIN SELECT RAISE(ABORT, 'test rollback'); END;""")
    with pytest.raises(sqlite3.IntegrityError, match="test rollback"):
        seed.apply_bundle(exported, target, cache, tmp_path / "backup", apply=True)
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT COUNT(*) FROM registered_models").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
        assert connection.execute("SELECT credential FROM users").fetchone()[0] == "keep-admin"
    assert json.loads(cache.read_text()) == {"models": {}}
