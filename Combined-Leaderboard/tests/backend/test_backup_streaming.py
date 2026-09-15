"""Backups must stay bounded as submitted evidence enlarges the database."""

import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import pytest

import backup
import backup_cli


def test_backup_mirror_restore_and_cli_stream_large_payloads(tmp_path, monkeypatch, capsys):
    database = tmp_path / "leaderboard.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, payload BLOB)")
        connection.executemany(
            "INSERT INTO evidence (payload) VALUES (randomblob(65536))",
            [()] * 40,
        )
        connection.commit()
    cache = tmp_path / "leaderboard_store.json"
    cache.write_text('{"models":{}}', encoding="utf-8")

    original_connect = sqlite3.connect
    original_zip_read = zipfile.ZipFile.read

    def disk_connect(database, *args, **kwargs):
        assert database != ":memory:", "SQLite snapshots must not copy the DB into RAM"
        return original_connect(database, *args, **kwargs)

    def metadata_read(archive, name, *args, **kwargs):
        assert name == "manifest.json", "Large ZIP entries must be streamed"
        return original_zip_read(archive, name, *args, **kwargs)

    def reject_whole_file_read(_path):
        pytest.fail("Backup files must be streamed, not read completely into memory")

    monkeypatch.setattr(sqlite3, "connect", disk_connect)
    monkeypatch.setattr(zipfile.ZipFile, "read", metadata_read)
    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
    arguments = {
        "database_urls": {"main": f"sqlite:///{database}"},
        "extra_files": {"leaderboard_store": cache},
    }
    stream, _name, _manifest = backup.create_backup_archive(**arguments)
    with stream:
        assert stream._rolled is True
        assert backup.validate_backup_archive(stream)["sqlite_snapshots"] == 1

    archive, _manifest = backup.write_backup_archive(
        tmp_path / "primary", retention_count=2, **arguments
    )
    mirror, validation = backup.mirror_backup_archive(
        archive, tmp_path / "mirror", retention_count=2
    )
    assert validation["source_sqlite_snapshots"] == 1
    assert validation["mirror_sqlite_snapshots"] == 1
    assert backup_cli.main(["verify", str(mirror)]) == 0
    assert '"sqlite_quick_check": "ok"' in capsys.readouterr().out
    recovery = tmp_path / "recovery"
    backup.restore_backup_archive(mirror, recovery)
    with closing(sqlite3.connect(next((recovery / "sqlite").glob("*.db")))) as connection:
        assert connection.execute("SELECT count(*), sum(length(payload)) FROM evidence").fetchone() == (
            40, 40 * 65536
        )
    assert archive.stat().st_mode & 0o777 == 0o600
    assert mirror.stat().st_mode & 0o777 == 0o600


def test_failed_mirror_verification_preserves_existing_archive(tmp_path, monkeypatch):
    database = tmp_path / "leaderboard.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY)")
        connection.commit()
    archive, _manifest = backup.write_backup_archive(
        tmp_path / "primary",
        retention_count=2,
        database_urls={"main": f"sqlite:///{database}"},
        extra_files={"missing_cache": tmp_path / "missing.json"},
    )
    mirror, _validation = backup.mirror_backup_archive(
        archive, tmp_path / "mirror", retention_count=2
    )
    original_digest = backup._file_sha256
    verified_digest = original_digest(mirror)

    def damaged_digest(path):
        return "0" * 64 if path.suffix == ".tmp" else original_digest(path)

    monkeypatch.setattr(backup, "_file_sha256", damaged_digest)
    with pytest.raises(RuntimeError, match="checksum does not match"):
        backup.mirror_backup_archive(archive, mirror.parent, retention_count=2)
    assert original_digest(mirror) == verified_digest
    assert not list(mirror.parent.glob("*.tmp"))


def test_backup_creation_closes_temporary_stream_on_failure(tmp_path, monkeypatch):
    database = tmp_path / "leaderboard.db"
    database.touch()
    original_spool = backup.tempfile.SpooledTemporaryFile
    streams = []

    def spool(*args, **kwargs):
        stream = original_spool(*args, **kwargs)
        streams.append(stream)
        return stream

    def failed_snapshot(*_args):
        raise OSError("No space left on device")

    monkeypatch.setattr(backup.tempfile, "SpooledTemporaryFile", spool)
    monkeypatch.setattr(backup, "_write_sqlite_snapshot", failed_snapshot)
    with pytest.raises(OSError, match="No space"):
        backup.create_backup_archive(database_urls={"main": f"sqlite:///{database}"})
    assert len(streams) == 1
    assert streams[0].closed
