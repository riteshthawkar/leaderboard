#!/usr/bin/env python3
"""Transfer published results into an empty deployment without migrating accounts.

Export includes only the latest visible scored run per model/task and its
evidence. Apply is dry-run by default, refuses nonempty/conflicting result
stores, and never writes authentication tables. Stop the API while applying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


TABLES = (
    "registered_models", "spatial_benchmark_contracts", "submissions",
    "submission_answers", "submission_artifacts",
)
FILES = ("published.sqlite", "leaderboard_store.json")


def connect(path: Path, mode: str = "ro") -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]


def column_signature(connection: sqlite3.Connection, table: str) -> dict:
    return {row[1]: (row[2].upper(), row[5]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def quoted_columns(names: list[str]) -> str:
    return ",".join('"' + name.replace('"', '""') + '"' for name in names)


def table_profile(connection: sqlite3.Connection) -> dict:
    result = {}
    for table in TABLES:
        digest = hashlib.sha256()
        count = 0
        signature = column_signature(connection, table)
        selection = quoted_columns(sorted(signature))
        primary = sorted((spec[1], name) for name, spec in signature.items() if spec[1])
        if not primary:
            raise ValueError(f"Primary key missing: {table}")
        ordering = quoted_columns([name for _, name in primary])
        for row in connection.execute(f'SELECT {selection} FROM "{table}" ORDER BY {ordering}'):
            count += 1
            for value in row:
                payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
                digest.update(b"B" if isinstance(value, bytes) else b"J")
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
        result[table] = {"rows": count, "sha256": digest.hexdigest()}
    return result


def validate(connection: sqlite3.Connection, cache: dict) -> dict:
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("Orphan evidence records found")
    expected = {}
    seen = set()
    answer_counts = dict(connection.execute(
        "SELECT submission_id, COUNT(*) FROM submission_answers GROUP BY submission_id",
    ))
    for row in connection.execute("""
        SELECT s.*, m.owner_email FROM submissions s
        LEFT JOIN registered_models m ON m.id=s.model_id
    """):
        key = (row["model_id"], row["task_id"])
        if key in seen or row["status"] != "scored" or row["moderation_status"] != "visible":
            raise ValueError("Bundle must contain only one published run per model/task")
        seen.add(key)
        if not row["owner_email"] or row["owner_email"] != row["user_email"]:
            raise ValueError("Model ownership mismatch")
        actual = answer_counts.get(row["id"], 0)
        if actual != row["row_count"]:
            raise ValueError("Stored answer count mismatch")
        score = json.loads(row["latest_score_json"])
        score.update({
            "submission_id": row["score_submission_id"], "model_id": row["model_id"],
            "model_name": row["model_name"], "task_id": row["task_id"],
            "model_meta": json.loads(row["model_meta_json"]),
        })
        if not row["score_submission_id"] or row["score_submission_id"] in expected:
            raise ValueError("Missing or duplicate public submission ID")
        expected[row["score_submission_id"]] = canonical_hash(score)
    actual = {}
    for key, model in cache["models"].items():
        if key != model["model_id"]:
            raise ValueError("Cache model ID mismatch")
        for score in model["tasks"].values():
            submission_id = score["submission_id"]
            if submission_id in actual or score["model_id"] != key:
                raise ValueError("Duplicate or mismatched cached submission")
            actual[submission_id] = canonical_hash(score)
    if not expected or actual != expected:
        raise ValueError("Published cache and stored scores disagree")
    for row in connection.execute("SELECT raw_answer_text, answer_sha256 FROM submission_answers"):
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("Answer content hash mismatch")
    for row in connection.execute("SELECT content, size_bytes, sha256 FROM submission_artifacts"):
        if len(row[0]) != row[1] or hashlib.sha256(row[0]).hexdigest() != row[2]:
            raise ValueError("Evidence artifact hash mismatch")
    return table_profile(connection)


def export_bundle(source_db: Path, source_cache: Path, destination: Path) -> dict:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    with closing(connect(source_db)) as source, closing(connect(destination / FILES[0], "rwc")) as target:
        source.execute("BEGIN")
        source.execute("""CREATE TEMP TABLE selected AS
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY model_id, task_id ORDER BY created_at DESC, id DESC
                ) AS position FROM submissions
                WHERE status='scored' AND moderation_status='visible' AND model_id IS NOT NULL
            ) WHERE position=1""")
        selectors = {
            "submissions": "SELECT s.* FROM submissions s JOIN selected ON selected.id=s.id",
            "registered_models": "SELECT * FROM registered_models WHERE id IN (SELECT model_id FROM submissions JOIN selected USING(id))",
            "spatial_benchmark_contracts": "SELECT * FROM spatial_benchmark_contracts WHERE manifest_sha256 IN (SELECT spatial_contract_sha256 FROM submissions JOIN selected USING(id))",
            "submission_answers": "SELECT a.* FROM submission_answers a JOIN selected ON selected.id=a.submission_id",
            "submission_artifacts": "SELECT a.* FROM submission_artifacts a JOIN selected ON selected.id=a.submission_id",
        }
        for table in TABLES:
            schema = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if schema is None:
                raise ValueError(f"Source table missing: {table}")
            target.execute(schema[0])
            placeholders = ",".join("?" for _ in columns(source, table))
            target.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', source.execute(selectors[table]))
        target.commit()
        cache_bytes = source_cache.read_bytes()
        profile = validate(target, json.loads(cache_bytes))
        (destination / FILES[1]).write_bytes(cache_bytes)
    manifest = {
        "format": "ms-vista-published-seed-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tables": profile,
        "files": {name: file_hash(destination / name) for name in FILES},
        "authentication_included": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in destination.iterdir():
        path.chmod(0o600)
    return manifest


def apply_bundle(bundle: Path, target_db: Path, target_cache: Path, backup_dir: Path, *, apply: bool = False) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format") != "ms-vista-published-seed-v1" or manifest.get("authentication_included") is not False:
        raise ValueError("Unsupported seed manifest")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Unexpected bundle files")
    for name in FILES:
        if file_hash(bundle / name) != manifest["files"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    cache_bytes = (bundle / FILES[1]).read_bytes()
    cache = json.loads(cache_bytes)
    with closing(connect(bundle / FILES[0])) as source, closing(connect(target_db, "rw")) as target:
        tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != set(TABLES):
            raise ValueError("Bundle contains unapproved tables")
        profile = validate(source, cache)
        if profile != manifest["tables"]:
            raise ValueError("Bundle table fingerprint mismatch")
        for table in TABLES:
            if column_signature(source, table) != column_signature(target, table):
                raise ValueError(f"Schema mismatch: {table}")
        current = table_profile(target)
        exact = current == profile
        if not exact and any(item["rows"] for item in current.values()):
            raise ValueError("Target already contains results; refusing to replace or merge them")
        existing_cache = json.loads(target_cache.read_bytes()) if target_cache.exists() else {"models": {}}
        if existing_cache.get("models") and canonical_hash(existing_cache) != canonical_hash(cache):
            raise ValueError("Target already contains different published rankings")
        report = {"mode": "apply" if apply else "dry-run", "already_imported": exact, "tables": profile}
        if not apply:
            return report
        backup_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        with closing(sqlite3.connect(backup_dir / "before.sqlite")) as backup:
            target.backup(backup)
        (backup_dir / "leaderboard_store.json").write_text(json.dumps(existing_cache))
        for path in backup_dir.iterdir():
            path.chmod(0o600)
        # The caller keeps the API stopped until both the SQL store and cache pass validation.
        target.execute("BEGIN IMMEDIATE")
        try:
            if table_profile(target) != current:
                raise ValueError("Target changed during preflight; retry with the API stopped")
            if not exact:
                for table in TABLES:
                    names = columns(source, table)
                    placeholders = ",".join("?" for _ in names)
                    selection = quoted_columns(names)
                    target.executemany(
                        f'INSERT INTO "{table}" ({selection}) VALUES ({placeholders})',
                        source.execute(f'SELECT {selection} FROM "{table}"'),
                    )
            if validate(target, cache) != profile:
                raise ValueError("Imported evidence does not match the source")
            target.commit()
        except Exception:
            target.rollback()
            raise
        target_cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = target_cache.with_name(target_cache.name + ".seed-tmp")
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                owner = (target_cache if target_cache.exists() else target_db).stat()
                os.chown(temporary, owner.st_uid, owner.st_gid)
            handle.write(cache_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target_cache)
        report["backup"] = str(backup_dir)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source-db", type=Path, required=True)
    export.add_argument("--source-cache", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    load = commands.add_parser("import")
    load.add_argument("--bundle", type=Path, required=True)
    load.add_argument("--target-db", type=Path, required=True)
    load.add_argument("--target-cache", type=Path, required=True)
    load.add_argument("--backup-dir", type=Path, required=True)
    load.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        report = export_bundle(args.source_db, args.source_cache, args.destination)
    else:
        report = apply_bundle(args.bundle, args.target_db, args.target_cache, args.backup_dir, apply=args.apply)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
