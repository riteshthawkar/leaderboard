"""Transactional schema version tracking for the single-instance database."""

from datetime import datetime, timezone
from typing import Callable, Iterable

from sqlalchemy import text

Migration = tuple[int, Callable]


def _ensure_version_table(connection) -> None:
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS app_schema_versions ("
        "component VARCHAR(64) PRIMARY KEY, "
        "version INTEGER NOT NULL, "
        "updated_at VARCHAR(64) NOT NULL"
        ")"
    ))


def current_schema_version(connection, component: str) -> int:
    _ensure_version_table(connection)
    value = connection.execute(
        text("SELECT version FROM app_schema_versions WHERE component = :component"),
        {"component": component},
    ).scalar()
    return int(value or 0)


def run_schema_migrations(
    connection,
    component: str,
    migrations: Iterable[Migration],
) -> int:
    """Apply ordered, idempotent migrations and record each committed version."""
    ordered = sorted(migrations, key=lambda item: item[0])
    if not ordered or ordered[0][0] <= 0:
        raise ValueError("At least one positive schema migration version is required.")
    versions = [version for version, _migration in ordered]
    if versions != list(range(1, versions[-1] + 1)):
        raise ValueError(f"Schema migrations for {component} must be contiguous from version 1.")

    current = current_schema_version(connection, component)
    target = versions[-1]
    if current > target:
        raise RuntimeError(
            f"Database schema for {component} is version {current}, newer than this service supports ({target})."
        )

    for version, migration in ordered:
        if version <= current:
            continue
        migration(connection)
        updated_at = datetime.now(timezone.utc).isoformat()
        existing = connection.execute(
            text("SELECT 1 FROM app_schema_versions WHERE component = :component"),
            {"component": component},
        ).scalar()
        if existing:
            connection.execute(
                text(
                    "UPDATE app_schema_versions SET version = :version, updated_at = :updated_at "
                    "WHERE component = :component"
                ),
                {"component": component, "version": version, "updated_at": updated_at},
            )
        else:
            connection.execute(
                text(
                    "INSERT INTO app_schema_versions (component, version, updated_at) "
                    "VALUES (:component, :version, :updated_at)"
                ),
                {"component": component, "version": version, "updated_at": updated_at},
            )
        current = version
    return current
