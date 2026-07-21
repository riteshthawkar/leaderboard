"""SQLite runtime settings for the single-instance production deployment."""

import logging
import os
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine, make_url

logger = logging.getLogger(__name__)


def sqlite_database_path(db_url: str) -> Path | None:
    """Resolve a file-backed SQLite URL without touching non-SQLite databases."""
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).expanduser().resolve()


def harden_private_file(path: Path, mode: int = 0o600) -> None:
    """Restrict a runtime file when POSIX permissions are available."""
    if os.name == "nt" or not path.exists():
        return
    try:
        os.chmod(path, mode)
    except OSError as exc:
        logger.warning("Could not restrict runtime file permissions for %s: %s", path, exc)


def harden_sqlite_files(db_url: str) -> None:
    """Restrict the database and WAL sidecars to the service account."""
    path = sqlite_database_path(db_url)
    if path is None:
        return
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        harden_private_file(candidate)


def harden_private_directory(path: Path) -> None:
    """Create a private service-owned runtime directory."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        logger.warning("Could not restrict runtime directory permissions for %s: %s", path, exc)


def sqlite_connect_args(db_url: str, busy_timeout_ms: int) -> dict:
    if not make_url(db_url).drivername.startswith("sqlite"):
        return {}
    return {
        "check_same_thread": False,
        "timeout": busy_timeout_ms / 1000,
    }


def configure_sqlite_engine(
    engine: Engine,
    db_url: str,
    busy_timeout_ms: int,
) -> None:
    """Enable safe SQLite defaults for a low-concurrency web process."""
    if not make_url(db_url).drivername.startswith("sqlite"):
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()
        harden_sqlite_files(db_url)
