"""SQLite and leaderboard-cache backup helpers."""

import io
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Dict, Optional

from sqlalchemy.engine import make_url

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

logger = logging.getLogger(__name__)

try:
    from config import (
        AUTH_DATABASE_URL,
        DATABASE_URL,
        LEADERBOARD_STORE_FILE,
        SUBMISSION_DATABASE_URL,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        AUTH_DATABASE_URL,
        DATABASE_URL,
        LEADERBOARD_STORE_FILE,
        SUBMISSION_DATABASE_URL,
    )

try:
    from sqlite_runtime import harden_private_directory, harden_private_file
except ImportError:  # pragma: no cover - package import fallback
    from .sqlite_runtime import harden_private_directory, harden_private_file


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "database").strip("-")
    return label[:80] or "database"


def _sqlite_db_path(db_url: str) -> Optional[Path]:
    try:
        url = make_url(db_url)
    except Exception:
        return None
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).expanduser().resolve()


BACKUP_BUFFER_BYTES = 1024 * 1024


def _write_sqlite_snapshot(path: Path, destination: Path) -> None:
    """Keep a consistent online snapshot on disk, not in the API's memory."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(destination)) as target:
            harden_private_file(destination)
            source.execute("PRAGMA cache_size = -2048")
            target.execute("PRAGMA cache_size = -2048")
            source.backup(target, pages=256)


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    shutil.copyfileobj(source, destination, length=BACKUP_BUFFER_BYTES)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BACKUP_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def create_backup_archive(
    database_urls: Optional[Dict[str, str]] = None,
    extra_files: Optional[Dict[str, Path]] = None,
    now: Optional[datetime] = None,
) -> tuple[BinaryIO, str, dict]:
    """Build a ZIP archive containing SQLite snapshots and public cache files.

    Secrets such as `.env`, OAuth client secrets, ACS keys, and private ground
    truths are intentionally excluded. The caller owns the returned seekable
    stream and must close it; larger archives spill into a private temp file.
    """
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    database_urls = database_urls or {
        "database": DATABASE_URL,
        "auth": AUTH_DATABASE_URL,
        "submissions": SUBMISSION_DATABASE_URL,
    }
    extra_files = extra_files or {
        "leaderboard_store": LEADERBOARD_STORE_FILE,
        "submission_history": LEADERBOARD_STORE_FILE.parent / "submission_history.jsonl",
    }

    db_paths: Dict[Path, list[str]] = {}
    for label, url in database_urls.items():
        path = _sqlite_db_path(url)
        if path is not None:
            db_paths.setdefault(path, []).append(label)

    manifest = {
        "created_at": now.isoformat(),
        "format": "ms-vista-sqlite-backup-v1",
        "sqlite_databases": [],
        "files": [],
        "excluded": [".env", "ground_truths", "oauth_secrets", "acs_keys"],
    }

    archive = tempfile.SpooledTemporaryFile(max_size=BACKUP_BUFFER_BYTES, mode="w+b")
    try:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "README.txt",
                "MS-VISTA backup archive.\n"
                "Includes SQLite database snapshots and public leaderboard cache files.\n"
                "Does not include .env secrets or private ground-truth files.\n",
            )

            for index, (path, labels) in enumerate(sorted(db_paths.items(), key=lambda item: str(item[0])), start=1):
                entry = {
                    "labels": labels,
                    "source_path": str(path),
                    "exists": path.exists(),
                    "archive_path": None,
                }
                if path.exists():
                    archive_name = f"sqlite/{index:02d}-{_safe_label(labels[0])}-{path.name}"
                    with tempfile.TemporaryDirectory(prefix="ms-vista-snapshot-") as temp_dir:
                        snapshot = Path(temp_dir) / "snapshot.db"
                        _write_sqlite_snapshot(path, snapshot)
                        zf.write(snapshot, archive_name)
                    entry["archive_path"] = archive_name
                manifest["sqlite_databases"].append(entry)

            for label, path_value in extra_files.items():
                path = Path(path_value)
                entry = {
                    "label": label,
                    "source_path": str(path),
                    "exists": path.exists(),
                    "archive_path": None,
                }
                if path.exists() and path.is_file():
                    archive_name = f"files/{_safe_label(label)}{path.suffix}"
                    zf.write(path, archive_name)
                    entry["archive_path"] = archive_name
                manifest["files"].append(entry)

            zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.seek(0)
    except BaseException:
        archive.close()
        raise
    return archive, f"ms-vista-backup-{timestamp}.zip", manifest


def validate_backup_archive(archive: BinaryIO) -> dict:
    """Validate ZIP integrity and every included SQLite snapshot."""
    archive.seek(0)
    sqlite_entries = []
    with zipfile.ZipFile(archive) as zf:
        corrupt_entry = zf.testzip()
        if corrupt_entry:
            raise RuntimeError(f"Backup archive failed CRC validation at {corrupt_entry}.")
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise RuntimeError("Backup archive is missing manifest.json.")
        manifest = json.loads(zf.read("manifest.json"))
        sqlite_entries = [name for name in names if name.startswith("sqlite/")]
        if not sqlite_entries:
            raise RuntimeError("Backup archive does not contain a SQLite database snapshot.")
        with tempfile.TemporaryDirectory(prefix="ms-vista-backup-check-") as temp_dir:
            for index, name in enumerate(sorted(sqlite_entries), start=1):
                db_path = Path(temp_dir) / f"snapshot-{index}.db"
                with zf.open(name) as source, db_path.open("wb") as destination:
                    _copy_stream(source, destination)
                harden_private_file(db_path)
                with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as connection:
                    result = connection.execute("PRAGMA quick_check").fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError(f"SQLite integrity check failed for {name}.")
    archive.seek(0)
    return {
        "manifest": manifest,
        "sqlite_snapshots": len(sqlite_entries),
    }


def write_backup_archive(
    output_dir: Path,
    *,
    retention_count: int,
    now: Optional[datetime] = None,
    database_urls: Optional[Dict[str, str]] = None,
    extra_files: Optional[Dict[str, Path]] = None,
) -> tuple[Path, dict]:
    """Create, validate, atomically persist, and prune a scheduled backup."""
    if retention_count <= 0:
        raise ValueError("retention_count must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(output_dir, 0o700)
    except OSError:
        logger.warning("Could not restrict backup directory permissions: %s", output_dir)
    archive, filename, manifest = create_backup_archive(
        database_urls=database_urls,
        extra_files=extra_files,
        now=now,
    )
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    try:
        with archive:
            validation = validate_backup_archive(archive)
            with open(temporary, "wb") as handle:
                _copy_stream(archive, handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    backups = sorted(
        output_dir.glob("ms-vista-backup-*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_backup in backups[retention_count:]:
        stale_backup.unlink(missing_ok=True)

    return destination, {
        **manifest,
        "validation": {
            "sqlite_snapshots": validation["sqlite_snapshots"],
            "zip_crc": "ok",
            "sqlite_quick_check": "ok",
        },
    }


def backup_locations_separate(primary_dir: Path, mirror_dir: Path) -> Optional[bool]:
    """Return whether two existing backup directories are on different devices."""
    try:
        return os.stat(primary_dir).st_dev != os.stat(mirror_dir).st_dev
    except OSError:
        return None


def mirror_backup_archive(
    source: Path,
    mirror_dir: Path,
    *,
    retention_count: int,
) -> tuple[Path, dict]:
    """Atomically copy and revalidate a retained backup on a second filesystem."""
    if retention_count <= 0:
        raise ValueError("retention_count must be positive")
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Backup archive does not exist: {source}")
    mirror_dir = Path(mirror_dir).expanduser().resolve()
    if mirror_dir == source.parent:
        raise ValueError("Backup mirror directory must differ from the primary backup directory.")
    harden_private_directory(mirror_dir)

    source_digest = _file_sha256(source)
    with source.open("rb") as handle:
        source_validation = validate_backup_archive(handle)
    destination = mirror_dir / source.name
    temporary = mirror_dir / f".{source.name}.tmp"
    try:
        with source.open("rb") as source_handle, open(temporary, "wb") as handle:
            _copy_stream(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        harden_private_file(temporary)
        mirror_digest = _file_sha256(temporary)
        if not hmac.compare_digest(source_digest, mirror_digest):
            raise RuntimeError("Mirrored backup checksum does not match the source archive.")
        with temporary.open("rb") as handle:
            mirror_validation = validate_backup_archive(handle)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    backups = sorted(
        mirror_dir.glob("ms-vista-backup-*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_backup in backups[retention_count:]:
        stale_backup.unlink(missing_ok=True)

    return destination, {
        "sha256": mirror_digest,
        "source_sqlite_snapshots": source_validation["sqlite_snapshots"],
        "mirror_sqlite_snapshots": mirror_validation["sqlite_snapshots"],
        "zip_crc": "ok",
        "sqlite_quick_check": "ok",
    }


def _write_restored_file(destination: Path, source: BinaryIO, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Restore target already exists: {destination}")
    harden_private_directory(destination.parent)
    temporary = destination.parent / f".{destination.name}.tmp"
    try:
        with open(temporary, "wb") as handle:
            _copy_stream(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        harden_private_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_backup_archive(
    archive_path: Path,
    destination_dir: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Validate and safely unpack a backup into an offline recovery directory."""
    archive_path = Path(archive_path).expanduser().resolve()
    destination_dir = Path(destination_dir).expanduser().resolve()
    harden_private_directory(destination_dir)
    restored = []

    with archive_path.open("rb") as archive:
        validation = validate_backup_archive(archive)
    with zipfile.ZipFile(archive_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        selected = []
        selected.extend(
            entry.get("archive_path")
            for entry in manifest.get("sqlite_databases", [])
            if entry.get("archive_path")
        )
        selected.extend(
            entry.get("archive_path")
            for entry in manifest.get("files", [])
            if entry.get("archive_path")
        )
        for archive_name in selected:
            if archive_name not in zf.namelist():
                raise RuntimeError(f"Backup manifest references a missing entry: {archive_name}")
            group = "sqlite" if archive_name.startswith("sqlite/") else "files"
            safe_name = Path(archive_name).name
            target = destination_dir / group / safe_name
            with zf.open(archive_name) as source:
                _write_restored_file(target, source, overwrite=overwrite)
            restored.append(str(target))
        _write_restored_file(
            destination_dir / "manifest.json",
            io.BytesIO(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")),
            overwrite=overwrite,
        )

    return {
        "archive": archive_path.name,
        "destination": str(destination_dir),
        "restored_files": restored,
        "sqlite_snapshots": validation["sqlite_snapshots"],
        "zip_crc": "ok",
        "sqlite_quick_check": "ok",
    }


class BackupScheduler:
    """Single-process scheduler for periodic, retained SQLite backups."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        mirror_dir: Optional[Path] = None,
        require_mirror: bool = False,
        interval_hours: int = 48,
        retention_count: int = 15,
        poll_seconds: int = 300,
        run_on_start: bool = True,
    ) -> None:
        if interval_hours <= 0 or retention_count <= 0 or poll_seconds <= 0:
            raise ValueError("Backup interval, retention, and poll values must be positive.")
        self.enabled = enabled
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.mirror_dir = Path(mirror_dir).expanduser().resolve() if mirror_dir else None
        self.require_mirror = require_mirror
        self.interval = timedelta(hours=interval_hours)
        self.retention_count = retention_count
        self.poll_seconds = poll_seconds
        self.run_on_start = run_on_start
        self.started_at = datetime.now(timezone.utc)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_lock = threading.Lock()
        self._last_success_at: Optional[datetime] = None
        self._last_error_at: Optional[datetime] = None
        self._last_error = ""

    def _backup_files(self) -> list[Path]:
        if not self.output_dir.exists():
            return []
        return sorted(
            self.output_dir.glob("ms-vista-backup-*.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _latest_backup(self) -> Optional[Path]:
        backups = self._backup_files()
        return backups[0] if backups else None

    @contextmanager
    def _process_lock(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".backup.lock"
        with open(lock_path, "a+", encoding="utf-8") as lock_handle:
            harden_private_file(lock_path)
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _is_due(self, now: datetime, latest: Optional[Path]) -> bool:
        if latest is None:
            return self.run_on_start or now - self.started_at >= self.interval
        latest_at = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
        return now - latest_at >= self.interval

    def _mirror_is_current(self, latest: Optional[Path]) -> bool:
        if latest is None:
            return True
        if self.mirror_dir is None:
            return not self.require_mirror
        return (self.mirror_dir / latest.name).is_file()

    def run_if_due(self, *, force: bool = False) -> Optional[Path]:
        if not self.enabled and not force:
            return None
        with self._run_lock:
            with self._process_lock():
                now = datetime.now(timezone.utc)
                latest = self._latest_backup()
                # A failed mirror or validation run must be retried on the next
                # scheduler poll. The primary archive may already exist and look
                # fresh even though the complete backup contract did not finish.
                mirror_current = self._mirror_is_current(latest)
                if (
                    not force
                    and not self._last_error
                    and not self._is_due(now, latest)
                    and mirror_current
                ):
                    return None
                try:
                    destination, _manifest = write_backup_archive(
                        self.output_dir,
                        retention_count=self.retention_count,
                        now=now,
                    )
                    if self.mirror_dir is not None:
                        mirror_destination, _mirror_validation = mirror_backup_archive(
                            destination,
                            self.mirror_dir,
                            retention_count=self.retention_count,
                        )
                        logger.info("Backup mirrored and verified: %s", mirror_destination.name)
                        if self.require_mirror and backup_locations_separate(
                            self.output_dir,
                            self.mirror_dir,
                        ) is not True:
                            raise RuntimeError(
                                "AUTO_BACKUP_MIRROR_DIR must be mounted on a different filesystem."
                            )
                    elif self.require_mirror:
                        raise RuntimeError(
                            "Off-volume backup is required but AUTO_BACKUP_MIRROR_DIR is not configured."
                        )
                    self._last_success_at = now
                    self._last_error_at = None
                    self._last_error = ""
                    logger.info("Scheduled SQLite backup created: %s", destination.name)
                    return destination
                except Exception as exc:
                    self._last_error_at = now
                    self._last_error = str(exc)
                    logger.error("Scheduled SQLite backup failed: %s", exc, exc_info=True)
                    raise

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_if_due()
            except Exception:
                pass
            if self._stop_event.wait(self.poll_seconds):
                break

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.output_dir, 0o700)
        except OSError:
            logger.warning("Could not restrict backup directory permissions: %s", self.output_dir)
        self._thread = threading.Thread(
            target=self._run,
            name="ms-vista-backup-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def status(self, *, include_error: bool = False) -> tuple[str, dict]:
        now = datetime.now(timezone.utc)
        latest = self._latest_backup()
        latest_at = (
            datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)
            if latest is not None
            else None
        )
        age_hours = (
            round((now - latest_at).total_seconds() / 3600, 2)
            if latest_at is not None
            else None
        )
        grace = timedelta(seconds=max(self.poll_seconds * 2, 3600))
        overdue = latest_at is not None and now - latest_at > self.interval + grace
        mirror_files = (
            sorted(
                self.mirror_dir.glob("ms-vista-backup-*.zip"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if self.mirror_dir is not None and self.mirror_dir.exists()
            else []
        )
        latest_mirror = mirror_files[0] if mirror_files else None
        latest_mirror_at = (
            datetime.fromtimestamp(latest_mirror.stat().st_mtime, timezone.utc)
            if latest_mirror is not None
            else None
        )
        mirror_synchronized = bool(
            latest is not None
            and latest_mirror is not None
            and latest.name == latest_mirror.name
        )
        mirror_overdue = (
            latest_mirror_at is not None
            and now - latest_mirror_at > self.interval + grace
        )
        separate_filesystem = (
            backup_locations_separate(self.output_dir, self.mirror_dir)
            if self.mirror_dir is not None
            else None
        )
        if not self.enabled:
            status = "disabled"
        elif self._last_error:
            status = "unhealthy"
        elif self.require_mirror and (
            self.mirror_dir is None
            or latest_mirror is None
            or not mirror_synchronized
            or mirror_overdue
            or separate_filesystem is not True
        ):
            status = "unhealthy"
        elif latest is None:
            status = "pending"
        elif overdue:
            status = "overdue"
        else:
            status = "healthy"
        details = {
            "enabled": self.enabled,
            "status": status,
            "interval_hours": int(self.interval.total_seconds() // 3600),
            "retention_count": self.retention_count,
            "backup_count": len(self._backup_files()),
            "latest_backup": latest.name if latest is not None else None,
            "latest_backup_at": latest_at.isoformat() if latest_at is not None else None,
            "age_hours": age_hours,
            "scheduler_running": bool(self._thread and self._thread.is_alive()),
            "mirror_required": self.require_mirror,
            "mirror_configured": self.mirror_dir is not None,
            "mirror_separate_filesystem": separate_filesystem,
            "mirror_backup_count": len(mirror_files),
            "latest_mirror_backup": latest_mirror.name if latest_mirror is not None else None,
            "latest_mirror_backup_at": latest_mirror_at.isoformat() if latest_mirror_at else None,
            "mirror_synchronized": mirror_synchronized,
            "last_success_at": self._last_success_at.isoformat() if self._last_success_at else None,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }
        if include_error and self._last_error:
            details["last_error"] = self._last_error
        return status, details
