"""
Persistent model registry, per-benchmark quota, and submission audit log.

Every accepted submission is linked to a stable owned model ID and recorded in
SQLite so each benchmark's daily quota survives restarts. Only accepted/scored
submissions count; failed submissions are refunded.

The quota is a rolling 24-hour window (not a calendar day), so a user who submits
their allowance is unblocked exactly 24h after their oldest counted submission.
"""

import logging
import hashlib
import json
import threading
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Boolean,
    Integer,
    Index,
    LargeBinary,
    Text,
    ForeignKey,
    UniqueConstraint,
    and_,
    inspect,
    func,
    or_,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

from constants import (
    SUBMISSION_DAILY_LIMIT_PER_BENCHMARK,
    SUBMISSION_RESERVATION_TIMEOUT_MINUTES,
)
try:
    from config import (
        DB_MAX_OVERFLOW,
        DB_POOL_RECYCLE,
        DB_POOL_SIZE,
        SQLITE_BUSY_TIMEOUT_MS,
        SUBMISSION_DATABASE_URL,
    )
    from sqlite_runtime import configure_sqlite_engine, harden_private_file, sqlite_connect_args
    from schema_migrations import run_schema_migrations
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        DB_MAX_OVERFLOW,
        DB_POOL_RECYCLE,
        DB_POOL_SIZE,
        SQLITE_BUSY_TIMEOUT_MS,
        SUBMISSION_DATABASE_URL,
    )
    from .sqlite_runtime import configure_sqlite_engine, harden_private_file, sqlite_connect_args
    from .schema_migrations import run_schema_migrations

logger = logging.getLogger(__name__)
Base = declarative_base()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

_DB_URL = SUBMISSION_DATABASE_URL
SUBMISSION_SCHEMA_VERSION = 5


def _sqlite_db_path(db_url: str) -> Optional[Path]:
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).expanduser().resolve()


def _engine_kwargs(db_url: str) -> dict:
    url = make_url(db_url)
    if url.drivername.startswith("sqlite"):
        return {"connect_args": sqlite_connect_args(db_url, SQLITE_BUSY_TIMEOUT_MS)}
    return {
        "pool_pre_ping": True,
        "pool_size": DB_POOL_SIZE,
        "max_overflow": DB_MAX_OVERFLOW,
        "pool_recycle": DB_POOL_RECYCLE,
    }


_DB_PATH = _sqlite_db_path(_DB_URL)
_DB_DRIVER = make_url(_DB_URL).drivername
_engine = create_engine(
    _DB_URL,
    echo=False,
    **_engine_kwargs(_DB_URL),
)
configure_sqlite_engine(_engine, _DB_URL, SQLITE_BUSY_TIMEOUT_MS)
_Session = sessionmaker(bind=_engine)


@contextmanager
def _schema_file_lock():
    if _DB_PATH is None or fcntl is None:
        yield
        return
    lock_path = _DB_PATH.with_suffix(_DB_PATH.suffix + ".schema.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        harden_private_file(lock_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

# Serialises the count+insert so two concurrent submissions can't both slip past
# the limit within a single process. (For multi-process/Postgres, switch to a
# SELECT ... FOR UPDATE / transactional guard.)
_lock = threading.Lock()

WINDOW = timedelta(hours=24)
RESERVATION_TIMEOUT = timedelta(minutes=SUBMISSION_RESERVATION_TIMEOUT_MINUTES)
BENCHMARK_TASK_IDS = ("do_you_see_me", "minds_eye", "spatial")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _acquire_quota_lock(session, user_email: str, task_id: str) -> None:
    """Serialize quota count+insert across workers for supported backends."""
    if _DB_DRIVER.startswith("postgresql"):
        # Transaction-scoped per-user lock. hashtext is stable within Postgres
        # and avoids holding a global table lock for all submitters.
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
            {"lock_key": f"submission-quota:{user_email}:{task_id}"},
        )
    elif _DB_DRIVER.startswith("sqlite"):
        # SQLite has no row locks; BEGIN IMMEDIATE obtains a reserved write lock
        # before the count, preventing another process from inserting until this
        # transaction commits or rolls back.
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False, index=True)
    task_id = Column(String(64), nullable=False)
    model_name = Column(String(255), nullable=True)
    model_id = Column(String(64), ForeignKey("registered_models.id"), nullable=True, index=True)
    status = Column(String(16), nullable=False, default="accepted")  # accepted | scored | failed
    request_id = Column(String(64), nullable=True)
    ip = Column(String(64), nullable=True)
    score_submission_id = Column(String(64), nullable=True)
    file_sha256 = Column(String(64), nullable=True)
    row_count = Column(Integer, nullable=True)
    model_meta_json = Column(Text, nullable=True)
    latest_score_json = Column(Text, nullable=True)
    spatial_contract_sha256 = Column(
        String(64),
        ForeignKey("spatial_benchmark_contracts.manifest_sha256"),
        nullable=True,
        index=True,
    )
    moderation_status = Column(String(16), nullable=False, default="visible", index=True)
    moderation_reason = Column(Text, nullable=True)
    moderated_by = Column(String(255), nullable=True)
    moderated_at = Column(DateTime, nullable=True)
    rescored_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


Index("ix_sub_user_created", Submission.user_email, Submission.created_at)
Index(
    "ux_submissions_score_submission_id",
    Submission.score_submission_id,
    unique=True,
)


class RegisteredModel(Base):
    """Canonical model identity shared by all benchmark submissions."""

    __tablename__ = "registered_models"

    id = Column(String(64), primary_key=True)
    owner_email = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, unique=True, index=True)
    organization = Column(String(200), nullable=False, default="")
    access = Column(String(80), nullable=False, default="")
    parameter_count = Column(String(80), nullable=True)
    base_model = Column(String(200), nullable=False, default="")
    training_data = Column(Text, nullable=False, default="")
    paper_url = Column(String(500), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow)



class SubmissionAnswer(Base):
    __tablename__ = "submission_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    row_index = Column(Integer, nullable=False)
    line_number = Column(Integer, nullable=True)
    question_id = Column(String(255), nullable=False)
    condition = Column(String(32), nullable=False, default="standard")
    raw_answer_text = Column(Text, nullable=False, default="")
    answer_sha256 = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("submission_id", "condition", "question_id", name="uq_submission_answer_key"),
        Index("ix_submission_answers_export", "submission_id", "row_index"),
    )


class SpatialBenchmarkContract(Base):
    """Immutable public Spatial contract retained for reproducible rescoring."""

    __tablename__ = "spatial_benchmark_contracts"

    manifest_sha256 = Column(String(64), primary_key=True)
    benchmark_version = Column(String(128), nullable=False)
    manifest_content = Column(LargeBinary, nullable=False)
    template_content = Column(LargeBinary, nullable=False)
    questions_content = Column(LargeBinary, nullable=False)
    template_sha256 = Column(String(64), nullable=False)
    questions_sha256 = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class SubmissionArtifact(Base):
    """Exact public evidence artifacts retained for a scored submission."""

    __tablename__ = "submission_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(
        Integer,
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_name = Column(String(255), nullable=False)
    media_type = Column(String(128), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    sha256 = Column(String(64), nullable=False)
    content = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "submission_id",
            "artifact_name",
            name="uq_submission_artifact_name",
        ),
        Index("ix_submission_artifacts_public", "submission_id", "artifact_name"),
    )


def init_db() -> None:
    migrations = [
        (1, _ensure_submission_columns),
        (2, _ensure_model_registry),
        (3, _ensure_submission_integrity_indexes),
        (4, _ensure_submission_artifacts),
        (5, _ensure_spatial_contracts),
    ]
    if _DB_DRIVER.startswith("postgresql"):
        with _engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('submission-schema-init'))"))
            Base.metadata.create_all(connection)
            run_schema_migrations(connection, "submissions", migrations)
    else:
        with _schema_file_lock():
            Base.metadata.create_all(_engine)
            with _engine.begin() as connection:
                run_schema_migrations(connection, "submissions", migrations)
    expire_stale_submission_reservations()


def _ensure_submission_columns(connection) -> None:
    """Add non-destructive columns for existing deployments.

    SQLAlchemy's create_all creates new tables but does not migrate existing
    tables. These nullable columns let old quota databases keep working while
    new submissions can be linked to stored answer rows.
    """
    existing = {col["name"] for col in inspect(connection).get_columns("submissions")}
    datetime_type = "TIMESTAMP" if _DB_DRIVER.startswith("postgresql") else "DATETIME"
    additions = {
        "score_submission_id": "VARCHAR(64)",
        "file_sha256": "VARCHAR(64)",
        "row_count": "INTEGER",
        "model_meta_json": "TEXT",
        "latest_score_json": "TEXT",
        "moderation_status": "VARCHAR(16) DEFAULT 'visible'",
        "moderation_reason": "TEXT",
        "moderated_by": "VARCHAR(255)",
        "moderated_at": datetime_type,
        "rescored_at": datetime_type,
    }
    for column, sql_type in additions.items():
        if column not in existing:
            connection.execute(text(f"ALTER TABLE submissions ADD COLUMN {column} {sql_type}"))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_submissions_score_submission_id "
        "ON submissions (score_submission_id)"
    ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_submissions_moderation_status "
        "ON submissions (moderation_status)"
    ))


def normalize_model_name(value: str) -> str:
    """Canonical comparison key for model-name ownership and collision checks."""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.split()).casefold()


def _canonical_model_meta(raw_value: Optional[str]) -> Dict:
    meta = _json_loads(raw_value)
    return {
        "organization": str(meta.get("organization") or meta.get("org") or "").strip(),
        "access": str(meta.get("access") or meta.get("type") or "").strip(),
        "parameter_count": str(meta.get("parameter_count") or "").strip() or None,
        "base_model": str(meta.get("base_model") or "").strip(),
        "training_data": str(meta.get("training_data") or "").strip(),
        "paper_url": str(meta.get("paper_url") or "").strip() or None,
    }


def _ensure_model_registry(connection) -> None:
    """Add model links and backfill one owned identity per legacy name."""
    existing = {col["name"] for col in inspect(connection).get_columns("submissions")}
    if "model_id" not in existing:
        connection.execute(text("ALTER TABLE submissions ADD COLUMN model_id VARCHAR(64)"))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_submissions_model_id ON submissions (model_id)"
    ))

    rows = connection.execute(text(
        "SELECT id, user_email, model_name, model_meta_json, created_at "
        "FROM submissions WHERE model_name IS NOT NULL AND TRIM(model_name) <> '' "
        "ORDER BY created_at DESC, id DESC"
    )).mappings().all()
    model_ids: Dict[tuple[str, str], str] = {}
    now = _utcnow()
    database_now = now if _DB_DRIVER.startswith("postgresql") else now.isoformat()
    for row in rows:
        owner_email = str(row["user_email"] or "").strip().lower()
        display_name = " ".join(str(row["model_name"] or "").split())
        normalized_name = normalize_model_name(display_name)
        if not owner_email or not normalized_name:
            continue
        key = (owner_email, normalized_name)
        model_id = model_ids.get(key)
        if model_id is None:
            existing_model = connection.execute(text(
                "SELECT id, owner_email FROM registered_models "
                "WHERE normalized_name = :normalized_name"
            ), {
                "normalized_name": normalized_name,
            }).mappings().first()
            if existing_model and str(existing_model["owner_email"]).lower() != owner_email:
                raise RuntimeError(
                    f"Cannot migrate model name '{display_name}' because multiple accounts claim it"
                )
            model_id = str(existing_model["id"]) if existing_model else f"mdl_{uuid.uuid4().hex}"
            if not existing_model:
                meta = _canonical_model_meta(row["model_meta_json"])
                created_at = row["created_at"] or database_now
                if not _DB_DRIVER.startswith("postgresql") and isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                connection.execute(text(
                    "INSERT INTO registered_models ("
                    "id, owner_email, display_name, normalized_name, organization, access, "
                    "parameter_count, base_model, training_data, paper_url, active, created_at, updated_at"
                    ") VALUES ("
                    ":id, :owner_email, :display_name, :normalized_name, :organization, :access, "
                    ":parameter_count, :base_model, :training_data, :paper_url, :active, :created_at, :updated_at"
                    ")"
                ), {
                    "id": model_id,
                    "owner_email": owner_email,
                    "display_name": display_name,
                    "normalized_name": normalized_name,
                    **meta,
                    "active": True,
                    "created_at": created_at,
                    "updated_at": database_now,
                })
            model_ids[key] = model_id
        connection.execute(text(
            "UPDATE submissions SET model_id = :model_id WHERE id = :submission_id"
        ), {"model_id": model_id, "submission_id": row["id"]})


def _ensure_submission_integrity_indexes(connection) -> None:
    """Enforce globally unique public submission identifiers."""
    connection.execute(text("DROP INDEX IF EXISTS ix_submissions_score_submission_id"))
    connection.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_submissions_score_submission_id "
        "ON submissions (score_submission_id)"
    ))


def _ensure_submission_artifacts(connection) -> None:
    """Create exact evidence storage for installations upgrading in place."""
    SubmissionArtifact.__table__.create(bind=connection, checkfirst=True)


def _ensure_spatial_contracts(connection) -> None:
    """Retain immutable benchmark contracts and link Spatial submissions to them."""
    SpatialBenchmarkContract.__table__.create(bind=connection, checkfirst=True)
    existing = {col["name"] for col in inspect(connection).get_columns("submissions")}
    if "spatial_contract_sha256" not in existing:
        connection.execute(text(
            "ALTER TABLE submissions ADD COLUMN spatial_contract_sha256 VARCHAR(64)"
        ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_submissions_spatial_contract_sha256 "
        "ON submissions (spatial_contract_sha256)"
    ))


@dataclass
class QuotaResult:
    allowed: bool
    used: int
    limit: int
    remaining: int
    reset_at: Optional[datetime]
    retry_after: int
    submission_id: Optional[int] = None


def _counted_query(
    session,
    user_email: str,
    task_id: str,
    window_start: datetime,
    now: datetime,
):
    return (
        session.query(Submission)
        .filter(
            Submission.user_email == user_email,
            Submission.task_id == task_id,
            Submission.created_at >= window_start,
            or_(
                Submission.status == "scored",
                and_(
                    Submission.status == "accepted",
                    Submission.created_at >= now - RESERVATION_TIMEOUT,
                ),
            ),
        )
        .order_by(Submission.created_at.asc())
    )


def _quota_expiry(row: Submission) -> datetime:
    created_at = _aware(row.created_at) or _utcnow()
    return created_at + (
        RESERVATION_TIMEOUT if row.status == "accepted" else WINDOW
    )


def expire_stale_submission_reservations(
    *,
    user_email: Optional[str] = None,
    task_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Fail abandoned in-flight reservations so they no longer consume quota."""
    now = now or _utcnow()
    with _Session() as session:
        query = session.query(Submission).filter(
            Submission.status == "accepted",
            Submission.created_at < now - RESERVATION_TIMEOUT,
        )
        if user_email:
            query = query.filter(Submission.user_email == user_email)
        if task_id:
            query = query.filter(Submission.task_id == task_id)
        expired = query.update(
            {Submission.status: "failed"},
            synchronize_session=False,
        )
        session.commit()
        return int(expired or 0)


def try_consume_quota(
    user_email: str,
    task_id: str,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip: Optional[str] = None,
    limit: Optional[int] = None,
) -> QuotaResult:
    """Reserve one daily submission slot. Inserts an 'accepted' row if allowed."""
    limit = SUBMISSION_DAILY_LIMIT_PER_BENCHMARK if limit is None else limit
    if limit <= 0:
        raise ValueError("Submission quota limit must be positive")
    now = _utcnow()
    window_start = now - WINDOW
    with _lock, _Session() as session:
        _acquire_quota_lock(session, user_email, task_id)
        session.query(Submission).filter(
            Submission.user_email == user_email,
            Submission.task_id == task_id,
            Submission.status == "accepted",
            Submission.created_at < now - RESERVATION_TIMEOUT,
        ).update(
            {Submission.status: "failed"},
            synchronize_session=False,
        )
        rows = _counted_query(session, user_email, task_id, window_start, now).all()
        used = len(rows)
        if used >= limit:
            reset_at = min(_quota_expiry(row) for row in rows)
            retry_after = max(1, int((reset_at - now).total_seconds()))
            return QuotaResult(False, used, limit, 0, reset_at, retry_after)
        row = Submission(
            user_email=user_email,
            task_id=task_id,
            model_name=model_name,
            model_id=model_id,
            status="accepted",
            request_id=request_id,
            ip=ip,
            created_at=now,
        )
        session.add(row)
        session.commit()
        used_after = used + 1
        reset_at = min(
            (_quota_expiry(existing) for existing in [*rows, row]),
            default=now + WINDOW,
        )
        return QuotaResult(
            True, used_after, limit, max(0, limit - used_after), reset_at, 0, row.id
        )


def finalize_submission(submission_id: Optional[int], success: bool) -> None:
    """Mark a reserved submission as scored (counts) or failed (refunded)."""
    if not submission_id:
        return
    with _Session() as session:
        row = session.get(Submission, submission_id)
        if row is not None:
            row.status = "scored" if success else "failed"
            session.commit()


def _json_dumps(value: Optional[Dict]) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: Optional[str]) -> Dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def submission_integrity_status() -> Dict:
    """Return logical database invariants that SQLite quick_check cannot see."""
    with _Session() as session:
        scored = session.query(Submission).filter(Submission.status == "scored").all()
        models = {
            row.id: row
            for row in session.query(RegisteredModel).all()
        }
        answer_counts = {
            submission_id: count
            for submission_id, count in (
                session.query(
                    SubmissionAnswer.submission_id,
                    func.count(SubmissionAnswer.id),
                )
                .group_by(SubmissionAnswer.submission_id)
                .all()
            )
        }
        artifact_metadata: Dict[int, Dict[str, tuple[int, str]]] = {}
        for submission_id, artifact_name, size_bytes, sha256 in session.query(
            SubmissionArtifact.submission_id,
            SubmissionArtifact.artifact_name,
            SubmissionArtifact.size_bytes,
            SubmissionArtifact.sha256,
        ).all():
            artifact_metadata.setdefault(submission_id, {})[artifact_name] = (
                size_bytes,
                sha256,
            )
        contracts = {
            contract.manifest_sha256: contract
            for contract in session.query(SpatialBenchmarkContract).all()
        }

    invalid_contracts = set()
    for contract_sha256, contract in contracts.items():
        manifest_content = bytes(contract.manifest_content or b"")
        template_content = bytes(contract.template_content or b"")
        questions_content = bytes(contract.questions_content or b"")
        if (
            hashlib.sha256(manifest_content).hexdigest() != contract_sha256
            or hashlib.sha256(template_content).hexdigest() != contract.template_sha256
            or hashlib.sha256(questions_content).hexdigest() != contract.questions_sha256
            or not str(contract.benchmark_version or "").strip()
        ):
            invalid_contracts.add(contract_sha256)

    score_ids = Counter(
        str(row.score_submission_id).strip()
        for row in scored
        if str(row.score_submission_id or "").strip()
    )
    counts = {
        "missing_model_id_count": 0,
        "unknown_model_id_count": 0,
        "model_owner_mismatch_count": 0,
        "missing_score_submission_id_count": 0,
        "duplicate_score_submission_id_count": sum(
            1 for count in score_ids.values() if count > 1
        ),
        "missing_score_payload_count": 0,
        "malformed_score_payload_count": 0,
        "answer_row_count_mismatch_count": 0,
        "spatial_evidence_artifact_mismatch_count": 0,
        "missing_spatial_contract_count": 0,
        "invalid_spatial_contract_count": 0,
        "invalid_artifact_metadata_count": 0,
        "invalid_task_count": 0,
    }
    for row in scored:
        if not row.model_id:
            counts["missing_model_id_count"] += 1
            model = None
        else:
            model = models.get(row.model_id)
            if model is None:
                counts["unknown_model_id_count"] += 1
        if model is not None and str(model.owner_email).strip().lower() != str(row.user_email).strip().lower():
            counts["model_owner_mismatch_count"] += 1
        if not str(row.score_submission_id or "").strip():
            counts["missing_score_submission_id_count"] += 1
        if not row.latest_score_json:
            counts["missing_score_payload_count"] += 1
        else:
            try:
                payload = json.loads(row.latest_score_json)
                if not isinstance(payload, dict) or payload.get("accuracy") is None:
                    counts["malformed_score_payload_count"] += 1
            except (TypeError, json.JSONDecodeError):
                counts["malformed_score_payload_count"] += 1
        expected_answers = row.row_count
        actual_answers = int(answer_counts.get(row.id, 0))
        if expected_answers is None or int(expected_answers) != actual_answers:
            counts["answer_row_count_mismatch_count"] += 1
        if row.task_id not in BENCHMARK_TASK_IDS:
            counts["invalid_task_count"] += 1
        artifacts = artifact_metadata.get(row.id, {})
        for size_bytes, sha256 in artifacts.values():
            if (
                not isinstance(size_bytes, int)
                or size_bytes <= 0
                or len(str(sha256 or "")) != 64
            ):
                counts["invalid_artifact_metadata_count"] += 1
        score_payload = _json_loads(row.latest_score_json)
        score_metadata = score_payload.get("metadata")
        public_evidence = (
            score_metadata.get("public_evidence")
            if isinstance(score_metadata, dict)
            else None
        )
        if not isinstance(public_evidence, dict):
            public_evidence = {}
        if row.task_id == "spatial" and public_evidence.get("available") is True:
            required_artifacts = {
                "spatial_reasoning_submission.zip",
                "submission.jsonl",
                "run_manifest.json",
                "leaderboard.json",
            }
            archive_metadata = artifacts.get("spatial_reasoning_submission.zip")
            if (
                set(artifacts) != required_artifacts
                or archive_metadata is None
                or archive_metadata[1] != row.file_sha256
            ):
                counts["spatial_evidence_artifact_mismatch_count"] += 1
            if not row.spatial_contract_sha256 or row.spatial_contract_sha256 not in contracts:
                counts["missing_spatial_contract_count"] += 1
            elif row.spatial_contract_sha256 in invalid_contracts:
                counts["invalid_spatial_contract_count"] += 1

    issue_count = sum(counts.values())
    return {
        "healthy": issue_count == 0,
        "scored_submission_count": len(scored),
        "stored_answer_count": sum(answer_counts.values()),
        "stored_artifact_count": sum(len(values) for values in artifact_metadata.values()),
        "issue_count": issue_count,
        **counts,
    }


class ModelNameConflictError(ValueError):
    """Raised when a canonical model name is already owned by another record."""


def _registered_model_meta(row: RegisteredModel) -> Dict:
    meta = {
        "organization": row.organization or "",
        "org": row.organization or "",
        "access": row.access or "",
        "type": row.access or "",
        "base_model": row.base_model or "",
        "training_data": row.training_data or "",
        "paper_url": row.paper_url or "",
    }
    if row.parameter_count:
        meta["parameter_count"] = row.parameter_count
    return meta


def _registered_model_summary(row: RegisteredModel, benchmarks: Optional[Dict] = None) -> Dict:
    created_at = _aware(row.created_at)
    updated_at = _aware(row.updated_at)
    return {
        "model_id": row.id,
        "model_name": row.display_name,
        "owner_email": row.owner_email,
        "organization": row.organization or "",
        "access": row.access or "",
        "parameter_count": row.parameter_count,
        "base_model": row.base_model or "",
        "training_data": row.training_data or "",
        "paper_url": row.paper_url,
        "active": bool(row.active),
        "created_at": created_at.isoformat() if created_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
        "model_meta": _registered_model_meta(row),
        "benchmarks": benchmarks or {},
    }


def create_registered_model(
    owner_email: str,
    display_name: str,
    model_meta: Dict,
) -> Dict:
    """Register a globally unique model name under one account."""
    owner_email = str(owner_email or "").strip().lower()
    display_name = " ".join(str(display_name or "").split())
    normalized_name = normalize_model_name(display_name)
    if not owner_email or not normalized_name:
        raise ValueError("Owner email and model name are required")
    now = _utcnow()
    with _lock, _Session() as session:
        existing = session.query(RegisteredModel).filter(
            RegisteredModel.normalized_name == normalized_name,
        ).first()
        if existing is not None:
            raise ModelNameConflictError(existing.id)
        row = RegisteredModel(
            id=f"mdl_{uuid.uuid4().hex}",
            owner_email=owner_email,
            display_name=display_name,
            normalized_name=normalized_name,
            organization=str(model_meta.get("organization") or "").strip(),
            access=str(model_meta.get("access") or "").strip(),
            parameter_count=str(model_meta.get("parameter_count") or "").strip() or None,
            base_model=str(model_meta.get("base_model") or "").strip(),
            training_data=str(model_meta.get("training_data") or "").strip(),
            paper_url=str(model_meta.get("paper_url") or "").strip() or None,
            active=True,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ModelNameConflictError("model_name_conflict") from exc
        session.refresh(row)
        return _registered_model_summary(row)


def get_owned_model(model_id: str, owner_email: str) -> Optional[Dict]:
    """Return an active registered model only when the account owns it."""
    owner_email = str(owner_email or "").strip().lower()
    with _Session() as session:
        row = session.query(RegisteredModel).filter(
            RegisteredModel.id == str(model_id or "").strip(),
            RegisteredModel.owner_email == owner_email,
            RegisteredModel.active.is_(True),
        ).one_or_none()
        return _registered_model_summary(row) if row is not None else None


def find_owned_model_by_name(display_name: str, owner_email: str) -> Optional[Dict]:
    """Compatibility lookup for trusted import and development submission paths."""
    owner_email = str(owner_email or "").strip().lower()
    normalized_name = normalize_model_name(display_name)
    with _Session() as session:
        row = session.query(RegisteredModel).filter(
            RegisteredModel.owner_email == owner_email,
            RegisteredModel.normalized_name == normalized_name,
            RegisteredModel.active.is_(True),
        ).one_or_none()
        return _registered_model_summary(row) if row is not None else None


def list_registered_models(owner_email: Optional[str] = None) -> List[Dict]:
    """List model records, with the latest visible score for every benchmark."""
    normalized_owner = str(owner_email or "").strip().lower()
    with _Session() as session:
        query = session.query(RegisteredModel).filter(RegisteredModel.active.is_(True))
        if normalized_owner:
            query = query.filter(RegisteredModel.owner_email == normalized_owner)
        models = query.order_by(
            RegisteredModel.display_name.asc(),
            RegisteredModel.created_at.asc(),
        ).all()
        if not models:
            return []
        ids = [row.id for row in models]
        submissions = session.query(Submission).filter(
            Submission.model_id.in_(ids),
            Submission.status == "scored",
            Submission.moderation_status != "deleted",
        ).order_by(Submission.created_at.desc(), Submission.id.desc()).all()
        by_model: Dict[str, Dict[str, Dict]] = {model_id: {} for model_id in ids}
        for submission in submissions:
            task_rows = by_model.setdefault(submission.model_id, {})
            if submission.task_id not in task_rows:
                task_rows[submission.task_id] = _submission_summary(submission)
        return [
            _registered_model_summary(row, by_model.get(row.id, {}))
            for row in models
        ]


def archive_registered_model(model_id: str, owner_email: str) -> Optional[Dict]:
    """Archive an owned model after all of its submissions are deleted.

    Historical rows remain available for audit, while the model is removed from
    registration controls and public model listings.
    """
    normalized_model_id = str(model_id or "").strip()
    normalized_owner = str(owner_email or "").strip().lower()
    if not normalized_model_id or not normalized_owner:
        return None
    with _lock, _Session() as session:
        row = session.query(RegisteredModel).filter(
            RegisteredModel.id == normalized_model_id,
            RegisteredModel.owner_email == normalized_owner,
            RegisteredModel.active.is_(True),
        ).one_or_none()
        if row is None:
            return None
        retained_submission = session.query(Submission.id).filter(
            Submission.model_id == normalized_model_id,
            Submission.moderation_status != "deleted",
        ).first()
        if retained_submission is not None:
            raise ValueError(
                "Delete every submission for this model before archiving it."
            )
        row.active = False
        row.updated_at = _utcnow()
        session.commit()
        session.refresh(row)
        return _registered_model_summary(row)


def _submission_summary(row: Submission) -> Dict:
    score = _json_loads(row.latest_score_json)
    created_at = _aware(row.created_at)
    moderated_at = _aware(row.moderated_at)
    rescored_at = _aware(row.rescored_at)
    return {
        "submission_id": row.score_submission_id,
        "model_id": row.model_id,
        "task_id": row.task_id,
        "model_name": row.model_name,
        "user_email": row.user_email,
        "status": row.status,
        "moderation_status": row.moderation_status or "visible",
        "moderation_reason": row.moderation_reason,
        "moderated_by": row.moderated_by,
        "moderated_at": moderated_at.isoformat() if moderated_at else None,
        "rescored_at": rescored_at.isoformat() if rescored_at else None,
        "file_sha256": row.file_sha256,
        "row_count": row.row_count,
        "spatial_contract_sha256": row.spatial_contract_sha256,
        "created_at": created_at.isoformat() if created_at else None,
        "accuracy": score.get("accuracy"),
        "micro_accuracy": score.get("micro_accuracy", score.get("accuracy")),
        "macro_accuracy": score.get("macro_accuracy"),
        "task_spread": score.get("task_spread", score.get("accuracy_std")),
        "accuracy_std": score.get("accuracy_std"),
        "score_method": score.get("score_method"),
        "total_samples": score.get("total_samples"),
        "correct_samples": score.get("correct_samples"),
        "diagnostics": score.get("diagnostics"),
        "grading": score.get("grading"),
        "metadata": score.get("metadata", {}),
        "model_meta": _json_loads(row.model_meta_json),
        "submission_export_url": (
            f"/api/submissions/{row.score_submission_id}/export.jsonl"
            if row.score_submission_id and row.moderation_status != "deleted" else None
        ),
        "public_evidence_url": (
            f"/api/public/submissions/{row.score_submission_id}/evidence"
            if (
                row.task_id == "spatial"
                and row.score_submission_id
                and row.status == "scored"
                and row.moderation_status == "visible"
            )
            else None
        ),
    }


def _answer_hash(answer: str) -> str:
    return hashlib.sha256(str(answer or "").encode("utf-8")).hexdigest()


def store_submission_answers(
    submission_id: Optional[int],
    score_submission_id: str,
    file_sha256: str,
    records: Iterable[Dict],
    model_meta: Optional[Dict] = None,
    score_json: Optional[Dict] = None,
    artifacts: Optional[Iterable[Dict]] = None,
    spatial_contract: Optional[Dict] = None,
) -> None:
    """Persist normalized answers, public evidence, and its immutable contract."""
    if not submission_id:
        return
    normalized_records = list(records)
    with _Session() as session:
        row = session.get(Submission, submission_id)
        if row is None:
            raise ValueError(f"Unknown reserved submission id: {submission_id}")
        row.score_submission_id = score_submission_id
        row.file_sha256 = file_sha256
        row.row_count = len(normalized_records)
        if model_meta is not None:
            row.model_meta_json = _json_dumps(model_meta)
        if score_json is not None:
            row.latest_score_json = _json_dumps(score_json)
        if spatial_contract is not None:
            if row.task_id != "spatial":
                raise ValueError("A Spatial contract cannot be attached to another benchmark")
            contract_values = {
                key: spatial_contract.get(key)
                for key in ("manifest", "template", "questions")
            }
            if any(not isinstance(value, bytes) or not value for value in contract_values.values()):
                raise ValueError("Spatial contract files must be non-empty bytes")
            manifest_sha256 = hashlib.sha256(contract_values["manifest"]).hexdigest()
            declared_sha256 = str(spatial_contract.get("manifest_sha256") or "")
            if declared_sha256 and declared_sha256 != manifest_sha256:
                raise ValueError("Spatial contract manifest hash does not match its content")
            try:
                manifest = json.loads(contract_values["manifest"].decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Spatial contract manifest is not valid UTF-8 JSON") from exc
            benchmark_version = str(manifest.get("benchmark_version") or "").strip()
            artifact_metadata = manifest.get("artifacts") or {}
            template_sha256 = hashlib.sha256(contract_values["template"]).hexdigest()
            questions_sha256 = hashlib.sha256(contract_values["questions"]).hexdigest()
            if not benchmark_version:
                raise ValueError("Spatial contract manifest is missing benchmark_version")
            if (
                (artifact_metadata.get("submission_template") or {}).get("sha256")
                != template_sha256
                or (artifact_metadata.get("questions") or {}).get("sha256")
                != questions_sha256
            ):
                raise ValueError("Spatial contract files do not match the manifest")
            existing_contract = session.get(SpatialBenchmarkContract, manifest_sha256)
            if existing_contract is None:
                session.add(SpatialBenchmarkContract(
                    manifest_sha256=manifest_sha256,
                    benchmark_version=benchmark_version,
                    manifest_content=contract_values["manifest"],
                    template_content=contract_values["template"],
                    questions_content=contract_values["questions"],
                    template_sha256=template_sha256,
                    questions_sha256=questions_sha256,
                    created_at=_utcnow(),
                ))
            elif (
                bytes(existing_contract.manifest_content) != contract_values["manifest"]
                or bytes(existing_contract.template_content) != contract_values["template"]
                or bytes(existing_contract.questions_content) != contract_values["questions"]
                or existing_contract.benchmark_version != benchmark_version
            ):
                raise ValueError("Stored Spatial contract conflicts with the submitted contract")
            row.spatial_contract_sha256 = manifest_sha256
        elif row.task_id == "spatial" and artifacts:
            raise ValueError("Spatial evidence cannot be stored without its benchmark contract")
        session.query(SubmissionAnswer).filter(
            SubmissionAnswer.submission_id == submission_id
        ).delete(synchronize_session=False)
        session.query(SubmissionArtifact).filter(
            SubmissionArtifact.submission_id == submission_id
        ).delete(synchronize_session=False)
        created_at = _utcnow()
        answer_rows = []
        for rec in normalized_records:
            answer = str(rec.get("answer") or "")
            answer_rows.append({
                "submission_id": submission_id,
                "row_index": int(rec.get("row_index") or 0),
                "line_number": int(rec["line_number"]) if rec.get("line_number") else None,
                "question_id": str(rec.get("question_id") or ""),
                "condition": str(rec.get("condition") or "standard"),
                "raw_answer_text": answer,
                "answer_sha256": _answer_hash(answer),
                "created_at": created_at,
            })
        if answer_rows:
            session.bulk_insert_mappings(SubmissionAnswer, answer_rows)
        artifact_rows = []
        seen_artifact_names = set()
        for artifact in artifacts or []:
            artifact_name = str(artifact.get("artifact_name") or "").strip()
            media_type = str(artifact.get("media_type") or "").strip()
            content = artifact.get("content")
            if (
                not artifact_name
                or len(artifact_name) > 255
                or Path(artifact_name).name != artifact_name
                or artifact_name in seen_artifact_names
                or not media_type
                or len(media_type) > 128
                or not isinstance(content, bytes)
                or not content
            ):
                raise ValueError("Invalid or duplicate submission artifact")
            seen_artifact_names.add(artifact_name)
            artifact_rows.append({
                "submission_id": submission_id,
                "artifact_name": artifact_name,
                "media_type": media_type,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content": content,
                "created_at": created_at,
            })
        if artifact_rows:
            session.bulk_insert_mappings(SubmissionArtifact, artifact_rows)
        session.commit()


def update_submission_score(
    score_submission_id: str,
    score_json: Dict,
    model_meta: Optional[Dict] = None,
) -> None:
    """Persist the latest scoring snapshot for an existing submission."""
    with _Session() as session:
        row = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id)
        ).one_or_none()
        if row is None:
            raise ValueError(f"Unknown submission: {score_submission_id}")
        row.latest_score_json = _json_dumps(score_json)
        if model_meta is not None:
            row.model_meta_json = _json_dumps(model_meta)
        row.rescored_at = _utcnow()
        session.commit()


def get_submission_export(
    score_submission_id: str,
    user_email: Optional[str] = None,
) -> Optional[Dict]:
    """Return metadata + stored final-answer rows for a scored submission."""
    with _Session() as session:
        query = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id),
            Submission.status == "scored",
            Submission.moderation_status != "deleted",
        )
        if user_email:
            query = query.filter(Submission.user_email == user_email)
        submission = query.one_or_none()
        if submission is None:
            return None
        answers = (
            session.query(SubmissionAnswer)
            .filter(SubmissionAnswer.submission_id == submission.id)
            .order_by(SubmissionAnswer.row_index.asc(), SubmissionAnswer.id.asc())
            .all()
        )
        rows: List[Dict[str, str]] = [
            {
                "question_id": answer.question_id,
                "condition": answer.condition,
                "answer": answer.raw_answer_text,
            }
            for answer in answers
        ]
        return {
            "submission_id": submission.score_submission_id,
            "model_id": submission.model_id,
            "task_id": submission.task_id,
            "model_name": submission.model_name,
            "file_sha256": submission.file_sha256,
            "row_count": submission.row_count,
            "created_at": _aware(submission.created_at).isoformat()
            if submission.created_at else None,
            "rows": rows,
        }


def get_public_spatial_evidence(score_submission_id: str) -> Optional[Dict]:
    """Return public, non-owner evidence metadata for one visible spatial run."""
    with _Session() as session:
        submission = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id),
            Submission.task_id == "spatial",
            Submission.status == "scored",
            Submission.moderation_status == "visible",
        ).one_or_none()
        if submission is None:
            return None
        artifacts = (
            session.query(SubmissionArtifact)
            .filter(SubmissionArtifact.submission_id == submission.id)
            .order_by(SubmissionArtifact.artifact_name.asc())
            .all()
        )
        score = _json_loads(submission.latest_score_json)
        base_url = f"/api/public/submissions/{submission.score_submission_id}"
        return {
            "submission_id": submission.score_submission_id,
            "model_id": submission.model_id,
            "model_name": submission.model_name,
            "task_id": submission.task_id,
            "created_at": _aware(submission.created_at).isoformat()
            if submission.created_at
            else None,
            "archive_sha256": submission.file_sha256,
            "row_count": submission.row_count,
            "score": score,
            "verification": {
                "level": "provenance_and_arithmetic",
                "server_ground_truth_evaluation": False,
                "description": (
                    "The server validates the official harness version, package hashes, public sample coverage, "
                    "and agreement between per-sample correctness flags and aggregate scores. It does not "
                    "independently compare spatial answers with private ground truth."
                ),
            },
            "answers_url": f"{base_url}/answers.jsonl",
            "artifacts": [
                {
                    "name": artifact.artifact_name,
                    "media_type": artifact.media_type,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                    "url": f"{base_url}/artifacts/{artifact.artifact_name}",
                }
                for artifact in artifacts
            ],
        }


def get_public_spatial_artifact(
    score_submission_id: str,
    artifact_name: str,
) -> Optional[Dict]:
    """Read one exact artifact only when its spatial submission is public."""
    with _Session() as session:
        row = (
            session.query(SubmissionArtifact, Submission)
            .join(Submission, Submission.id == SubmissionArtifact.submission_id)
            .filter(
                Submission.score_submission_id == str(score_submission_id),
                Submission.task_id == "spatial",
                Submission.status == "scored",
                Submission.moderation_status == "visible",
                SubmissionArtifact.artifact_name == str(artifact_name),
            )
            .one_or_none()
        )
        if row is None:
            return None
        artifact, submission = row
        return {
            "submission_id": submission.score_submission_id,
            "artifact_name": artifact.artifact_name,
            "media_type": artifact.media_type,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "content": bytes(artifact.content),
        }


def get_submission_for_rescore(score_submission_id: str) -> Optional[Dict]:
    """Return stored answers and metadata for admin re-scoring."""
    with _Session() as session:
        submission = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id),
            Submission.status == "scored",
        ).one_or_none()
        if submission is None:
            return None
        answers = (
            session.query(SubmissionAnswer)
            .filter(SubmissionAnswer.submission_id == submission.id)
            .order_by(SubmissionAnswer.row_index.asc(), SubmissionAnswer.id.asc())
            .all()
        )
        artifacts = []
        spatial_contract = None
        if submission.task_id == "spatial":
            artifacts = (
                session.query(SubmissionArtifact)
                .filter(SubmissionArtifact.submission_id == submission.id)
                .order_by(SubmissionArtifact.artifact_name.asc())
                .all()
            )
            if submission.spatial_contract_sha256:
                contract = session.get(
                    SpatialBenchmarkContract,
                    submission.spatial_contract_sha256,
                )
                if contract is not None:
                    spatial_contract = {
                        "manifest_sha256": contract.manifest_sha256,
                        "benchmark_version": contract.benchmark_version,
                        "manifest": bytes(contract.manifest_content),
                        "template": bytes(contract.template_content),
                        "questions": bytes(contract.questions_content),
                    }
        created_at = _aware(submission.created_at)
        return {
            **_submission_summary(submission),
            "created_at": created_at.isoformat() if created_at else None,
            "answers": [
                {
                    "question_id": answer.question_id,
                    "condition": answer.condition,
                    "answer": answer.raw_answer_text,
                }
                for answer in answers
            ],
            "artifacts": {
                artifact.artifact_name: bytes(artifact.content)
                for artifact in artifacts
            },
            "spatial_contract": spatial_contract,
        }


def list_submissions(
    user_email: Optional[str] = None,
    limit: int = 100,
    include_failed: bool = True,
    include_hidden: bool = True,
    include_deleted: bool = True,
) -> List[Dict]:
    """List submissions for a user or admin view."""
    limit = max(1, min(int(limit or 100), 500))
    with _Session() as session:
        query = session.query(Submission)
        if user_email:
            query = query.filter(Submission.user_email == user_email)
        if not include_failed:
            query = query.filter(Submission.status == "scored")
        if not include_hidden:
            query = query.filter(Submission.moderation_status == "visible")
        if not include_deleted:
            query = query.filter(Submission.moderation_status != "deleted")
        rows = query.order_by(Submission.created_at.desc()).limit(limit).all()
        return [_submission_summary(row) for row in rows]


def visible_scored_submission_ids(limit: int = 1000) -> List[str]:
    """Submission ids eligible for rebuilding the public leaderboard."""
    limit = max(1, min(int(limit or 1000), 10_000))
    with _Session() as session:
        rows = (
            session.query(Submission.score_submission_id)
            .filter(
                Submission.status == "scored",
                Submission.score_submission_id.isnot(None),
                Submission.moderation_status == "visible",
            )
            .order_by(Submission.created_at.asc(), Submission.id.asc())
            .limit(limit)
            .all()
        )
        return [row[0] for row in rows if row[0]]


def latest_visible_scored_submission_id(
    model_id: str,
    task_id: str,
) -> Optional[str]:
    """Return the newest publishable run for one model and benchmark."""
    normalized_model_id = str(model_id or "").strip()
    normalized_task_id = str(task_id or "").strip()
    if not normalized_model_id or not normalized_task_id:
        return None
    with _Session() as session:
        row = (
            session.query(Submission.score_submission_id)
            .filter(
                Submission.model_id == normalized_model_id,
                Submission.task_id == normalized_task_id,
                Submission.status == "scored",
                Submission.score_submission_id.isnot(None),
                Submission.moderation_status == "visible",
            )
            .order_by(Submission.created_at.desc(), Submission.id.desc())
            .first()
        )
        return row[0] if row and row[0] else None


def latest_visible_scored_submission_ids(limit: int = 10_000) -> List[str]:
    """Return the exact submission IDs expected in the public cache."""
    limit = max(1, min(int(limit or 10_000), 100_000))
    with _Session() as session:
        rows = (
            session.query(
                Submission.model_id,
                Submission.task_id,
                Submission.score_submission_id,
            )
            .filter(
                Submission.model_id.isnot(None),
                Submission.status == "scored",
                Submission.score_submission_id.isnot(None),
                Submission.moderation_status == "visible",
            )
            .order_by(Submission.created_at.desc(), Submission.id.desc())
            .all()
        )
    seen = set()
    submission_ids = []
    for model_id, task_id, score_submission_id in rows:
        key = (model_id, task_id)
        if key in seen:
            continue
        seen.add(key)
        submission_ids.append(score_submission_id)
        if len(submission_ids) >= limit:
            break
    return submission_ids


def latest_visible_scored_submission_fingerprints(
    limit: int = 10_000,
) -> Dict[str, str]:
    """Return canonical score fingerprints expected in the public cache.

    Identity and run metadata are sourced from their authoritative submission
    columns. This prevents a stale serialized score payload from making the
    database and public cache appear synchronized after model metadata changes.
    """
    limit = max(1, min(int(limit or 10_000), 100_000))
    with _Session() as session:
        rows = (
            session.query(
                Submission.model_id,
                Submission.task_id,
                Submission.score_submission_id,
                Submission.model_name,
                Submission.model_meta_json,
                Submission.latest_score_json,
            )
            .filter(
                Submission.model_id.isnot(None),
                Submission.status == "scored",
                Submission.score_submission_id.isnot(None),
                Submission.moderation_status == "visible",
            )
            .order_by(Submission.created_at.desc(), Submission.id.desc())
            .all()
        )
    seen = set()
    fingerprints: Dict[str, str] = {}
    for (
        model_id,
        task_id,
        score_submission_id,
        model_name,
        model_meta_json,
        score_json,
    ) in rows:
        key = (model_id, task_id)
        if key in seen:
            continue
        seen.add(key)
        score = _json_loads(score_json)
        score.update({
            "submission_id": score_submission_id,
            "model_id": model_id,
            "model_name": model_name,
            "task_id": task_id,
            "model_meta": _json_loads(model_meta_json),
        })
        canonical = json.dumps(
            score,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprints[str(score_submission_id)] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        if len(fingerprints) >= limit:
            break
    return fingerprints


def set_moderation_status(
    score_submission_id: str,
    moderation_status: str,
    reason: Optional[str] = None,
    moderated_by: Optional[str] = None,
) -> Optional[Dict]:
    """Soft-hide, restore, or delete a scored submission."""
    moderation_status = (moderation_status or "").strip().lower()
    if moderation_status not in {"visible", "hidden", "deleted"}:
        raise ValueError("Invalid moderation status")
    with _Session() as session:
        row = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id),
            Submission.status == "scored",
        ).one_or_none()
        if row is None:
            return None
        previous = {
            "moderation_status": row.moderation_status or "visible",
            "moderation_reason": row.moderation_reason,
            "moderated_by": row.moderated_by,
        }
        row.moderation_status = moderation_status
        row.moderation_reason = (reason or "").strip() or None
        row.moderated_by = moderated_by
        row.moderated_at = _utcnow()
        session.commit()
        session.refresh(row)
        return {**_submission_summary(row), "previous_moderation": previous}


def delete_owned_submission(
    score_submission_id: str,
    user_email: str,
) -> Optional[Dict]:
    """Soft-delete a scored submission owned by one verified account.

    The scored row remains for quota and audit integrity, while member history,
    exports, and public leaderboard reads exclude the tombstoned submission.
    """
    normalized_email = str(user_email or "").strip().lower()
    if not normalized_email:
        return None
    with _Session() as session:
        row = session.query(Submission).filter(
            Submission.score_submission_id == str(score_submission_id),
            Submission.status == "scored",
            Submission.user_email == normalized_email,
            Submission.moderation_status != "deleted",
        ).one_or_none()
        if row is None:
            return None
        previous = {
            "moderation_status": row.moderation_status or "visible",
            "moderation_reason": row.moderation_reason,
            "moderated_by": row.moderated_by,
        }
        row.moderation_status = "deleted"
        row.moderation_reason = "Deleted by submission owner"
        row.moderated_by = normalized_email
        row.moderated_at = _utcnow()
        session.commit()
        session.refresh(row)
        return {**_submission_summary(row), "previous_moderation": previous}


def _quota_bucket(session, user_email: str, task_id: str, limit: int, now: datetime) -> dict:
    rows = _counted_query(session, user_email, task_id, now - WINDOW, now).all()
    used = len(rows)
    reset_at = min((_quota_expiry(row) for row in rows), default=None)
    return {
        "task_id": task_id,
        "limit": limit,
        "used": used,
        "remaining": max(0, limit - used),
        "reset_at": reset_at.isoformat() if reset_at else None,
    }


def quota_status(
    user_email: str,
    task_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """Return one benchmark quota bucket or the aggregate account summary."""
    limit = SUBMISSION_DAILY_LIMIT_PER_BENCHMARK if limit is None else limit
    if limit <= 0:
        raise ValueError("Submission quota limit must be positive")
    now = _utcnow()
    with _Session() as session:
        if task_id:
            return _quota_bucket(session, user_email, task_id, limit, now)
        buckets = {
            benchmark_id: _quota_bucket(session, user_email, benchmark_id, limit, now)
            for benchmark_id in BENCHMARK_TASK_IDS
        }
    used = sum(bucket["used"] for bucket in buckets.values())
    total_limit = limit * len(BENCHMARK_TASK_IDS)
    reset_values = [bucket["reset_at"] for bucket in buckets.values() if bucket["reset_at"]]
    return {
        "limit": total_limit,
        "used": used,
        "remaining": max(0, total_limit - used),
        "reset_at": min(reset_values) if reset_values else None,
        "per_benchmark_limit": limit,
        "per_benchmark": buckets,
    }
