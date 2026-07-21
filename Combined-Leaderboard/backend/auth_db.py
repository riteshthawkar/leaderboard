"""
User authentication database — email + password with email verification.

Accounts are keyed by email address (the email IS the username). New accounts
are created unverified and must confirm ownership of the email via a
verification link before they can sign in. OAuth (Google/Microsoft) identities
are treated as pre-verified. Authentication is session-cookie based — there are
no API tokens.
"""

import hashlib
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Boolean, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from config import (
        AUTH_DATABASE_URL,
        DB_MAX_OVERFLOW,
        DB_POOL_RECYCLE,
        DB_POOL_SIZE,
        SQLITE_BUSY_TIMEOUT_MS,
    )
    from sqlite_runtime import configure_sqlite_engine, harden_private_file, sqlite_connect_args
    from schema_migrations import run_schema_migrations
except ImportError:  # pragma: no cover - package import fallback
    from .config import (
        AUTH_DATABASE_URL,
        DB_MAX_OVERFLOW,
        DB_POOL_RECYCLE,
        DB_POOL_SIZE,
        SQLITE_BUSY_TIMEOUT_MS,
    )
    from .sqlite_runtime import configure_sqlite_engine, harden_private_file, sqlite_connect_args
    from .schema_migrations import run_schema_migrations

Base = declarative_base()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

# Constant-time sentinel: always run check_password_hash even for unknown users
# to prevent timing-based account enumeration.
_DUMMY_HASH: str = generate_password_hash("__dummy_sentinel__")

# Email / verification policy
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VERIFICATION_TTL_HOURS = 24
PASSWORD_RESET_TTL_HOURS = 1
MAX_EMAIL_LENGTH = 254
MIN_PASSWORD_LENGTH = 15
MAX_PASSWORD_LENGTH = 128
COMMON_PASSWORDS = {
    "123456789012345",
    "adminadminadmin",
    "correct horse battery staple",
    "correcthorsebatterystaple",
    "letmeinletmeinletmein",
    "microsoftresearch",
    "msvistaleaderboard",
    "password123456",
    "passwordpassword",
    "qwertyuiopasdfgh",
}


class OAuthIdentityConflictError(Exception):
    """Raised when an OAuth identity would take over an existing email account."""


_DB_URL = AUTH_DATABASE_URL
AUTH_SCHEMA_VERSION = 2


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _legacy_schema(db_path: Path) -> bool:
    """True if an existing users.db uses the old username/api_token schema."""
    if not db_path.exists():
        return False
    con = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(users)").fetchall()}
    except sqlite3.DatabaseError:
        return False
    finally:
        con.close()
    if not cols:
        return False
    return "email" not in cols or "api_token" in cols


# One-time migration: the previous schema stored username + api_token. Those
# accounts can't be mapped to verified emails, so we archive the old DB and start
# fresh with the email/verification schema. The backup is kept for safety.
if _DB_PATH is not None and _legacy_schema(_DB_PATH):
    _backup = _DB_PATH.with_name(f"users.legacy-{int(_utcnow().timestamp())}.db")
    _DB_PATH.replace(_backup)
    harden_private_file(_backup)

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


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    email_verified = Column(Boolean, nullable=False, default=False)
    auth_provider = Column(String(32), nullable=False, default="password")
    verification_token = Column(String(64), nullable=True, index=True)
    verification_expires_at = Column(DateTime, nullable=True)
    password_reset_token = Column(String(64), nullable=True, index=True)
    password_reset_expires_at = Column(DateTime, nullable=True)
    session_version = Column(Integer, nullable=False, default=0)
    oauth_provider = Column(String(32), nullable=True)
    oauth_subject = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    def __repr__(self):
        return f"<User {self.email}>"


def init_db() -> None:
    """Create tables if they don't exist."""
    if _DB_DRIVER.startswith("postgresql"):
        with _engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('auth-schema-init'))"))
            Base.metadata.create_all(connection)
            run_schema_migrations(
                connection,
                "auth",
                [(1, _ensure_user_columns), (2, _ensure_auth_security_columns)],
            )
    else:
        with _schema_file_lock():
            Base.metadata.create_all(_engine)
            with _engine.begin() as connection:
                run_schema_migrations(
                    connection,
                    "auth",
                    [(1, _ensure_user_columns), (2, _ensure_auth_security_columns)],
                )


def _ensure_user_columns(connection) -> None:
    """Add non-destructive auth columns for existing SQLite/Postgres DBs."""
    existing = {col["name"] for col in inspect(connection).get_columns("users")}
    datetime_type = "TIMESTAMP" if _DB_DRIVER.startswith("postgresql") else "DATETIME"
    additions = {
        "password_reset_token": "VARCHAR(64)",
        "password_reset_expires_at": datetime_type,
    }
    for column, sql_type in additions.items():
        if column not in existing:
            connection.execute(text(f"ALTER TABLE users ADD COLUMN {column} {sql_type}"))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_users_password_reset_token "
        "ON users (password_reset_token)"
    ))
    _migrate_token_digests(connection)


def _ensure_auth_security_columns(connection) -> None:
    """Add session revocation and stable OAuth identity columns."""
    existing = {col["name"] for col in inspect(connection).get_columns("users")}
    additions = {
        "session_version": "INTEGER NOT NULL DEFAULT 0",
        "oauth_provider": "VARCHAR(32)",
        "oauth_subject": "VARCHAR(255)",
    }
    for column, sql_type in additions.items():
        if column not in existing:
            connection.execute(text(f"ALTER TABLE users ADD COLUMN {column} {sql_type}"))
    connection.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_oauth_identity "
        "ON users (oauth_provider, oauth_subject)"
    ))
    _migrate_token_digests(connection)


def _migrate_token_digests(connection) -> None:
    """Hash active tokens created by releases that stored them in plaintext."""
    digest_pattern = re.compile(r"^[0-9a-f]{64}$")
    for column in ("verification_token", "password_reset_token"):
        rows = connection.execute(
            text(f"SELECT id, {column} AS token FROM users WHERE {column} IS NOT NULL")
        ).mappings()
        for row in rows:
            value = str(row["token"] or "")
            if value and not digest_pattern.fullmatch(value):
                connection.execute(
                    text(f"UPDATE users SET {column} = :digest WHERE id = :user_id"),
                    {"digest": _token_digest(value), "user_id": row["id"]},
                )


def normalize_email(email: str) -> str:
    return email.strip().lower() if isinstance(email, str) else ""


def is_valid_email(email: str) -> bool:
    normalized = normalize_email(email)
    if len(normalized) > MAX_EMAIL_LENGTH or not EMAIL_RE.fullmatch(normalized):
        return False
    local, domain = normalized.rsplit("@", 1)
    if len(local) > 64 or local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if len(ascii_domain) > 253:
        return False
    labels = ascii_domain.split(".")
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        and re.fullmatch(r"[a-z0-9-]+", label)
        for label in labels
    )


def password_policy_status(password: str) -> str:
    """Validate length and a small high-frequency/context password blocklist."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        return "too_short"
    if len(password) > MAX_PASSWORD_LENGTH:
        return "too_long"
    if password.strip().casefold() in COMMON_PASSWORDS:
        return "common"
    return "ok"


def _new_verification() -> Tuple[str, datetime]:
    return secrets.token_urlsafe(32), _utcnow() + timedelta(hours=VERIFICATION_TTL_HOURS)


def _new_password_reset() -> Tuple[str, datetime]:
    return secrets.token_urlsafe(32), _utcnow() + timedelta(hours=PASSWORD_RESET_TTL_HOURS)


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_candidates(token: str) -> tuple[str, ...]:
    """Accept existing plaintext tokens while storing all newly issued tokens as digests."""
    digest = _token_digest(token)
    return (digest,) if digest == token else (digest, token)


def _token_consume_query(session, query):
    """Serialize one-time token consumption for the configured database."""
    if _DB_DRIVER.startswith("postgresql"):
        return query.with_for_update()
    if _DB_DRIVER.startswith("sqlite"):
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    return query


def register_user(email: str, password: str) -> Tuple[str, Optional[str]]:
    """
    Register a new email/password account (created unverified).

    Returns (status, verification_token):
      "created"  -> new account; email the token
      "resent"   -> existing UNVERIFIED account; email a fresh token
      "exists"   -> account already exists and is verified (token is None)
      "invalid"  -> malformed email or too-short password (token is None)
    """
    email = normalize_email(email)
    if not is_valid_email(email) or password_policy_status(password) != "ok":
        return "invalid", None
    token, expires = _new_verification()
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if user:
            if user.email_verified:
                return "exists", None
            # The newest verification token and password must belong to the
            # same registration attempt. Keeping the first password here would
            # let a pre-registration attacker retain credentials after the real
            # mailbox owner verifies a later request.
            user.password_hash = generate_password_hash(password)
            user.auth_provider = "password"
            user.oauth_provider = None
            user.oauth_subject = None
            user.verification_token = _token_digest(token)
            user.verification_expires_at = expires
            user.password_reset_token = None
            user.password_reset_expires_at = None
            user.session_version = (user.session_version or 0) + 1
            session.commit()
            return "resent", token
        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            email_verified=False,
            auth_provider="password",
            verification_token=_token_digest(token),
            verification_expires_at=expires,
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            # A concurrent registration may have created the address after the
            # initial lookup. Apply the same verified/unverified policy after
            # rollback instead of surfacing a generic database failure.
            session.rollback()
            user = session.query(User).filter_by(email=email).one_or_none()
            if user is None:
                raise
            if user.email_verified:
                return "exists", None
            user.password_hash = generate_password_hash(password)
            user.auth_provider = "password"
            user.oauth_provider = None
            user.oauth_subject = None
            user.verification_token = _token_digest(token)
            user.verification_expires_at = expires
            user.password_reset_token = None
            user.password_reset_expires_at = None
            user.session_version = (user.session_version or 0) + 1
            session.commit()
            return "resent", token
        return "created", token


def verify_email(token: str) -> Optional[str]:
    """Consume a verification token. Returns the verified email, or None if invalid/expired."""
    token = (token or "").strip()
    if not token:
        return None
    with _Session() as session:
        query = session.query(User).filter(
            User.verification_token.in_(_token_candidates(token))
        )
        user = _token_consume_query(session, query).first()
        if not user:
            return None
        expires = user.verification_expires_at
        if expires is not None:
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < _utcnow():
                user.verification_token = None
                user.verification_expires_at = None
                session.commit()
                return None
        user.email_verified = True
        user.verification_token = None
        user.verification_expires_at = None
        session.commit()
        return user.email


def resend_verification(email: str) -> Tuple[str, Optional[str]]:
    """
    Issue a fresh verification token for an unverified account.
    Returns (status, token): "resent" | "verified" | "unknown" | "invalid".
    """
    email = normalize_email(email)
    if not is_valid_email(email):
        return "invalid", None
    token, expires = _new_verification()
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return "unknown", None
        if user.email_verified:
            return "verified", None
        user.verification_token = _token_digest(token)
        user.verification_expires_at = expires
        session.commit()
        return "resent", token


def login_user(email: str, password: str) -> str:
    """
    Verify credentials. Returns "ok" | "unverified" | "invalid".
    Always runs check_password_hash regardless of account existence to prevent
    timing-based account enumeration.
    """
    email = normalize_email(email)
    password = password if isinstance(password, str) else ""
    if len(password) > MAX_PASSWORD_LENGTH:
        check_password_hash(_DUMMY_HASH, "")
        return "invalid"
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        target_hash = user.password_hash if user else _DUMMY_HASH
        match = check_password_hash(target_hash, password or "")
        if not (match and user):
            return "invalid"
        if not user.email_verified:
            return "unverified"
        return "ok"


def request_password_reset(email: str) -> Tuple[str, Optional[str]]:
    """
    Create a short-lived password reset token for an account if it exists.

    Returns ("sent", token_or_none) for both known and unknown valid emails so
    the API can avoid account enumeration. Returns ("invalid", None) for
    malformed emails.
    """
    email = normalize_email(email)
    if not is_valid_email(email):
        return "invalid", None
    token, expires = _new_password_reset()
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return "sent", None
        user.password_reset_token = _token_digest(token)
        user.password_reset_expires_at = expires
        session.commit()
        return "sent", token


def reset_password(token: str, password: str) -> str:
    """Consume a reset token and set a new password."""
    token = (token or "").strip()
    policy_status = password_policy_status(password)
    if policy_status == "too_short":
        return "weak"
    if policy_status == "too_long":
        return "too_long"
    if policy_status == "common":
        return "common"
    if not token:
        return "invalid"
    with _Session() as session:
        query = session.query(User).filter(
            User.password_reset_token.in_(_token_candidates(token))
        )
        user = _token_consume_query(session, query).first()
        if not user:
            return "invalid"
        expires = user.password_reset_expires_at
        if expires is not None:
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < _utcnow():
                user.password_reset_token = None
                user.password_reset_expires_at = None
                session.commit()
                return "invalid"
        user.password_hash = generate_password_hash(password)
        user.email_verified = True
        user.auth_provider = "password"
        user.password_reset_token = None
        user.password_reset_expires_at = None
        user.verification_token = None
        user.verification_expires_at = None
        user.session_version = (user.session_version or 0) + 1
        session.commit()
        return "ok"


def oauth_upsert_user(
    provider: str,
    email: Optional[str],
    subject: Optional[str],
) -> Optional[str]:
    """
    Create or reuse an account for an OAuth identity (email is provider-verified).
    Returns the account email, or None if the provider gave no usable email.
    """
    email = normalize_email(email or "")
    provider = provider.strip().lower() if isinstance(provider, str) else ""
    provider = provider or "oauth"
    subject = subject.strip() if isinstance(subject, str) else ""
    if not is_valid_email(email) or not subject or len(subject) > 255:
        return None
    with _Session() as session:
        identity_user = session.query(User).filter_by(
            oauth_provider=provider,
            oauth_subject=subject,
        ).first()
        if identity_user:
            return identity_user.email

        user = session.query(User).filter_by(email=email).first()
        if user:
            # Email equality alone is not enough to link identities. Some OIDC
            # providers return mutable usernames in the email-shaped claim, so
            # automatic linking could replace an existing account's sign-in
            # method. Linking must be a separate authenticated action.
            raise OAuthIdentityConflictError(email)
        user = User(
            email=email,
            password_hash=generate_password_hash(secrets.token_urlsafe(32)),
            email_verified=True,
            auth_provider=provider,
            oauth_provider=provider,
            oauth_subject=subject,
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            # Another callback may have inserted the same identity or email
            # after the lookups above. Re-read after rollback and preserve the
            # same fail-closed linking rules instead of returning a generic DB
            # error or replacing an existing account.
            session.rollback()
            identity_user = session.query(User).filter_by(
                oauth_provider=provider,
                oauth_subject=subject,
            ).first()
            if identity_user:
                return identity_user.email
            if session.query(User).filter_by(email=email).first():
                raise OAuthIdentityConflictError(email)
            raise
        return user.email


def get_user(email: str) -> Optional[dict]:
    """Return public account metadata for an email, or None."""
    email = normalize_email(email)
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return None
        return {
            "email": user.email,
            "email_verified": bool(user.email_verified),
            "auth_provider": user.auth_provider,
            "session_version": int(user.session_version or 0),
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }


def get_verified_admin_emails(emails) -> list[str]:
    """Return configured addresses that map to verified local accounts."""
    normalized = sorted({normalize_email(email) for email in emails if normalize_email(email)})
    if not normalized:
        return []
    with _Session() as session:
        rows = (
            session.query(User.email)
            .filter(User.email.in_(normalized), User.email_verified.is_(True))
            .all()
        )
    return sorted(row[0] for row in rows)
