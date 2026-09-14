"""
User authentication database — email + password with email verification.

Accounts are keyed by email address (the email IS the username). New accounts
are created unverified and must confirm ownership of the email via a
verification link before they can sign in. OAuth (Google/Microsoft) identities
are treated as pre-verified. The web service can authenticate with either a
same-origin session cookie or short-lived access tokens backed by rotating,
server-revocable refresh tokens.
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
AUTH_SCHEMA_VERSION = 3


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


class RefreshToken(Base):
    """Hashed, rotating refresh credential for browser bearer sessions."""

    __tablename__ = "auth_refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_digest = Column(String(64), unique=True, nullable=False, index=True)
    user_email = Column(String(255), nullable=False, index=True)
    family_id = Column(String(64), nullable=False, index=True)
    session_version = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    rotated_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    replaced_by_digest = Column(String(64), nullable=True)


class OAuthExchangeCode(Base):
    """One-time code used to hand an OAuth login back to a static frontend."""

    __tablename__ = "auth_oauth_exchange_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code_digest = Column(String(64), unique=True, nullable=False, index=True)
    user_email = Column(String(255), nullable=False, index=True)
    session_version = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    consumed_at = Column(DateTime, nullable=True)


def init_db() -> None:
    """Create tables if they don't exist."""
    if _DB_DRIVER.startswith("postgresql"):
        with _engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('auth-schema-init'))"))
            Base.metadata.create_all(connection)
            run_schema_migrations(
                connection,
                "auth",
                [
                    (1, _ensure_user_columns),
                    (2, _ensure_auth_security_columns),
                    (3, _ensure_bearer_auth_tables),
                ],
            )
    else:
        with _schema_file_lock():
            Base.metadata.create_all(_engine)
            with _engine.begin() as connection:
                run_schema_migrations(
                    connection,
                    "auth",
                    [
                        (1, _ensure_user_columns),
                        (2, _ensure_auth_security_columns),
                        (3, _ensure_bearer_auth_tables),
                    ],
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


def _ensure_bearer_auth_tables(connection) -> None:
    """Create the additive token tables used by cross-site static clients."""
    RefreshToken.__table__.create(connection, checkfirst=True)
    OAuthExchangeCode.__table__.create(connection, checkfirst=True)


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


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _purge_expired_browser_credentials(session) -> None:
    """Bound token-table growth while retaining live reuse-detection records."""
    now = _utcnow()
    session.query(RefreshToken).filter(RefreshToken.expires_at < now).delete(
        synchronize_session=False
    )
    session.query(OAuthExchangeCode).filter(OAuthExchangeCode.expires_at < now).delete(
        synchronize_session=False
    )


def _verified_user(session, email: str) -> Optional[User]:
    user = session.query(User).filter_by(email=normalize_email(email)).first()
    return user if user and user.email_verified else None


def _revoke_user_refresh_tokens_in_session(session, email: str, now: datetime) -> int:
    return int(session.query(RefreshToken).filter(
        RefreshToken.user_email == normalize_email(email),
        RefreshToken.revoked_at.is_(None),
    ).update(
        {RefreshToken.revoked_at: now},
        synchronize_session=False,
    ) or 0)


def issue_refresh_token(email: str, ttl_days: int) -> Optional[str]:
    """Create a new refresh-token family for a verified account."""
    if ttl_days <= 0:
        raise ValueError("ttl_days must be positive")
    raw_token = secrets.token_urlsafe(48)
    with _Session() as session:
        user = _verified_user(session, email)
        if user is None:
            return None
        _purge_expired_browser_credentials(session)
        session.add(RefreshToken(
            token_digest=_token_digest(raw_token),
            user_email=user.email,
            family_id=secrets.token_hex(32),
            session_version=int(user.session_version or 0),
            expires_at=_utcnow() + timedelta(days=ttl_days),
        ))
        session.commit()
    return raw_token


def rotate_refresh_token(token: str, ttl_days: int) -> tuple[str, Optional[str], Optional[str]]:
    """Consume and replace a refresh token.

    Returns ``(status, email, replacement)`` where status is ``ok``,
    ``invalid``, ``expired``, ``revoked``, or ``reused``. Reuse of a token that
    was already rotated revokes its entire family.
    """
    token = (token or "").strip()
    if not token or ttl_days <= 0:
        return "invalid", None, None
    digest = _token_digest(token)
    replacement = secrets.token_urlsafe(48)
    replacement_digest = _token_digest(replacement)
    now = _utcnow()
    with _Session() as session:
        query = session.query(RefreshToken).filter_by(token_digest=digest)
        row = _token_consume_query(session, query).first()
        if row is None:
            return "invalid", None, None
        if row.rotated_at is not None:
            session.query(RefreshToken).filter_by(family_id=row.family_id).update(
                {RefreshToken.revoked_at: now},
                synchronize_session=False,
            )
            session.commit()
            return "reused", None, None
        if row.revoked_at is not None:
            return "revoked", None, None
        if (_as_utc(row.expires_at) or now) <= now:
            row.revoked_at = now
            session.commit()
            return "expired", None, None
        user = _verified_user(session, row.user_email)
        if (
            user is None
            or int(row.session_version) != int(user.session_version or 0)
        ):
            session.query(RefreshToken).filter_by(family_id=row.family_id).update(
                {RefreshToken.revoked_at: now},
                synchronize_session=False,
            )
            session.commit()
            return "revoked", None, None

        row.rotated_at = now
        row.replaced_by_digest = replacement_digest
        session.add(RefreshToken(
            token_digest=replacement_digest,
            user_email=user.email,
            family_id=row.family_id,
            session_version=int(user.session_version or 0),
            expires_at=now + timedelta(days=ttl_days),
        ))
        session.commit()
        return "ok", user.email, replacement


def revoke_refresh_token(token: str) -> bool:
    """Revoke one refresh token without revealing whether it existed."""
    token = (token or "").strip()
    if not token:
        return False
    now = _utcnow()
    with _Session() as session:
        query = session.query(RefreshToken).filter_by(token_digest=_token_digest(token))
        row = _token_consume_query(session, query).first()
        if row is None:
            return False
        if row.revoked_at is None:
            row.revoked_at = now
            session.commit()
        return True


def revoke_user_refresh_tokens(email: str) -> int:
    """Revoke every active refresh token for an account."""
    now = _utcnow()
    with _Session() as session:
        count = _revoke_user_refresh_tokens_in_session(session, email, now)
        session.commit()
        return count


def issue_oauth_exchange_code(email: str, ttl_seconds: int = 60) -> Optional[str]:
    """Issue a short-lived one-time code after a successful OAuth callback."""
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    raw_code = secrets.token_urlsafe(32)
    with _Session() as session:
        user = _verified_user(session, email)
        if user is None:
            return None
        _purge_expired_browser_credentials(session)
        session.add(OAuthExchangeCode(
            code_digest=_token_digest(raw_code),
            user_email=user.email,
            session_version=int(user.session_version or 0),
            expires_at=_utcnow() + timedelta(seconds=ttl_seconds),
        ))
        session.commit()
    return raw_code


def consume_oauth_exchange_code(code: str) -> Optional[str]:
    """Atomically consume an OAuth handoff code and return its account email."""
    code = (code or "").strip()
    if not code:
        return None
    now = _utcnow()
    with _Session() as session:
        query = session.query(OAuthExchangeCode).filter_by(
            code_digest=_token_digest(code)
        )
        row = _token_consume_query(session, query).first()
        if row is None or row.consumed_at is not None:
            return None
        if (_as_utc(row.expires_at) or now) <= now:
            row.consumed_at = now
            session.commit()
            return None
        user = _verified_user(session, row.user_email)
        if (
            user is None
            or int(row.session_version) != int(user.session_version or 0)
        ):
            row.consumed_at = now
            session.commit()
            return None
        row.consumed_at = now
        session.commit()
        return user.email


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
            _revoke_user_refresh_tokens_in_session(session, user.email, _utcnow())
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
            _revoke_user_refresh_tokens_in_session(session, user.email, _utcnow())
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
        _revoke_user_refresh_tokens_in_session(session, user.email, _utcnow())
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


# --- data subject rights -----------------------------------------------------------------
ANONYMIZED_EMAIL_DOMAIN = "anonymized.invalid"  # .invalid is reserved (RFC 2606): never routable


def _anonymized_email() -> str:
    return f"deleted-user-{secrets.token_hex(6)}@{ANONYMIZED_EMAIL_DOMAIN}"


def allocate_anonymized_email() -> str:
    """Return a non-routable replacement address not already used by an account."""
    with _Session() as session:
        replacement = _anonymized_email()
        while session.query(User.id).filter_by(email=replacement).first():
            replacement = _anonymized_email()
        return replacement


def _export_datetime(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def export_user_data(email: str) -> Optional[dict]:
    """Export account and browser-session metadata without credential material."""
    email = normalize_email(email)
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return None
        account = {
            "email": user.email,
            "email_verified": bool(user.email_verified),
            "auth_provider": user.auth_provider,
            "oauth_provider": user.oauth_provider,
            "oauth_subject": user.oauth_subject,
            "created_at": _export_datetime(user.created_at),
        }
        refresh_tokens = session.query(RefreshToken).filter_by(user_email=email).all()
        exchange_codes = session.query(OAuthExchangeCode).filter_by(user_email=email).all()
        records = {
            "auth_refresh_tokens": [
                {
                    "session_version": row.session_version,
                    "expires_at": _export_datetime(row.expires_at),
                    "created_at": _export_datetime(row.created_at),
                    "rotated_at": _export_datetime(row.rotated_at),
                    "revoked_at": _export_datetime(row.revoked_at),
                }
                for row in refresh_tokens
            ],
            "auth_oauth_exchange_codes": [
                {
                    "session_version": row.session_version,
                    "expires_at": _export_datetime(row.expires_at),
                    "created_at": _export_datetime(row.created_at),
                    "consumed_at": _export_datetime(row.consumed_at),
                }
                for row in exchange_codes
            ],
        }
        return {"account": account, "records": records}


def anonymize_user(email: str, replacement: Optional[str] = None) -> Optional[str]:
    """Erase an account identity and its browser credentials.

    Submission ownership is rewritten through ``submission_store`` because a
    deployment may keep authentication and submissions in separate databases.
    """
    email = normalize_email(email)
    replacement = normalize_email(replacement or allocate_anonymized_email())
    if not replacement.endswith(f"@{ANONYMIZED_EMAIL_DOMAIN}"):
        raise ValueError("replacement must use the reserved anonymized email domain")
    with _Session() as session:
        user = session.query(User).filter_by(email=email).first()
        if not user:
            return None
        collision = session.query(User.id).filter(
            User.email == replacement,
            User.id != user.id,
        ).first()
        if collision:
            raise ValueError("replacement email is already in use")

        session.query(RefreshToken).filter_by(user_email=email).delete(
            synchronize_session=False
        )
        session.query(OAuthExchangeCode).filter_by(user_email=email).delete(
            synchronize_session=False
        )

        user.email = replacement
        user.password_hash = generate_password_hash(secrets.token_urlsafe(32))
        user.email_verified = False
        user.auth_provider = "anonymized"
        user.oauth_provider = None
        user.oauth_subject = None
        user.verification_token = None
        user.verification_expires_at = None
        user.password_reset_token = None
        user.password_reset_expires_at = None
        user.session_version = int(user.session_version or 0) + 1
        session.commit()
        return replacement
