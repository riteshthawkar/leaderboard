"""Production Flask API service for the MS-VISTA leaderboard."""

import os
import re
import secrets
import sys
import json
import hashlib
import hmac
import base64
import logging
import atexit
import io
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlparse
from typing import Dict, Optional

# Ensure the backend package directory is importable when running this file
# directly (e.g. `python backend/web/app.py`) so that `from config import ...`
# and the other top-level module imports below resolve correctly.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from flask import (
    Flask,
    Request as FlaskRequest,
    Response,
    g,
    jsonify,
    redirect,
    request,
    send_file,
    session,
    url_for,
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy.engine import make_url
from werkzeug.middleware.proxy_fix import ProxyFix
import requests

from config import (
    TASKS, SECTIONS, SPATIAL_DATASETS, SPATIAL_MANIFEST_FILE, SPATIAL_HARNESS_DIR,
    NO_IMAGE_PLUS_OPTION,
    LAYER_LABELS, VCI_LAYER_WEIGHTS, EVAL_CONDITIONS, GRADING,
    OFFICIAL_SPATIAL_MIN_SAMPLES, REQUIRE_OFFICIAL_SPATIAL,
    AUTH_DATABASE_URL, AUTO_BACKUP_DIR, AUTO_BACKUP_ENABLED,
    AUTO_BACKUP_INTERVAL_HOURS, AUTO_BACKUP_POLL_SECONDS,
    AUTO_BACKUP_RETENTION_COUNT, AUTO_BACKUP_RUN_ON_START,
    AUTO_BACKUP_MIRROR_DIR, REQUIRE_OFFSITE_BACKUP,
    DATABASE_URL, DATA_DIR, LEADERBOARD_STORE_FILE, RESULTS_DIR,
    SQLITE_BUSY_TIMEOUT_MS, SUBMISSION_DATABASE_URL, DEPLOYMENT_MODE,
    WEB_CONCURRENCY,
    PUBLIC_DEPLOYMENT, MIN_FREE_DISK_BYTES, MIN_FREE_DISK_PERCENT,
    GROUND_TRUTHS_SOURCE, GROUND_TRUTHS_HF_REVISION,
)
from auth_db import (
    OAuthIdentityConflictError,
    init_db as init_auth_db,
    register_user,
    login_user,
    oauth_upsert_user,
    verify_email,
    resend_verification,
    request_password_reset,
    reset_password,
    get_user,
    get_verified_admin_emails,
    is_valid_email,
    normalize_email,
    password_policy_status,
    MAX_EMAIL_LENGTH,
    MIN_PASSWORD_LENGTH,
    MAX_PASSWORD_LENGTH,
)
from emailer import send_password_reset_email, send_verification_email
from scoring.task_scorer import SubmissionValidationError, TaskScorer
from spatial_submission import (
    SPATIAL_ARCHIVE_MEMBERS,
    SPATIAL_MANIFEST_MEMBER,
    SPATIAL_PUBLIC_ARTIFACT_NAMES,
    SPATIAL_REPORT_MEMBER,
    SPATIAL_SUBMISSION_ARCHIVE_NAME,
    SPATIAL_SUBMISSION_MEMBER,
    build_spatial_task_score,
    parse_spatial_evidence,
    read_spatial_submission_archive,
    spatial_bundle_health as inspect_spatial_bundle,
    validate_run_manifest,
    validate_spatial_report,
)
from leaderboard_store import LeaderboardStore
from submission_store import (
    ModelNameConflictError,
    create_registered_model,
    find_owned_model_by_name,
    get_owned_model,
    init_db as init_submission_db,
    try_consume_quota,
    finalize_submission,
    quota_status,
    store_submission_answers,
    get_submission_export,
    get_public_spatial_artifact,
    get_public_spatial_evidence,
    get_submission_for_rescore,
    delete_owned_submission,
    list_registered_models,
    list_submissions,
    set_moderation_status,
    update_submission_score,
    latest_visible_scored_submission_id,
    latest_visible_scored_submission_ids,
    latest_visible_scored_submission_fingerprints,
    submission_integrity_status,
)
from data_handlers.ground_truth import GroundTruthManager
from request_models import HealthCheckResponse
from file_security import FileSecurityValidator
from constants import (
    SUBMISSIONS_PER_HOUR,
    SUBMISSIONS_PER_DAY,
    DEFAULT_LEADERBOARD_LIMIT,
    MIN_LEADERBOARD_LIMIT,
    MAX_LEADERBOARD_LIMIT,
    MAX_MODEL_NAME_LENGTH,
    MAX_SPATIAL_ARCHIVE_BYTES,
    MAX_SPATIAL_MANIFEST_BYTES,
    MAX_SPATIAL_MULTIPART_BYTES,
    MAX_SPATIAL_SUBMISSION_BYTES,
    MAX_SPATIAL_ZIP_COMPRESSION_RATIO,
)
from logging_config import logger
from backup import BackupScheduler, create_backup_archive, validate_backup_archive

# Flask is intentionally API-only. The React application is built and hosted as
# a separate service, so the backend has no template or static-file routes.
app = Flask(__name__, static_folder=None)


class InMemorySpatialUploadRequest(FlaskRequest):
    """Keep the bounded spatial upload stream in RAM instead of a temp file."""

    def _get_file_stream(
        self,
        total_content_length,
        content_type,
        filename=None,
        content_length=None,
    ):
        if self.path.rstrip("/") == "/api/tasks/spatial/submit":
            return io.BytesIO()
        return super()._get_file_stream(
            total_content_length,
            content_type,
            filename,
            content_length,
        )


app.request_class = InMemorySpatialUploadRequest


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str) -> list[str]:
    values = [item.strip() for item in os.getenv(name, default).split(",")]
    return [item for item in values if item]


def _nonnegative_int_env(name: str, default: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative whole number.") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative whole number.")
    return value


# Only trust forwarding headers when the deployment explicitly declares how
# many reverse proxies sit in front of the API. This keeps client-IP rate limits
# correct without trusting spoofable X-Forwarded-* headers on direct installs.
TRUST_PROXY_HOPS = _nonnegative_int_env("TRUST_PROXY_HOPS")
if TRUST_PROXY_HOPS:
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=TRUST_PROXY_HOPS,
        x_proto=TRUST_PROXY_HOPS,
    )


# Secret key – required for Flask internals (CSRF helpers, signed cookies, etc.)
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
_secret = os.getenv("SECRET_KEY", "")
_secret_is_placeholder = _secret.strip().lower() in {
    "",
    "change-me",
    "change-me-to-a-long-random-hex-string",
}
_strict_production_auth = (
    (PUBLIC_DEPLOYMENT or os.getenv("FLASK_ENV", "").strip().lower() == "production")
    and not _env_bool("FLASK_DEBUG", False)
    and not _env_bool("AUTH_DEV_MODE", False)
    and not _env_bool("ALLOW_INSECURE_SECRET", False)
    and "pytest" not in sys.modules
)
if _secret_is_placeholder:
    if _strict_production_auth:
        raise RuntimeError(
            "SECRET_KEY must be set to a long random value before starting in production."
        )
    import warnings
    warnings.warn(
        "SECRET_KEY is missing or still set to the placeholder value. Using a "
        "random key — sessions will not survive server restarts. Set SECRET_KEY "
        "in your .env for production.",
        stacklevel=1,
    )
    _secret = secrets.token_hex(32)
elif _strict_production_auth and len(_secret) < 32:
    raise RuntimeError(
        "SECRET_KEY must contain at least 32 characters before starting in production."
    )
app.secret_key = _secret

_session_cookie_samesite_raw = os.getenv("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
_session_cookie_samesite_values = {"lax": "Lax", "strict": "Strict", "none": "None"}
if _session_cookie_samesite_raw.lower() not in _session_cookie_samesite_values:
    raise RuntimeError("SESSION_COOKIE_SAMESITE must be Lax, Strict, or None.")
_session_cookie_samesite = _session_cookie_samesite_values[
    _session_cookie_samesite_raw.lower()
]
_session_cookie_secure = _env_bool("SESSION_COOKIE_SECURE", not _env_bool("AUTH_DEV_MODE", False))
if _session_cookie_samesite == "None" and not _session_cookie_secure:
    raise RuntimeError(
        "SESSION_COOKIE_SAMESITE=None requires SESSION_COOKIE_SECURE=true."
    )
_session_lifetime_days = _nonnegative_int_env("SESSION_LIFETIME_DAYS", 7)
if _session_lifetime_days <= 0:
    raise RuntimeError("SESSION_LIFETIME_DAYS must be a positive whole number.")

# Session cookie hardening (auth is cookie-session based; there are no API tokens).
app.config.update(
    SESSION_COOKIE_NAME=os.getenv("SESSION_COOKIE_NAME", "ms_vista_session").strip()
    or "ms_vista_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_PATH="/",
    SESSION_COOKIE_SAMESITE=_session_cookie_samesite,
    SESSION_COOKIE_SECURE=_session_cookie_secure,
    PERMANENT_SESSION_LIFETIME=timedelta(days=_session_lifetime_days),
)

logger.info(
    "Task submissions use deterministic ground-truth matching; spatial final answers also require a verified harness judge manifest."
)

# CORS configuration. Credentialed browser requests require exact frontend
# origins; wildcard origins are intentionally unsupported.
CORS_ORIGINS = [origin.rstrip("/") for origin in _csv_env(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174",
)]
if "*" in CORS_ORIGINS:
    raise RuntimeError(
        "CORS_ORIGINS cannot contain '*' because the API uses credentialed session cookies."
    )
invalid_cors_origins = []
for origin in CORS_ORIGINS:
    parsed_origin = urlparse(origin)
    normalized_origin = (
        f"{parsed_origin.scheme}://{parsed_origin.netloc}"
        if parsed_origin.scheme in {"http", "https"} and parsed_origin.netloc
        else ""
    )
    if normalized_origin != origin:
        invalid_cors_origins.append(origin)
if invalid_cors_origins:
    raise RuntimeError(
        "CORS_ORIGINS entries must be HTTP(S) origins without paths: "
        + ", ".join(invalid_cors_origins)
    )
CORS(app, resources={
    r"/api(?:/.*)?$": {
        "origins": CORS_ORIGINS,
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Accept", "Content-Type", "X-CSRF-Token"],
        "expose_headers": ["Content-Disposition", "Retry-After", "X-Request-Id"],
        "supports_credentials": True,
        "max_age": 600,
    }
})

# File upload configuration
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))
MAX_AUTH_REQUEST_BYTES = int(os.getenv("MAX_AUTH_REQUEST_BYTES", str(16 * 1024)))
if app.config["MAX_CONTENT_LENGTH"] <= 0:
    raise RuntimeError("MAX_CONTENT_LENGTH must be a positive integer.")
if MAX_AUTH_REQUEST_BYTES <= 0:
    raise RuntimeError("MAX_AUTH_REQUEST_BYTES must be a positive integer.")
if any(
    value <= 0
    for value in (
        MAX_SPATIAL_ARCHIVE_BYTES,
        MAX_SPATIAL_MULTIPART_BYTES,
        MAX_SPATIAL_SUBMISSION_BYTES,
        MAX_SPATIAL_MANIFEST_BYTES,
        MAX_SPATIAL_ZIP_COMPRESSION_RATIO,
    )
):
    raise RuntimeError("Spatial upload size and compression limits must be positive integers.")
if MAX_SPATIAL_MULTIPART_BYTES <= MAX_SPATIAL_ARCHIVE_BYTES:
    raise RuntimeError(
        "MAX_SPATIAL_MULTIPART_BYTES must exceed MAX_SPATIAL_ARCHIVE_BYTES."
    )

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("LIMITER_STORAGE_URI", "memory://")
)

HEALTH_CACHE_SECONDS = min(
    _nonnegative_int_env("HEALTH_CACHE_SECONDS", 15),
    300,
)
_health_cache_lock = threading.Lock()
_health_cache = {
    "expires_at": 0.0,
    "payload": None,
    "status_code": None,
}

_email_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="auth-email")
atexit.register(_email_executor.shutdown, wait=False, cancel_futures=True)

# Authentication — cookie session based (no API tokens).
SUBMISSION_AUTH_DISABLED = _env_bool("DISABLE_SUBMISSION_AUTH", False)
TEST_SUBMISSION_USER = os.getenv("TEST_SUBMISSION_USER", "local-test@ms-vista.local")
AUTH_DEV_MODE = _env_bool("AUTH_DEV_MODE", False)
ADMIN_EMAILS = {email.lower() for email in _csv_env("ADMIN_EMAILS", "")}

OAUTH_STATE_COOKIE = "vista_oauth_state"
OAUTH_STATE_MAX_AGE = 600
CSRF_SESSION_KEY = "csrf_token"
AUTH_SESSION_VERSION_KEY = "auth_session_version"
OAUTH_PKCE_SESSION_KEY = "oauth_pkce_verifier"
CSRF_HEADER = "X-CSRF-Token"
CSRF_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSRF_EXEMPT_ENDPOINTS = {
    "auth_register",
    "auth_verify",
    "auth_login",
    "auth_forgot_password",
    "auth_reset_password",
    "auth_resend",
    "auth_oauth_start",
    "auth_oauth_callback",
}
OAUTH_PROVIDERS = {
    "google": {
        "label": "Google",
        "client_id_env": "GOOGLE_CLIENT_ID",
        "client_secret_env": "GOOGLE_CLIENT_SECRET",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "microsoft": {
        "label": "Microsoft",
        "client_id_env": "MICROSOFT_CLIENT_ID",
        "client_secret_env": "MICROSOFT_CLIENT_SECRET",
        "authorize_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/oidc/userinfo",
        "scope": "openid email profile",
    },
}
MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "common").strip() or "common"
if (
    len(MICROSOFT_TENANT_ID) > 128
    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", MICROSOFT_TENANT_ID)
    or ".." in MICROSOFT_TENANT_ID
):
    raise RuntimeError(
        "MICROSOFT_TENANT_ID must be common, organizations, consumers, a tenant GUID, or a tenant domain."
    )


def _is_public_url(value: str) -> bool:
    parsed = urlparse((value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _url_origin(value: str) -> str:
    parsed = urlparse((value or "").strip())
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""


def _cookie_site(value: str) -> str:
    """Return a conservative registrable-site key for cookie readiness checks."""
    hostname = (urlparse((value or "").strip()).hostname or "").lower().rstrip(".")
    if not hostname or hostname in {"localhost", "127.0.0.1", "::1"}:
        return hostname
    hosted_suffixes = {
        "azurewebsites.net",
        "firebaseapp.com",
        "github.io",
        "hf.space",
        "netlify.app",
        "pages.dev",
        "vercel.app",
        "web.app",
    }
    for suffix in hosted_suffixes:
        if hostname == suffix:
            return hostname
        if hostname.endswith(f".{suffix}"):
            prefix = hostname[: -(len(suffix) + 1)]
            return f"{prefix.rsplit('.', 1)[-1]}.{suffix}"
    labels = hostname.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else hostname


def _validate_production_public_urls() -> None:
    """Prevent auth links from being built from an untrusted Host header."""
    production = (
        PUBLIC_DEPLOYMENT
        and not _env_bool("FLASK_DEBUG", False)
        and not AUTH_DEV_MODE
        and not _env_bool("ALLOW_INSECURE_SECRET", False)
    )
    if not production or SUBMISSION_AUTH_DISABLED:
        return

    frontend_base = os.getenv("FRONTEND_BASE_URL", "").strip()
    api_base = (
        os.getenv("API_BASE_URL", "").strip()
        or os.getenv("OAUTH_REDIRECT_BASE_URL", "").strip()
    )
    invalid = []
    if not _is_public_url(frontend_base) or urlparse(frontend_base).scheme != "https":
        invalid.append("FRONTEND_BASE_URL")
    if not _is_public_url(api_base) or urlparse(api_base).scheme != "https":
        invalid.append("API_BASE_URL")
    frontend_origin = _url_origin(frontend_base)
    allowed_origins = {origin.rstrip("/") for origin in CORS_ORIGINS}
    if frontend_origin and frontend_origin not in allowed_origins:
        invalid.append("CORS_ORIGINS containing the FRONTEND_BASE_URL origin")

    oauth_configured = any(
        os.getenv(config[key], "").strip()
        for config in OAUTH_PROVIDERS.values()
        for key in ("client_id_env", "client_secret_env")
    )
    oauth_redirect = os.getenv("OAUTH_REDIRECT_BASE_URL", "").strip()
    if oauth_configured and (
        not _is_public_url(oauth_redirect)
        or urlparse(oauth_redirect).scheme != "https"
    ):
        invalid.append("HTTPS OAUTH_REDIRECT_BASE_URL when OAuth is configured")
    privacy_policy = os.getenv("PRIVACY_POLICY_URL", "").strip()
    if not _is_public_url(privacy_policy) or urlparse(privacy_policy).scheme != "https":
        invalid.append("HTTPS PRIVACY_POLICY_URL")
    if not app.config["SESSION_COOKIE_SECURE"]:
        invalid.append("SESSION_COOKIE_SECURE=true")
    if invalid:
        raise RuntimeError(
            "Production auth requires explicit public URL configuration: "
            + ", ".join(invalid)
            + ". Refusing to derive account links from the request Host header."
        )


_validate_production_public_urls()


def _error_response(
    message: str,
    code: str,
    status: int = 400,
    *,
    field_errors: Optional[Dict[str, str]] = None,
    retryable: bool = False,
    retry_after: Optional[int] = None,
    extra: Optional[dict] = None,
):
    """Return the stable error envelope consumed by the web client."""
    payload = {
        "error": message,
        "code": code,
        "request_id": getattr(g, "request_id", None),
        "retryable": retryable,
    }
    if field_errors:
        payload["field_errors"] = field_errors
    if extra:
        payload.update(extra)
    response = jsonify(payload)
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = str(max(0, int(retry_after)))
    return response


def _csrf_token() -> str:
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def _rotate_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    session[CSRF_SESSION_KEY] = token
    return token


def _csrf_payload() -> dict:
    if SUBMISSION_AUTH_DISABLED:
        return {}
    return {"csrf_token": _csrf_token()}


def _csrf_error():
    return _error_response(
        "Your session security token is missing or expired. Refresh the page, sign in again if prompted, and retry.",
        "csrf_required",
        403,
    )


def _csrf_protect_request():
    if request.method not in CSRF_MUTATING_METHODS:
        return None
    if SUBMISSION_AUTH_DISABLED:
        return None
    if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
        return None
    if not session.get("user_email"):
        return None
    expected = session.get(CSRF_SESSION_KEY)
    supplied = request.headers.get(CSRF_HEADER) or request.form.get("csrf_token")
    if not expected or not supplied or not hmac.compare_digest(str(expected), str(supplied)):
        return _csrf_error()
    return None


def _safe_next_path(value: Optional[str]) -> str:
    candidate = value or ""
    if (
        not re.match(r"^/[^/]", candidate)
        or "\\" in candidate
        or re.search(r"[\x00-\x1f\x7f]", candidate)
    ):
        return "/submit"
    return candidate


def _auth_json_body():
    """Read a small JSON object for an authentication endpoint."""
    if not request.is_json:
        return None, _error_response(
            "This account action requires a JSON request body.",
            "auth_json_required",
            415,
        )
    data = request.get_json(silent=True)
    if data is None:
        return None, _error_response(
            "The account request contains malformed JSON. Correct the request body and try again.",
            "invalid_auth_json",
            400,
        )
    if not isinstance(data, dict):
        return None, _error_response(
            "The account request must be a JSON object with named fields.",
            "invalid_auth_json_object",
            400,
        )
    return data, None


def _optional_json_object():
    """Parse an optional JSON object for authenticated action endpoints."""
    if not request.get_data(cache=True):
        return {}, None
    if not request.is_json:
        return None, _error_response(
            "This action requires a JSON request body.",
            "json_required",
            415,
        )
    data = request.get_json(silent=True)
    if data is None:
        return None, _error_response(
            "The action request contains malformed JSON.",
            "invalid_json",
            400,
        )
    if not isinstance(data, dict):
        return None, _error_response(
            "The action request must be a JSON object with named fields.",
            "invalid_json_object",
            400,
        )
    return data, None


def _auth_string(data: dict, field: str, *aliases: str):
    value = data.get(field)
    if value in (None, ""):
        for alias in aliases:
            alias_value = data.get(alias)
            if alias_value not in (None, ""):
                value = alias_value
                break
    if value is None:
        return "", None
    if not isinstance(value, str):
        return "", _error_response(
            f"{field.replace('_', ' ').capitalize()} must be text.",
            "invalid_auth_field_type",
            400,
            field_errors={field: "Enter text in this field."},
        )
    return value, None


def _establish_user_session(email: str) -> str:
    account = get_user(email)
    if not account or not account.get("email_verified"):
        raise RuntimeError("verified account unavailable")
    session.clear()
    session["user_email"] = account["email"]
    session[AUTH_SESSION_VERSION_KEY] = int(account.get("session_version") or 0)
    csrf_token = _rotate_csrf_token()
    session.permanent = True
    return csrf_token


def _oauth_serializer():
    from itsdangerous import URLSafeTimedSerializer
    return URLSafeTimedSerializer(app.secret_key, salt="vista-oauth-state")


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _oauth_config(provider: str):
    config = OAUTH_PROVIDERS.get(provider)
    if not config:
        return None
    return {
        **config,
        "authorize_url": config["authorize_url"].format(tenant=MICROSOFT_TENANT_ID),
        "token_url": config["token_url"].format(tenant=MICROSOFT_TENANT_ID),
    }


def _oauth_redirect_uri(provider: str) -> str:
    base_url = (
        os.getenv("OAUTH_REDIRECT_BASE_URL", "").strip()
        or os.getenv("API_BASE_URL", "").strip()
        or request.host_url.rstrip("/")
    )
    return f"{base_url}{url_for('auth_oauth_callback', provider=provider)}"


def _frontend_base_url() -> str:
    return (
        os.getenv("FRONTEND_BASE_URL", "").strip()
        or request.host_url.rstrip("/")
    ).rstrip("/")


def _api_base_url() -> str:
    return (
        os.getenv("API_BASE_URL", "").strip()
        or os.getenv("OAUTH_REDIRECT_BASE_URL", "").strip()
        or request.host_url.rstrip("/")
    ).rstrip("/")


def _frontend_url(path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"{_frontend_base_url()}{path}"


def _login_redirect(next_path: str, fragment: Optional[Dict[str, str]] = None):
    session.pop(OAUTH_PKCE_SESSION_KEY, None)
    target = _frontend_url(f"/login?{urlencode({'next': _safe_next_path(next_path)})}")
    if fragment:
        target = f"{target}#{urlencode(fragment)}"
    response = redirect(target)
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


def current_user_email() -> Optional[str]:
    """Return the signed-in, verified account email from the session, or None."""
    email = session.get("user_email")
    if not email:
        return None
    user = get_user(email)
    session_version = session.get(AUTH_SESSION_VERSION_KEY)
    try:
        version_matches = bool(
            user
            and session_version is not None
            and int(session_version) == int(user.get("session_version") or 0)
        )
    except (TypeError, ValueError):
        version_matches = False
    if (
        not user
        or not user["email_verified"]
        or not version_matches
    ):
        session.clear()
        return None
    return user["email"]


def _is_admin_email(email: Optional[str]) -> bool:
    return bool(email and email.lower() in ADMIN_EMAILS)


def admin_required(func):
    """Require a signed-in admin account listed in ADMIN_EMAILS."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            email = TEST_SUBMISSION_USER if SUBMISSION_AUTH_DISABLED else current_user_email()
        except Exception as exc:
            logger.error(
                "Administrator session lookup failed: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            return _error_response(
                "Administrator access could not be checked because the account service is temporarily unavailable. No action was taken; retry shortly.",
                "session_check_unavailable",
                503,
                retryable=True,
            )
        if not email:
            return _error_response(
                "Your sign-in session is missing or has expired. Sign in again, then retry this action.",
                "auth_required",
                401,
            )
        if not _is_admin_email(email):
            return _error_response(
                "This action requires an administrator account. Sign in with an address listed in ADMIN_EMAILS.",
                "admin_required",
                403,
            )
        g.user_id = email
        return func(*args, **kwargs)

    return wrapper


def _identity_key() -> str:
    """Rate-limit key: the signed-in account when present, else the client IP."""
    email = session.get("user_email")
    return ("user:" + email) if email else ("ip:" + get_remote_address())


def submission_auth_required(func):
    """Require a signed-in, email-verified account (unless disabled for local testing)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if SUBMISSION_AUTH_DISABLED:
            if not getattr(g, "user_id", None):
                g.user_id = TEST_SUBMISSION_USER
            return func(*args, **kwargs)
        try:
            email = current_user_email()
        except Exception as exc:
            logger.error(
                "Submission session lookup failed: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            return _error_response(
                "Your account session could not be checked because the account service is temporarily unavailable. No submission data was changed; retry shortly.",
                "session_check_unavailable",
                503,
                retryable=True,
            )
        if not email:
            return _error_response(
                "Your sign-in session is missing or has expired. Sign in again, then retry this action.",
                "auth_required",
                401,
            )
        g.user_id = email
        return func(*args, **kwargs)

    return wrapper


def _enforce_quota(task_id: str, model_name: str, model_id: str):
    """Reserve one rolling daily slot for this account and benchmark.

    Returns (submission_id, error_response). If error_response is not None the
    caller must return it (HTTP 429). Skipped when submission auth is disabled
    (local testing).
    """
    email = getattr(g, "user_id", None)
    if SUBMISSION_AUTH_DISABLED:
        result = try_consume_quota(
            TEST_SUBMISSION_USER,
            task_id,
            model_name,
            model_id=model_id,
            request_id=getattr(g, "request_id", None),
            ip=get_remote_address(),
            limit=1_000_000,
        )
        return result.submission_id, None
    if not email:
        return None, None
    result = try_consume_quota(
        email, task_id, model_name, model_id=model_id,
        request_id=getattr(g, "request_id", None), ip=get_remote_address(),
    )
    if result.allowed:
        return result.submission_id, None
    reset_txt = result.reset_at.strftime("%H:%M UTC") if result.reset_at else "later"
    resp = _error_response(
        f"You have used the {result.limit} submission available for this benchmark in the current 24-hour window. You can submit to this benchmark again after {reset_txt}.",
        "quota_exceeded",
        429,
        retryable=True,
        retry_after=result.retry_after,
        extra={
            "limit": result.limit,
            "used": result.used,
            "remaining": 0,
            "reset_at": result.reset_at.isoformat() if result.reset_at else None,
        },
    )
    return None, resp


def _stored_predictions(answer_rows: list[dict]) -> Dict[str, Dict[str, str]]:
    predictions: Dict[str, Dict[str, str]] = {}
    for row in answer_rows:
        condition = str(row.get("condition") or "standard").strip().lower()
        question_id = str(row.get("question_id") or "").strip()
        predictions.setdefault(condition, {})[question_id] = str(row.get("answer") or "")
    return predictions


def _rescore_stored_submission(score_submission_id: str):
    try:
        stored = get_submission_for_rescore(score_submission_id)
    except Exception as exc:
        logger.error(
            "Stored submission lookup failed for %s: %s",
            score_submission_id,
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return None, {
            "error": "The stored submission could not be read from the audit database.",
            "code": "submission_storage_unavailable",
        }
    if stored is None:
        return None, {"error": "The submission could not be found.", "code": "submission_not_found"}
    task_id = stored.get("task_id")
    if task_id not in task_scorers:
        return None, {
            "error": "This submission references a benchmark that is no longer configured.",
            "code": "submission_task_unavailable",
        }
    try:
        created_at = stored.get("created_at")
        submitted_at = (
            datetime.fromisoformat(created_at)
            if created_at
            else datetime.now(timezone.utc)
        )
        if task_id == "spatial":
            artifacts = stored.get("artifacts") or {}
            required_artifacts = set(SPATIAL_PUBLIC_ARTIFACT_NAMES)
            if set(artifacts) != required_artifacts:
                raise ValueError(
                    "Stored spatial evidence artifacts are missing or incomplete"
                )
            archive_submission, archive_manifest, archive_report = (
                read_spatial_submission_archive(
                    artifacts[SPATIAL_SUBMISSION_ARCHIVE_NAME]
                )
            )
            for artifact_name, archive_value in (
                (SPATIAL_SUBMISSION_MEMBER, archive_submission),
                (SPATIAL_MANIFEST_MEMBER, archive_manifest),
                (SPATIAL_REPORT_MEMBER, archive_report),
            ):
                if not hmac.compare_digest(artifacts[artifact_name], archive_value):
                    raise ValueError(
                        f"Stored spatial artifact {artifact_name} does not match the retained archive"
                    )
            stored_contract = stored.get("spatial_contract")
            if stored_contract:
                benchmark_manifest_source = stored_contract["manifest"]
                template_source = stored_contract["template"]
                questions_source = stored_contract["questions"]
            else:
                # Compatibility for submissions accepted before contract snapshots
                # were introduced. Manifest hash validation below prevents using a
                # different live contract for an older run.
                benchmark_manifest_source = SPATIAL_MANIFEST_FILE
                template_source = TASKS["spatial"]["paths"]["template_jsonl"]
                questions_source = TASKS["spatial"]["paths"]["questions_jsonl"]
            records, computed_report, _manifest = parse_spatial_evidence(
                archive_submission,
                benchmark_manifest_source,
                template_source,
                questions_source,
            )
            run_metadata = validate_run_manifest(
                archive_manifest,
                archive_submission,
                archive_report,
                stored.get("model_name") or "",
                records,
                benchmark_manifest_source,
            )
            report = validate_spatial_report(
                archive_report,
                stored.get("model_name") or "",
                computed_report,
            )
            score = build_spatial_task_score(
                report,
                stored.get("model_name") or "",
                stored.get("model_meta") or {},
                run_metadata,
                submission_id=stored["submission_id"],
            )
        else:
            score = task_scorers[task_id].score_predictions(
                _stored_predictions(stored.get("answers") or []),
                model_name=stored.get("model_name") or "",
                model_meta=stored.get("model_meta") or {},
                submission_metadata=stored.get("metadata") or None,
            )
        score.submission_id = stored["submission_id"]
        score.model_id = stored.get("model_id")
        score.submitted_at = submitted_at
        record = score.to_dict()
        update_submission_score(
            score.submission_id,
            score_json=record,
            model_meta=score.model_meta,
        )
        return (score, stored), None
    except Exception as exc:
        logger.error(
            "Rescore failed for submission %s: %s",
            score_submission_id,
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return None, {
            "error": "The submission could not be rescored because its stored answers or grading data are unavailable.",
            "code": "rescore_failed",
        }


def _refresh_public_model_task(model_id: str, task_id: str) -> dict:
    """Publish the newest visible run for one model/benchmark pair."""
    latest_submission_id = latest_visible_scored_submission_id(model_id, task_id)
    if latest_submission_id is None:
        removed = leaderboard_store.remove_model_task(model_id, task_id)
        return {
            "published": False,
            "submission_id": None,
            "removed": removed,
            "score": None,
        }

    result, error = _rescore_stored_submission(latest_submission_id)
    if error:
        raise RuntimeError(error.get("error") or "Latest visible submission could not be rescored")
    score, stored = result
    leaderboard_store.add_result(score, submitted_by=stored.get("user_email"))
    return {
        "published": True,
        "submission_id": latest_submission_id,
        "removed": False,
        "score": score,
    }


def _rollback_moderation(submission_id: str, previous: dict) -> None:
    """Restore the prior moderation state after a cache publication failure."""
    set_moderation_status(
        submission_id,
        previous.get("moderation_status") or "visible",
        reason=previous.get("moderation_reason"),
        moderated_by=previous.get("moderated_by"),
    )


def _set_moderation_and_refresh(
    submission_id: str,
    moderation_status: str,
    *,
    reason: Optional[str],
    moderated_by: Optional[str],
) -> tuple[Optional[dict], Optional[dict]]:
    """Apply moderation and keep the public cache on the latest visible run."""
    row = set_moderation_status(
        submission_id,
        moderation_status,
        reason=reason,
        moderated_by=moderated_by,
    )
    if row is None:
        return None, None
    previous = row.pop("previous_moderation", {})
    try:
        refresh = _refresh_public_model_task(row.get("model_id"), row.get("task_id"))
    except Exception:
        _rollback_moderation(submission_id, previous)
        try:
            _refresh_public_model_task(row.get("model_id"), row.get("task_id"))
        except Exception as rollback_error:
            logger.critical(
                "Moderation rollback for %s could not republish the prior state: %s",
                submission_id,
                rollback_error,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
        raise
    return row, refresh


def _form_value(name: str) -> str:
    return str(request.form.get(name) or "").strip()


def _word_count(value: str) -> int:
    return len(re.findall(r"\b\S+\b", value or ""))


def _query_limit(default: int = DEFAULT_LEADERBOARD_LIMIT):
    raw_limit = request.args.get("limit", str(default))
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return None, _error_response(
            f"The limit must be a whole number from {MIN_LEADERBOARD_LIMIT} to {MAX_LEADERBOARD_LIMIT}.",
            "invalid_limit",
            400,
        )

    if limit < MIN_LEADERBOARD_LIMIT or limit > MAX_LEADERBOARD_LIMIT:
        return None, _error_response(
            f"The limit must be between {MIN_LEADERBOARD_LIMIT} and {MAX_LEADERBOARD_LIMIT}.",
            "invalid_limit",
            400,
        )

    return limit, None


def _valid_model_name(model_name: str) -> bool:
    if not model_name or len(model_name) > MAX_MODEL_NAME_LENGTH:
        return False
    return not re.search(r"[\x00-\x1f\x7f]", model_name)


_MODEL_META_JSON_MAX_CHARS = 50_000
_MODEL_META_MAX_LENGTHS = {
    "organization": 200,
    "access": 80,
    "parameter_count": 80,
    "base_model": 200,
    "training_data": 5_000,
    "method_description": 10_000,
    "cot_used": 80,
    "prompt_template": 20_000,
    "changes_from_previous": 10_000,
    "paper_url": 500,
}
_MODEL_META_LABELS = {
    "organization": "Organisation",
    "access": "Open/closed source status",
    "parameter_count": "Parameter count",
    "base_model": "Base model",
    "training_data": "Training data",
    "method_description": "Method description",
    "cot_used": "CoT usage",
    "prompt_template": "Prompt template",
    "changes_from_previous": "Changes from previous submission",
    "paper_url": "Paper / arXiv link",
}


def _valid_external_http_url(value: str) -> bool:
    """Accept a complete HTTP(S) URL without credentials or whitespace."""
    if not value or re.search(r"\s|[\x00-\x1f\x7f]", value):
        return False
    try:
        parsed = urlparse(value)
        _ = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.netloc
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
    )


def _submission_model_meta(task_id: str, registered_meta: Optional[Dict] = None):
    meta = {}
    raw_meta = request.form.get("model_meta")
    if raw_meta:
        if len(raw_meta) > _MODEL_META_JSON_MAX_CHARS:
            return None, f"model_meta is too large (max {_MODEL_META_JSON_MAX_CHARS} characters)"
        try:
            parsed = json.loads(raw_meta)
            if not isinstance(parsed, dict):
                return None, "model_meta must be a JSON object with named fields"
            unsupported = sorted(set(parsed) - set(_MODEL_META_MAX_LENGTHS))
            if unsupported:
                shown = ", ".join(unsupported[:5])
                suffix = "" if len(unsupported) <= 5 else ", and more"
                return None, f"model_meta contains unsupported fields: {shown}{suffix}"
            invalid_types = sorted(
                key
                for key, value in parsed.items()
                if value is not None and not isinstance(value, (str, int, float, bool))
            )
            if invalid_types:
                return None, (
                    "model_meta fields must contain text or scalar values: "
                    + ", ".join(invalid_types[:5])
                )
            meta.update(parsed)
        except json.JSONDecodeError:
            return None, "model_meta must be valid JSON"

    field_map = {
        "organization": "organization",
        "model_access": "access",
        "parameter_count": "parameter_count",
        "base_model": "base_model",
        "training_data": "training_data",
        "method_description": "method_description",
        "cot_used": "cot_used",
        "prompt_template": "prompt_template",
        "changes_from_previous": "changes_from_previous",
        "paper_url": "paper_url",
    }
    for form_name, meta_name in field_map.items():
        value = _form_value(form_name)
        if value:
            meta[meta_name] = value

    if registered_meta:
        for key in (
            "organization",
            "access",
            "parameter_count",
            "base_model",
            "training_data",
            "paper_url",
        ):
            if key in registered_meta:
                meta[key] = registered_meta[key]

    for key, max_len in _MODEL_META_MAX_LENGTHS.items():
        if key not in meta:
            continue
        value = str(meta.get(key) or "").strip()
        if len(value) > max_len:
            return None, f"{_MODEL_META_LABELS[key]} is too long (max {max_len} characters)"
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            return None, f"{_MODEL_META_LABELS[key]} contains invalid control characters"
        meta[key] = value

    if task_id == "spatial":
        meta["cot_used"] = "mixed"
        meta["prompt_template"] = (
            "Official spatial harness non-CoT and CoT prompts, verified by run manifest."
        )

    required = {
        "organization": "organisation",
        "access": "open/closed source status",
        "method_description": "method description",
        "cot_used": "CoT usage",
        "prompt_template": "prompt template",
        "changes_from_previous": "changes from previous submission",
    }
    missing = [label for key, label in required.items() if not str(meta.get(key) or "").strip()]
    if missing:
        return None, "Missing required metadata: " + ", ".join(missing)
    if _word_count(str(meta.get("method_description") or "")) < 100:
        return None, "Method description must be at least 100 words"
    if _word_count(str(meta.get("changes_from_previous") or "")) < 50:
        return None, "Changes from previous submission must be at least 50 words"
    paper_url = str(meta.get("paper_url") or "").strip()
    if paper_url and not _valid_external_http_url(paper_url):
        return None, "Paper / arXiv link must be a complete http:// or https:// URL without embedded credentials"

    meta["org"] = str(meta.get("organization") or "").strip()
    meta["type"] = str(meta.get("access") or "").strip()
    meta["submission_track"] = task_id
    return meta, None


_MODEL_ACCESS_VALUES = {"open", "open_weights", "closed", "research"}
_REGISTERED_MODEL_FIELDS = {
    "model_name": (MAX_MODEL_NAME_LENGTH, True),
    "organization": (_MODEL_META_MAX_LENGTHS["organization"], True),
    "access": (_MODEL_META_MAX_LENGTHS["access"], True),
    "parameter_count": (_MODEL_META_MAX_LENGTHS["parameter_count"], False),
    "paper_url": (_MODEL_META_MAX_LENGTHS["paper_url"], False),
}


def _registered_model_payload(data: dict):
    """Validate canonical metadata stored once for a model identity."""
    values = {}
    field_errors = {}
    for field, (max_length, required) in _REGISTERED_MODEL_FIELDS.items():
        raw = data.get(field)
        if raw is not None and not isinstance(raw, str):
            field_errors[field] = "Enter text for this field."
            continue
        value = str(raw or "").strip()
        if required and not value:
            field_errors[field] = "This field is required."
        elif len(value) > max_length:
            field_errors[field] = f"Use no more than {max_length} characters."
        elif re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            field_errors[field] = "Remove invalid control characters."
        values[field] = value
    if values.get("model_name") and not _valid_model_name(values["model_name"]):
        field_errors["model_name"] = (
            f"Use 1 to {MAX_MODEL_NAME_LENGTH} printable characters."
        )
    if values.get("access") and values["access"] not in _MODEL_ACCESS_VALUES:
        field_errors["access"] = "Choose one of the available source status options."
    paper_url = values.get("paper_url") or ""
    if paper_url and not _valid_external_http_url(paper_url):
        field_errors["paper_url"] = (
            "Use a complete http:// or https:// URL without embedded credentials."
        )
    if field_errors:
        return None, field_errors
    return values, None


def _metadata_field_errors(message: str) -> Dict[str, str]:
    """Map metadata validation text back to the matching browser form field."""
    text = str(message or "")
    labels = {
        "Organisation": "organization",
        "Open/closed source status": "model_access",
        "Parameter count": "parameter_count",
        "Base model": "base_model",
        "Training data": "training_data",
        "Method description": "method_description",
        "CoT usage": "cot_used",
        "Prompt template": "prompt_template",
        "Changes from previous submission": "changes_from_previous",
        "Paper / arXiv link": "paper_url",
    }
    for label, field in labels.items():
        if text.startswith(label):
            return {field: text}
    if text.startswith("Missing required metadata:"):
        field_errors = {}
        missing = text.split(":", 1)[1].lower()
        aliases = {
            "organisation": "organization",
            "open/closed source status": "model_access",
            "base model": "base_model",
            "training data and fine-tuning information": "training_data",
            "method description": "method_description",
            "cot usage": "cot_used",
            "prompt template": "prompt_template",
            "changes from previous submission": "changes_from_previous",
        }
        for label, field in aliases.items():
            if label in missing:
                field_errors[field] = f"{label.capitalize()} is required."
        return field_errors
    return {}


def _finalize_submission_safely(submission_id: Optional[int], success: bool) -> None:
    if submission_id is None:
        return
    try:
        finalize_submission(submission_id, success)
    except Exception as exc:
        logger.error(
            "Failed to finalize submission audit row %s: %s",
            submission_id,
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )


def _submission_validation_response(
    exc: SubmissionValidationError,
    *,
    spatial_archive: bool = False,
):
    message = str(exc)
    field = str(exc.details.get("field") or "file")
    if spatial_archive:
        field = "file"
        correction = (
            "Rerun the current spatial harness and upload its unchanged "
            f"{SPATIAL_SUBMISSION_ARCHIVE_NAME} package."
        )
    else:
        correction = (
            "Correct the JSONL issue described above, then choose the updated file and submit again."
        )
    return _error_response(
        message,
        exc.code,
        400,
        field_errors={field: correction},
        extra={"validation": exc.to_dict()},
    )


def _ground_truth_bundle_health() -> tuple[str, dict]:
    """Verify visual-task question IDs and private answer IDs are one release."""
    tasks = {}
    synchronized = True
    for task_id in ("do_you_see_me", "minds_eye"):
        scorer = task_scorers.get(task_id)
        if scorer is None:
            continue
        ground_truth_ids = set(scorer.ground_truth)
        question_ids = set(scorer.questions)
        missing_question_ids = ground_truth_ids - question_ids
        unknown_question_ids = question_ids - ground_truth_ids
        task_synchronized = bool(ground_truth_ids) and bool(question_ids) and not (
            missing_question_ids or unknown_question_ids
        )
        synchronized = synchronized and task_synchronized
        tasks[task_id] = {
            "ground_truth_count": len(ground_truth_ids),
            "question_count": len(question_ids),
            "synchronized": task_synchronized,
            "missing_public_question_count": len(missing_question_ids),
            "unknown_public_question_count": len(unknown_question_ids),
        }

    revision_pinned = bool(
        re.fullmatch(r"[0-9a-fA-F]{40}", GROUND_TRUTHS_HF_REVISION or "")
    )
    revision_ready = (
        not PUBLIC_DEPLOYMENT
        or GROUND_TRUTHS_SOURCE != "hf"
        or revision_pinned
    )
    healthy = synchronized and revision_ready
    return ("healthy" if healthy else "unhealthy"), {
        "source": GROUND_TRUTHS_SOURCE,
        "hf_revision_pinned": revision_pinned,
        "revision_ready": revision_ready,
        "tasks": tasks,
    }


def _spatial_bundle_health() -> tuple[str, dict]:
    """Return readiness status for the Spatial benchmark bundle."""
    status, details = inspect_spatial_bundle(
        SPATIAL_MANIFEST_FILE,
        TASKS["spatial"]["paths"]["template_jsonl"],
        TASKS["spatial"]["paths"]["questions_jsonl"],
    )
    return status, {**details, "required": REQUIRE_OFFICIAL_SPATIAL}


def _email_delivery_health() -> tuple[str, dict]:
    """Return readiness status for verification/reset email delivery."""
    if SUBMISSION_AUTH_DISABLED:
        return "skipped", {
            "provider": None,
            "reason": "Submission auth is disabled for test deployment.",
            "production_ready": False,
        }

    acs_connection = bool(os.getenv("ACS_CONNECTION_STRING", "").strip())
    acs_endpoint = bool(os.getenv("ACS_ENDPOINT", "").strip())
    acs_sender = bool(os.getenv("ACS_SENDER_ADDRESS", "").strip())
    smtp_host = bool(os.getenv("SMTP_HOST", "").strip())
    smtp_from = bool((os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USERNAME", "")).strip())
    try:
        send_timeout = float(os.getenv("EMAIL_SEND_TIMEOUT_SECONDS", "30"))
        timeout_ready = send_timeout > 0
    except ValueError:
        send_timeout = None
        timeout_ready = False

    if acs_connection or acs_endpoint:
        errors = []
        try:
            from azure.communication.email import EmailClient

            if acs_connection:
                EmailClient.from_connection_string(
                    os.getenv("ACS_CONNECTION_STRING", "").strip()
                )
            else:
                endpoint = os.getenv("ACS_ENDPOINT", "").strip()
                parsed_endpoint = urlparse(endpoint)
                if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
                    errors.append("ACS_ENDPOINT must be an absolute HTTPS URL")
                from azure.identity import DefaultAzureCredential  # noqa: F401
        except (ImportError, ValueError):
            errors.append("Azure Communication Services email client configuration is invalid")
        if not timeout_ready:
            errors.append("EMAIL_SEND_TIMEOUT_SECONDS must be a positive number")
        ready = acs_sender and not errors
        return (
            "healthy" if ready else "unhealthy",
            {
                "provider": "azure_communication_services",
                "auth": "connection_string" if acs_connection else "managed_identity",
                "sender_configured": acs_sender,
                "client_ready": not errors,
                "send_timeout_seconds": send_timeout,
                "errors": errors,
                "production_ready": ready,
            },
        )
    if smtp_host:
        smtp_port_raw = os.getenv("SMTP_PORT", "587").strip()
        try:
            smtp_port = int(smtp_port_raw)
            valid_port = 1 <= smtp_port <= 65535
        except ValueError:
            smtp_port = None
            valid_port = False
        errors = []
        if not valid_port:
            errors.append("SMTP_PORT must be an integer from 1 to 65535")
        if not smtp_from:
            errors.append("SMTP_FROM or SMTP_USERNAME must provide a sender address")
        if not timeout_ready:
            errors.append("EMAIL_SEND_TIMEOUT_SECONDS must be a positive number")
        ready = not errors
        return ("healthy" if ready else "unhealthy"), {
            "provider": "smtp",
            "from_configured": smtp_from,
            "port": smtp_port,
            "send_timeout_seconds": send_timeout,
            "errors": errors,
            "production_ready": ready,
        }
    if AUTH_DEV_MODE:
        return "dev", {
            "provider": "log",
            "delivery": "logged",
            "production_ready": False,
        }
    return "unhealthy", {
        "provider": None,
        "error": "No ACS or SMTP email provider configured",
        "production_ready": False,
    }


def _queue_password_reset_email(email: str, reset_url: Optional[str]) -> None:
    """Queue reset delivery so response timing does not reveal account existence."""
    request_id = getattr(g, "request_id", None)

    def deliver() -> None:
        if not reset_url:
            return
        try:
            delivery = send_password_reset_email(email, reset_url)
        except Exception as exc:
            logger.error(
                "Password reset email provider error: %s",
                exc,
                extra={"request_id": request_id},
                exc_info=True,
            )
            return
        if delivery != "sent":
            logger.error(
                "Password reset email could not be delivered",
                extra={"request_id": request_id},
            )

    _email_executor.submit(deliver)


def _database_storage_health() -> tuple[str, dict]:
    """Check SQLite integrity, WAL mode, and persistent storage writability."""
    database_urls = {
        "primary": DATABASE_URL,
        "auth": AUTH_DATABASE_URL,
        "submissions": SUBMISSION_DATABASE_URL,
    }
    databases = {}
    checked_paths = {}
    healthy = True
    for label, database_url in database_urls.items():
        url = make_url(database_url)
        if not url.drivername.startswith("sqlite"):
            databases[label] = {"driver": url.drivername, "status": "external"}
            continue
        if not url.database or url.database == ":memory:":
            databases[label] = {"driver": "sqlite", "status": "unhealthy"}
            healthy = False
            continue
        path = Path(url.database).expanduser().resolve()
        path_key = str(path)
        if path_key not in checked_paths:
            details = {
                "driver": "sqlite",
                "exists": path.exists(),
                "writable": path.exists() and os.access(path, os.W_OK),
                "journal_mode": None,
                "quick_check": None,
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            if path.exists():
                try:
                    with sqlite3.connect(
                        f"file:{path}?mode=rw",
                        uri=True,
                        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
                    ) as connection:
                        quick_check = connection.execute("PRAGMA quick_check").fetchone()
                        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
                    details["quick_check"] = quick_check[0] if quick_check else "failed"
                    details["journal_mode"] = journal_mode[0] if journal_mode else "unknown"
                except sqlite3.Error:
                    details["quick_check"] = "failed"
            details["status"] = (
                "healthy"
                if details["exists"]
                and details["writable"]
                and details["quick_check"] == "ok"
                and details["journal_mode"] == "wal"
                else "unhealthy"
            )
            checked_paths[path_key] = details
        databases[label] = checked_paths[path_key]
        healthy = healthy and databases[label]["status"] == "healthy"

    storage = {}
    for label, directory in {
        "data": DATA_DIR,
        "results": RESULTS_DIR,
        "backups": AUTO_BACKUP_DIR,
        "leaderboard_cache": LEADERBOARD_STORE_FILE.parent,
    }.items():
        exists = directory.exists()
        writable = exists and directory.is_dir() and os.access(directory, os.W_OK)
        storage[label] = {"exists": exists, "writable": writable}
        if label != "backups" or AUTO_BACKUP_ENABLED:
            healthy = healthy and writable

    try:
        disk = shutil.disk_usage(DATA_DIR)
        disk_details = {
            "free_bytes": disk.free,
            "free_percent": round((disk.free / disk.total) * 100, 2) if disk.total else 0,
            "minimum_free_bytes": MIN_FREE_DISK_BYTES,
            "minimum_free_percent": MIN_FREE_DISK_PERCENT,
        }
        low_space = (
            disk.free < MIN_FREE_DISK_BYTES
            or disk_details["free_percent"] < MIN_FREE_DISK_PERCENT
        )
        disk_details["low_space"] = low_space
        if low_space and PUBLIC_DEPLOYMENT:
            healthy = False
    except OSError:
        disk_details = {"free_bytes": None, "free_percent": None}
        healthy = False

    all_sqlite = all(
        make_url(database_url).drivername.startswith("sqlite")
        for database_url in database_urls.values()
    )
    single_worker_sqlite = not all_sqlite or WEB_CONCURRENCY == 1
    if PUBLIC_DEPLOYMENT and not single_worker_sqlite:
        healthy = False

    return ("healthy" if healthy else "unhealthy"), {
        "databases": databases,
        "storage": storage,
        "disk": disk_details,
        "sqlite_database": all_sqlite,
        "web_worker_count": WEB_CONCURRENCY,
        "single_worker_sqlite": single_worker_sqlite,
    }


def _auth_service_health() -> tuple[str, dict]:
    providers = {}
    incomplete = []
    for provider, config in OAUTH_PROVIDERS.items():
        client_id = bool(os.getenv(config["client_id_env"], "").strip())
        client_secret = bool(os.getenv(config["client_secret_env"], "").strip())
        configured = client_id and client_secret
        providers[provider] = {
            "configured": configured,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if client_id != client_secret:
            incomplete.append(provider)
    if SUBMISSION_AUTH_DISABLED:
        return "disabled", {
            "enabled": False,
            "providers": providers,
            "incomplete_providers": incomplete,
        }
    microsoft_tenant_ready = bool(MICROSOFT_TENANT_ID)
    verified_admins = get_verified_admin_emails(ADMIN_EMAILS)
    admin_ready = bool(ADMIN_EMAILS) and bool(verified_admins)
    healthy = (
        not incomplete
        and providers["microsoft"]["configured"]
        and microsoft_tenant_ready
        and not _secret_is_placeholder
        and len(app.secret_key or "") >= 32
        and (not PUBLIC_DEPLOYMENT or admin_ready)
    )
    return ("healthy" if healthy else "unhealthy"), {
        "enabled": True,
        "providers": providers,
        "incomplete_providers": incomplete,
        "microsoft_ready": providers["microsoft"]["configured"],
        "microsoft_tenant_ready": microsoft_tenant_ready,
        "session_secret_ready": not _secret_is_placeholder and len(app.secret_key or "") >= 32,
        "admin_addresses_configured": len(ADMIN_EMAILS),
        "verified_admin_accounts": len(verified_admins),
        "admin_ready": admin_ready,
    }


def _deployment_configuration_health() -> tuple[str, dict]:
    configured_urls = {
        "frontend": os.getenv("FRONTEND_BASE_URL", "").strip(),
        "api": os.getenv("API_BASE_URL", "").strip(),
        "oauth_redirect": os.getenv("OAUTH_REDIRECT_BASE_URL", "").strip(),
    }
    urls = {}
    for label, value in configured_urls.items():
        parsed = urlparse(value)
        local = (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}
        urls[label] = {
            "configured": bool(parsed.scheme and parsed.netloc),
            "scheme": parsed.scheme or None,
            "local": local,
            "secure": parsed.scheme == "https",
        }
    frontend_origin = _url_origin(configured_urls["frontend"])
    cors_ready = frontend_origin in {origin.rstrip("/") for origin in CORS_ORIGINS}
    frontend_site = _cookie_site(configured_urls["frontend"])
    api_site = _cookie_site(configured_urls["api"])
    same_site = bool(frontend_site and api_site and frontend_site == api_site)
    cookie_samesite = app.config["SESSION_COOKIE_SAMESITE"]
    credential_cookie_ready = same_site or cookie_samesite == "None"
    url_shapes_ready = all(
        details["configured"] and (details["secure"] or details["local"])
        for details in urls.values()
    )
    local_mode = all(details["local"] for details in urls.values())
    secure_cookie_ready = app.config["SESSION_COOKIE_SECURE"] or local_mode
    base_ready = (
        url_shapes_ready
        and cors_ready
        and secure_cookie_ready
        and credential_cookie_ready
    )
    public_ready = (
        base_ready
        and not local_mode
        and all(details["secure"] for details in urls.values())
        and app.config["SESSION_COOKIE_SECURE"]
    )
    privacy_policy_url = os.getenv("PRIVACY_POLICY_URL", "").strip()
    privacy_policy_ready = (
        _is_public_url(privacy_policy_url)
        and urlparse(privacy_policy_url).scheme == "https"
    )
    public_ready = public_ready and privacy_policy_ready
    mode_ready = public_ready if PUBLIC_DEPLOYMENT else local_mode
    healthy = base_ready and mode_ready
    return ("healthy" if healthy else "unhealthy"), {
        "urls": urls,
        "cors_contains_frontend": cors_ready,
        "secure_cookie": app.config["SESSION_COOKIE_SECURE"],
        "session_cookie_samesite": cookie_samesite,
        "frontend_api_same_site": same_site,
        "credential_cookie_ready": credential_cookie_ready,
        "mode": DEPLOYMENT_MODE,
        "detected_mode": "local" if local_mode else "public",
        "mode_matches_configuration": mode_ready,
        "public_deployment_ready": public_ready,
        "privacy_policy_ready": privacy_policy_ready,
    }


def _backup_health() -> tuple[str, dict]:
    return backup_scheduler.status()

# Initialize managers
gt_manager = GroundTruthManager()

# Three-task Visual Cognition / Spatial managers
task_scorers = {tid: TaskScorer(tid) for tid in TASKS}
leaderboard_store = LeaderboardStore()

# Initialise user auth DB + submission quota store
init_auth_db()
init_submission_db()
leaderboard_store.migrate_model_keys(list_registered_models())

backup_scheduler = BackupScheduler(
    enabled=AUTO_BACKUP_ENABLED,
    output_dir=AUTO_BACKUP_DIR,
    mirror_dir=AUTO_BACKUP_MIRROR_DIR,
    require_mirror=REQUIRE_OFFSITE_BACKUP,
    interval_hours=AUTO_BACKUP_INTERVAL_HOURS,
    retention_count=AUTO_BACKUP_RETENTION_COUNT,
    poll_seconds=AUTO_BACKUP_POLL_SECONDS,
    run_on_start=AUTO_BACKUP_RUN_ON_START,
)
if "pytest" not in sys.modules:
    backup_scheduler.start()
    atexit.register(backup_scheduler.stop)

# Generate request ID for tracking
@app.before_request
def before_request():
    """Generate request ID and log incoming request."""
    g.request_id = str(uuid.uuid4())
    g.start_time = datetime.now(timezone.utc)
    if (
        request.method in CSRF_MUTATING_METHODS
        and (
            request.path.startswith("/api/auth/")
            or request.path.rstrip("/") == "/api/models"
        )
    ):
        request.max_content_length = MAX_AUTH_REQUEST_BYTES
    if request.path.rstrip("/") == "/api/tasks/spatial/submit":
        request.max_content_length = MAX_SPATIAL_MULTIPART_BYTES
    csrf_response = _csrf_protect_request()
    if csrf_response is not None:
        return csrf_response

    # Log incoming request
    logger.debug(
        f"Incoming {request.method} {request.path}",
        extra={
            "request_id": g.request_id,
            "remote_addr": request.remote_addr,
            "user_agent": request.user_agent.string
        }
    )

@app.after_request
def after_request(response):
    """Log response and add headers."""
    if hasattr(g, 'start_time'):
        duration = (datetime.now(timezone.utc) - g.start_time).total_seconds()
        logger.debug(
            f"Response {response.status_code} ({duration:.3f}s)",
            extra={
                "request_id": g.request_id,
                "status_code": response.status_code,
                "duration_ms": duration * 1000
            }
        )

    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['X-Request-Id'] = getattr(g, 'request_id', '')
    private_response = (
        request.path.startswith("/api/auth/")
        or request.path.startswith("/api/submissions/")
        or request.path.startswith("/api/admin/")
        or request.path.rstrip("/") == "/api/models/mine"
    )
    if private_response:
        response.headers['Cache-Control'] = 'no-store, max-age=0'
        response.headers['Pragma'] = 'no-cache'

    return response

# (token auth removed — submissions authenticate via cookie sessions)

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors."""
    logger.warning(f"Bad request: {error}")
    return _error_response(
        "The request could not be understood. Check the entered values and try again.",
        "bad_request",
        400,
    )

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return _error_response(
        "The requested resource does not exist or is no longer available.",
        "not_found",
        404,
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 errors (file too large)."""
    logger.warning(f"Request entity too large from {request.remote_addr}")
    auth_request = request.path.startswith("/api/auth/")
    model_request = request.path.rstrip("/") == "/api/models"
    spatial_upload = request.path.rstrip("/") == "/api/tasks/spatial/submit"
    max_bytes = (
        MAX_AUTH_REQUEST_BYTES
        if auth_request or model_request
        else MAX_SPATIAL_ARCHIVE_BYTES
        if spatial_upload
        else app.config["MAX_CONTENT_LENGTH"]
    )
    max_mb = max(1, max_bytes // (1024 * 1024))
    if auth_request:
        return _error_response(
            f"The account request is too large. Keep it below {MAX_AUTH_REQUEST_BYTES // 1024} KB.",
            "auth_request_too_large",
            413,
        )
    if model_request:
        return _error_response(
            f"The model registration request is too large. Keep it below {MAX_AUTH_REQUEST_BYTES // 1024} KB.",
            "model_request_too_large",
            413,
        )
    file_label = "spatial ZIP package" if spatial_upload else "JSONL file"
    return _error_response(
        f"The uploaded file is too large. Choose a {file_label} smaller than {max_mb} MB.",
        "file_too_large",
        413,
        field_errors={"file": f"File must be smaller than {max_mb} MB."},
    )

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limiting."""
    logger.warning(f"Rate limit exceeded for {request.remote_addr}: {e.description}")
    try:
        retry_after = int(e.limit.limit.get_expiry())
    except (AttributeError, TypeError, ValueError):
        retry_after = None
    return _error_response(
        f"Too many requests were made in a short period ({e.description}). Wait a moment and try again.",
        "rate_limit_exceeded",
        429,
        retryable=True,
        retry_after=retry_after,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Return JSON instead of Flask's HTML method error page."""
    return _error_response(
        f"{request.method} is not supported for this endpoint. Check the action and try again.",
        "method_not_allowed",
        405,
    )

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    request_id = getattr(g, 'request_id', None)
    logger.error(f"Server error: {error}", extra={"request_id": request_id}, exc_info=True)
    return _error_response(
        "We could not complete this request because of an unexpected server error. Try again; if it continues, contact the administrator with the request reference shown below.",
        "internal_error",
        500,
        retryable=True,
    )

# Routes
# ------------------------------------------------------------------ auth
@app.route("/api/auth/register", methods=["POST"])
@limiter.limit("5 per hour")
def auth_register():
    """Register a new email/password account and email a verification link."""
    data, body_error = _auth_json_body()
    if body_error is not None:
        return body_error
    email_value, field_error = _auth_string(data, "email", "username")
    if field_error is not None:
        return field_error
    password, field_error = _auth_string(data, "password")
    if field_error is not None:
        return field_error
    email = normalize_email(email_value)
    if not is_valid_email(email):
        return _error_response(
            "Enter a complete email address, for example name@example.com.",
            "invalid_email",
            400,
            field_errors={
                "email": f"Enter a valid email address with no more than {MAX_EMAIL_LENGTH} characters."
            },
        )
    password_status = password_policy_status(password)
    if password_status == "too_short":
        return _error_response(
            f"Choose a password with at least {MIN_PASSWORD_LENGTH} characters.",
            "weak_password",
            400,
            field_errors={
                "password": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
            },
        )
    if password_status == "too_long":
        return _error_response(
            f"Choose a password with no more than {MAX_PASSWORD_LENGTH} characters.",
            "password_too_long",
            400,
            field_errors={
                "password": f"Password must be no more than {MAX_PASSWORD_LENGTH} characters."
            },
        )
    if password_status == "common":
        return _error_response(
            "This password is commonly guessed or closely related to this service. Choose a different passphrase.",
            "common_password",
            400,
            field_errors={
                "password": "Choose a less predictable passphrase that you do not use on another service."
            },
        )
    try:
        status, token = register_user(email, password)
    except Exception as exc:
        logger.error(
            "Account registration database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Account creation is temporarily unavailable. Your account was not created; wait a moment and try again.",
            "registration_unavailable",
            503,
            retryable=True,
        )
    if status == "invalid":
        return _error_response(
            "The email address or password does not meet the account requirements.",
            "invalid_registration",
            400,
        )
    if status == "exists":
        return _error_response(
            "An account with this email already exists. Sign in instead, or reset your password if needed.",
            "account_exists",
            409,
            field_errors={"email": "This email is already registered."},
        )
    verify_url = f"{_frontend_base_url()}/login#verify_token={token}"
    try:
        delivery = send_verification_email(email, verify_url)
    except Exception as exc:
        logger.error(
            "Verification email provider error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        delivery = "error"
    payload = {"status": "verification_sent", "email": email}
    # Dev convenience only: surface the link when there is no mail server AND
    # AUTH_DEV_MODE is explicitly enabled. Never exposed in a normal deployment.
    if delivery == "logged" and AUTH_DEV_MODE:
        payload["dev_verify_url"] = verify_url
    elif delivery != "sent":
        logger.error(
            "Account verification email could not be delivered",
            extra={"request_id": getattr(g, "request_id", None)},
        )
        return _error_response(
            "Your account was created, but we could not send the verification email. Use 'Resend verification email' from the sign-in screen; if delivery still fails, contact the leaderboard administrator.",
            "verification_delivery_failed",
            503,
            retryable=True,
            extra={"email": email, "account_created": status == "created"},
        )
    return jsonify(payload), 201


@app.route("/api/auth/verify", methods=["GET", "POST"])
@limiter.limit("30 per hour")
def auth_verify():
    """Confirm an email token; only the JSON flow establishes a session."""
    if request.method == "POST":
        data, body_error = _auth_json_body()
        if body_error is not None:
            return body_error
        token, field_error = _auth_string(data, "token")
        if field_error is not None:
            return field_error
        try:
            email = verify_email(token)
        except Exception as exc:
            logger.error(
                "Email verification database error: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            return _error_response(
                "Email verification is temporarily unavailable. Your link was not consumed; try it again shortly.",
                "verification_unavailable",
                503,
                retryable=True,
            )
        if not email:
            return _error_response(
                "This verification link is invalid, expired, or has already been used. Request a new link from the sign-in screen.",
                "invalid_verification_token",
                400,
            )
        try:
            csrf_token = _establish_user_session(email)
        except Exception as exc:
            logger.error(
                "Verified account session creation failed: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            return _error_response(
                "Your email was verified, but a session could not be created. Sign in with your email and password.",
                "verification_session_failed",
                503,
                retryable=True,
            )
        return jsonify({
            "status": "verified",
            "email": email,
            "csrf_token": csrf_token,
        }), 200

    # Backward-compatible redirect for links issued before fragment-based
    # verification was introduced.
    try:
        email = verify_email(request.args.get("token") or "")
    except Exception as exc:
        logger.error(
            "Email verification database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return redirect(_frontend_url("/login#" + urlencode({
            "verify_error": "Email verification is temporarily unavailable. Your link was not consumed; try it again shortly.",
        })))
    if not email:
        return redirect(_frontend_url("/login#" + urlencode({"verify_error": "This verification link is invalid or has expired."})))
    # A GET verification link can be opened from an untrusted page. Verifying
    # the address is safe, but logging that browser into the token owner's
    # account would create a login-CSRF path. Current links use the JSON POST
    # flow above, which is protected by the browser's CORS preflight.
    session.clear()
    return redirect(_frontend_url("/login#" + urlencode({"verified": "1"})))


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("10 per hour")
def auth_login():
    """Sign in with email + password. Sets a session cookie."""
    data, body_error = _auth_json_body()
    if body_error is not None:
        return body_error
    email_value, field_error = _auth_string(data, "email", "username")
    if field_error is not None:
        return field_error
    password, field_error = _auth_string(data, "password")
    if field_error is not None:
        return field_error
    email = normalize_email(email_value)
    if not email or not password:
        fields = {}
        if not email:
            fields["email"] = "Email is required."
        if not password:
            fields["password"] = "Password is required."
        return _error_response(
            "Enter both your email address and password.",
            "missing_credentials",
            400,
            field_errors=fields,
        )
    if not is_valid_email(email):
        return _error_response(
            "Enter the complete email address used for your account.",
            "invalid_email",
            400,
            field_errors={"email": "Enter a valid email address."},
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        return _error_response(
            f"The entered password exceeds the {MAX_PASSWORD_LENGTH} character limit.",
            "password_too_long",
            400,
            field_errors={
                "password": f"Password must be no more than {MAX_PASSWORD_LENGTH} characters."
            },
        )
    try:
        status = login_user(email, password)
    except Exception as exc:
        logger.error(
            "Sign-in database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Sign-in is temporarily unavailable. Your credentials were not changed; wait a moment and try again.",
            "login_unavailable",
            503,
            retryable=True,
        )
    if status == "unverified":
        return _error_response(
            "Your email address has not been verified. Open the verification link in your inbox or request a new link below.",
            "unverified",
            403,
            extra={"email": email},
        )
    if status != "ok":
        return _error_response(
            "The email address or password is incorrect. Check both fields or reset your password.",
            "invalid_credentials",
            401,
        )
    try:
        csrf_token = _establish_user_session(email)
    except Exception as exc:
        logger.error(
            "Sign-in session creation failed: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Your credentials were accepted, but the session could not be created. Wait a moment and sign in again.",
            "session_creation_failed",
            503,
            retryable=True,
        )
    return jsonify({"email": email, "csrf_token": csrf_token}), 200


@app.route("/api/auth/forgot-password", methods=["POST"])
@limiter.limit("5 per hour")
def auth_forgot_password():
    """Send a password reset link if the account exists."""
    data, body_error = _auth_json_body()
    if body_error is not None:
        return body_error
    email_value, field_error = _auth_string(data, "email")
    if field_error is not None:
        return field_error
    email = normalize_email(email_value)
    if not is_valid_email(email):
        return _error_response(
            "Enter the complete email address used for your account.",
            "invalid_email",
            400,
            field_errors={"email": "Enter a valid email address."},
        )

    email_status, _email_details = _email_delivery_health()
    if email_status == "unhealthy":
        return _error_response(
            "Password reset email is temporarily unavailable. No reset email was sent; contact the leaderboard administrator or try again later.",
            "email_delivery_unavailable",
            503,
            retryable=True,
        )

    try:
        status, token = request_password_reset(email)
    except Exception as exc:
        logger.error(
            "Password reset database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Password reset is temporarily unavailable. No account changes were made; wait a moment and try again.",
            "password_reset_unavailable",
            503,
            retryable=True,
        )
    payload = {"status": "reset_requested", "email": email}
    reset_url = (
        f"{_frontend_base_url()}/login#reset_token={token}"
        if status == "sent" and token
        else None
    )
    if AUTH_DEV_MODE and reset_url:
        try:
            delivery = send_password_reset_email(email, reset_url)
        except Exception as exc:
            logger.error(
                "Password reset email provider error: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            delivery = "error"
        if delivery == "logged":
            payload["dev_reset_url"] = reset_url
        elif delivery != "sent":
            logger.error(
                "Password reset email could not be delivered",
                extra={"request_id": getattr(g, "request_id", None)},
            )
    else:
        _queue_password_reset_email(email, reset_url)
    return jsonify(payload), 200


@app.route("/api/auth/reset-password", methods=["POST"])
@limiter.limit("10 per hour")
def auth_reset_password():
    """Consume a password reset token and set a new password."""
    data, body_error = _auth_json_body()
    if body_error is not None:
        return body_error
    token, field_error = _auth_string(data, "token")
    if field_error is not None:
        return field_error
    password, field_error = _auth_string(data, "password")
    if field_error is not None:
        return field_error
    try:
        status = reset_password(token, password)
    except Exception as exc:
        logger.error(
            "Password update database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "We could not update your password because the account service is temporarily unavailable. Your existing password is unchanged; retry shortly.",
            "password_update_unavailable",
            503,
            retryable=True,
        )
    if status == "weak":
        return _error_response(
            f"Choose a password with at least {MIN_PASSWORD_LENGTH} characters.",
            "weak_password",
            400,
            field_errors={
                "password": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
            },
        )
    if status == "too_long":
        return _error_response(
            f"Choose a password with no more than {MAX_PASSWORD_LENGTH} characters.",
            "password_too_long",
            400,
            field_errors={
                "password": f"Password must be no more than {MAX_PASSWORD_LENGTH} characters."
            },
        )
    if status == "common":
        return _error_response(
            "This password is commonly guessed or closely related to this service. Choose a different passphrase.",
            "common_password",
            400,
            field_errors={
                "password": "Choose a less predictable passphrase that you do not use on another service."
            },
        )
    if status != "ok":
        return _error_response(
            "This password-reset link is invalid, expired, or has already been used. Request a new reset link from the sign-in screen.",
            "invalid_reset_token",
            400,
        )
    session.clear()
    return jsonify({"status": "password_reset"}), 200


@app.route("/api/auth/logout", methods=["POST"])
@limiter.limit("30 per minute", key_func=_identity_key)
def auth_logout():
    """Clear the session."""
    session.clear()
    return jsonify({"status": "ok"}), 200


@app.route("/api/auth/me", methods=["GET"])
@limiter.limit("120 per minute", key_func=_identity_key)
def auth_me():
    """Return the current signed-in account, if any."""
    if SUBMISSION_AUTH_DISABLED:
        return jsonify({
            "authenticated": True,
            "email": TEST_SUBMISSION_USER,
            "quota": None,
            "auth_disabled": True,
            "is_admin": _is_admin_email(TEST_SUBMISSION_USER),
            "email_verified": True,
            "auth_provider": "development",
            "created_at": None,
        }), 200

    try:
        email = current_user_email()
    except Exception as exc:
        logger.error(
            "Account session lookup failed: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Your account session could not be checked because the account service is temporarily unavailable. Refresh and try again.",
            "session_check_unavailable",
            503,
            retryable=True,
        )
    if not email:
        session.pop(CSRF_SESSION_KEY, None)
        return jsonify({"authenticated": False}), 200
    try:
        account = get_user(email)
    except Exception as exc:
        logger.error(
            "Account profile lookup failed: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Your account is signed in, but its profile details could not be loaded. Refresh and try again.",
            "account_profile_unavailable",
            503,
            retryable=True,
        )
    if not account:
        session.pop("user_email", None)
        session.pop(CSRF_SESSION_KEY, None)
        return jsonify({"authenticated": False}), 200
    try:
        quota = quota_status(email)
    except Exception as exc:
        logger.error(
            "Submission quota lookup failed: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "Your account is signed in, but its submission quota could not be checked. Refresh before uploading a model.",
            "quota_status_unavailable",
            503,
            retryable=True,
        )
    return jsonify({
        "authenticated": True,
        "email": email,
        "quota": quota,
        "is_admin": _is_admin_email(email),
        "email_verified": bool(account.get("email_verified")),
        "auth_provider": account.get("auth_provider") or "password",
        "created_at": account.get("created_at"),
        **_csrf_payload(),
    }), 200


@app.route("/api/auth/resend", methods=["POST"])
@limiter.limit("5 per hour")
def auth_resend():
    """Resend the verification link for an unverified account."""
    data, body_error = _auth_json_body()
    if body_error is not None:
        return body_error
    email_value, field_error = _auth_string(data, "email")
    if field_error is not None:
        return field_error
    email = normalize_email(email_value)
    if not is_valid_email(email):
        return _error_response(
            "Enter the complete email address used for your account.",
            "invalid_email",
            400,
            field_errors={"email": "Enter a valid email address."},
        )
    email_status, _email_details = _email_delivery_health()
    if email_status == "unhealthy":
        return _error_response(
            "Verification email is temporarily unavailable. No message was sent; contact the leaderboard administrator or try again later.",
            "email_delivery_unavailable",
            503,
            retryable=True,
        )
    try:
        status, token = resend_verification(email)
    except Exception as exc:
        logger.error(
            "Verification resend database error: %s",
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _error_response(
            "We could not request a new verification link because the account service is temporarily unavailable. Try again shortly.",
            "verification_resend_unavailable",
            503,
            retryable=True,
        )
    if status == "resent" and token:
        verify_url = f"{_frontend_base_url()}/login#verify_token={token}"
        try:
            delivery = send_verification_email(email, verify_url)
        except Exception as exc:
            logger.error(
                "Verification resend provider error: %s",
                exc,
                extra={"request_id": getattr(g, "request_id", None)},
                exc_info=True,
            )
            delivery = "error"
        if delivery == "logged" and AUTH_DEV_MODE:
            return jsonify({
                "status": "verification_sent",
                "email": email,
                "dev_verify_url": verify_url,
            }), 200
        if delivery != "sent":
            logger.error(
                "Verification resend email could not be delivered",
                extra={"request_id": getattr(g, "request_id", None)},
            )
    # Generic response to avoid leaking which emails exist / are verified.
    return jsonify({"status": "verification_requested", "email": email}), 200


@app.route("/api/auth/providers", methods=["GET"])
@limiter.limit("60 per minute")
def auth_providers():
    """List OAuth providers that are fully configured (client id + secret set).

    The web UI uses this to only show sign-in buttons for providers that will
    actually work, instead of rendering buttons that redirect back with a
    "not configured" error.
    """
    available = [
        {"id": pid, "label": cfg["label"]}
        for pid, cfg in OAUTH_PROVIDERS.items()
        if os.getenv(cfg["client_id_env"], "").strip()
        and os.getenv(cfg["client_secret_env"], "").strip()
    ]
    return jsonify({"providers": available}), 200


@app.route("/api/auth/oauth/<provider>", methods=["GET"])
@limiter.limit("20 per hour")
def auth_oauth_start(provider):
    """Start an OAuth login/register flow for a configured identity provider."""
    provider = provider.strip().lower()
    config = _oauth_config(provider)
    next_path = _safe_next_path(request.args.get("next"))
    if not config:
        return _login_redirect(next_path, {"oauth_error": "Unsupported sign-in provider."})
    client_id = os.getenv(config["client_id_env"], "").strip()
    if not client_id or not os.getenv(config["client_secret_env"], "").strip():
        return _login_redirect(next_path, {"oauth_error": f"{config['label']} sign-in is not configured."})
    nonce = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(64)
    session[OAUTH_PKCE_SESSION_KEY] = code_verifier
    state = _oauth_serializer().dumps({"provider": provider, "next": next_path, "nonce": nonce})
    params = {
        "client_id": client_id,
        "redirect_uri": _oauth_redirect_uri(provider),
        "response_type": "code",
        "scope": config["scope"],
        "state": state,
        "prompt": "select_account",
        "code_challenge": _pkce_challenge(code_verifier),
        "code_challenge_method": "S256",
    }
    response = redirect(f"{config['authorize_url']}?{urlencode(params)}")
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        nonce,
        max_age=OAUTH_STATE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=app.config["SESSION_COOKIE_SECURE"],
    )
    return response


@app.route("/api/auth/oauth/<provider>/callback", methods=["GET"])
@limiter.limit("20 per hour")
def auth_oauth_callback(provider):
    """Complete OAuth login/register and establish the signed session cookie."""
    provider = provider.strip().lower()
    config = _oauth_config(provider)
    fallback_next = _safe_next_path(request.args.get("next"))
    if not config:
        return _login_redirect(fallback_next, {"oauth_error": "Unsupported sign-in provider."})
    if request.args.get("error"):
        message = (
            "Sign-in was cancelled."
            if request.args.get("error") == "access_denied"
            else "The identity provider could not complete sign-in. Please try again."
        )
        return _login_redirect(fallback_next, {"oauth_error": message})
    try:
        state = _oauth_serializer().loads(request.args.get("state") or "", max_age=OAUTH_STATE_MAX_AGE)
    except Exception:
        return _login_redirect(fallback_next, {"oauth_error": "Sign-in session expired. Please try again."})
    next_path = _safe_next_path(state.get("next"))
    if state.get("provider") != provider or state.get("nonce") != request.cookies.get(OAUTH_STATE_COOKIE):
        return _login_redirect(next_path, {"oauth_error": "Sign-in session could not be verified."})
    code_verifier = session.pop(OAUTH_PKCE_SESSION_KEY, None)
    if not code_verifier:
        return _login_redirect(next_path, {"oauth_error": "Sign-in session expired. Please try again."})
    code = request.args.get("code")
    if not code:
        return _login_redirect(next_path, {"oauth_error": "Sign-in did not return an authorization code."})
    client_id = os.getenv(config["client_id_env"], "").strip()
    client_secret = os.getenv(config["client_secret_env"], "").strip()
    try:
        token_response = requests.post(
            config["token_url"],
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": _oauth_redirect_uri(provider),
                "code_verifier": code_verifier,
            },
            timeout=10,
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not access_token:
            raise ValueError("missing access token")
        userinfo_response = requests.get(config["userinfo_url"], headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
    except Exception as exc:
        logger.warning(f"OAuth {provider} exchange failed: {exc}")
        return _login_redirect(next_path, {"oauth_error": f"{config['label']} sign-in failed. Please try again."})
    if not isinstance(userinfo, dict):
        return _login_redirect(next_path, {"oauth_error": f"{config['label']} returned an invalid account profile."})
    subject = userinfo.get("sub")
    email = userinfo.get("email")
    if provider == "microsoft" and not email:
        email = userinfo.get("preferred_username")
    if provider == "google" and userinfo.get("email_verified") is not True:
        return _login_redirect(next_path, {"oauth_error": "Google did not confirm that this email address is verified."})
    try:
        account_email = oauth_upsert_user(provider, email, subject)
    except OAuthIdentityConflictError:
        return _login_redirect(next_path, {
            "oauth_error": (
                "An account already exists with this email address. Sign in "
                "using its existing method; automatic account linking is disabled."
            ),
        })
    except Exception as exc:
        logger.error(
            "OAuth account persistence failed for %s: %s",
            provider,
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _login_redirect(next_path, {"oauth_error": "The identity provider signed you in, but the leaderboard could not create your local session. Please try again shortly."})
    if not account_email:
        return _login_redirect(next_path, {"oauth_error": f"{config['label']} did not return a usable, stable account identity."})
    try:
        _establish_user_session(account_email)
    except Exception as exc:
        logger.error(
            "OAuth session creation failed for %s: %s",
            provider,
            exc,
            extra={"request_id": getattr(g, "request_id", None)},
            exc_info=True,
        )
        return _login_redirect(next_path, {"oauth_error": "The account was accepted, but its local session could not be created. Please try again."})
    response = redirect(_frontend_url(next_path))
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


def _service_metadata() -> dict:
    return {
        "service": "MS-VISTA Leaderboard API",
        "version": "1",
        "api_only": True,
        "health": "/api/health",
        "liveness": "/api/health/live",
        "readiness": "/api/readiness",
        "frontend": _frontend_base_url(),
        "request_id": getattr(g, "request_id", None),
    }


@app.route("/", methods=["GET"])
@limiter.exempt
def service_index():
    """Describe this process without serving the frontend application."""
    return jsonify(_service_metadata()), 200


@app.route("/api", methods=["GET"])
@limiter.exempt
def api_index():
    """API discovery endpoint for clients and deployment checks."""
    return jsonify(_service_metadata()), 200


@app.route("/api/health", methods=["GET"])
@app.route("/api/readiness", methods=["GET"])
@limiter.exempt
def health_check():
    """Health check endpoint for monitoring.

    Exempt from rate limiting: liveness/readiness probes and load-balancer
    health checks poll frequently, and rate-limiting them causes false
    "down" signals. This readiness check intentionally verifies storage,
    publication, answer-bundle, authentication, and backup invariants.
    """
    cache_enabled = HEALTH_CACHE_SECONDS > 0 and not app.config.get("TESTING")
    if cache_enabled:
        with _health_cache_lock:
            if (
                _health_cache["payload"] is not None
                and _health_cache["expires_at"] > time.monotonic()
            ):
                return jsonify(_health_cache["payload"]), _health_cache["status_code"]

    try:
        components = {}
        details = {}

        try:
            database_status, database_details = _database_storage_health()
            components["database"] = database_status
            details["database"] = database_details
        except Exception as e:
            logger.error("Database health check failed: %s", e, exc_info=True)
            components["database"] = "unhealthy"
            details["database"] = {"error": "failed to inspect database storage"}

        try:
            submission_integrity = submission_integrity_status()
            components["submission_store"] = (
                "healthy" if submission_integrity["healthy"] else "unhealthy"
            )
            details["submission_store"] = submission_integrity
        except Exception as e:
            logger.error("Submission integrity health check failed: %s", e, exc_info=True)
            components["submission_store"] = "unhealthy"
            details["submission_store"] = {
                "healthy": False,
                "error": "failed to inspect submission integrity",
            }

        try:
            leaderboard_store.visual_cognition_leaderboard(limit=1)
            expected_fingerprints = latest_visible_scored_submission_fingerprints()
            cached_fingerprints = leaderboard_store.public_submission_fingerprints()
            expected_submission_ids = set(expected_fingerprints)
            cached_submission_ids = set(cached_fingerprints)
            mismatched_submission_ids = {
                submission_id
                for submission_id in expected_submission_ids & cached_submission_ids
                if expected_fingerprints[submission_id]
                != cached_fingerprints[submission_id]
            }
            cache_synchronized = (
                expected_submission_ids == cached_submission_ids
                and not mismatched_submission_ids
            )
            components["leaderboard_store"] = (
                "healthy" if cache_synchronized else "unhealthy"
            )
            details["leaderboard_store"] = {
                "synchronized": cache_synchronized,
                "expected_submission_count": len(expected_submission_ids),
                "cached_submission_count": len(cached_submission_ids),
                "missing_submission_count": len(
                    expected_submission_ids - cached_submission_ids
                ),
                "stale_submission_count": len(
                    cached_submission_ids - expected_submission_ids
                ),
                "score_mismatch_count": len(mismatched_submission_ids),
            }
        except Exception as e:
            logger.error("Leaderboard storage health check failed: %s", e, exc_info=True)
            components["leaderboard_store"] = "unhealthy"
            details["leaderboard_store"] = {
                "synchronized": False,
                "error": "failed to inspect leaderboard publication state",
            }

        # Check that visual-task private answers and public IDs are one release.
        try:
            ground_truth_status, ground_truth_details = _ground_truth_bundle_health()
            components["ground_truth"] = ground_truth_status
            details["ground_truth"] = ground_truth_details
        except Exception as e:
            logger.error(f"Ground truth health check failed: {e}")
            components["ground_truth"] = "unhealthy"
            details["ground_truth"] = {
                "error": "failed to verify visual task IDs against private ground truth"
            }

        try:
            spatial_status, spatial_details = _spatial_bundle_health()
            components["spatial_bundle"] = spatial_status
        except Exception as e:
            logger.error(f"Spatial bundle health check failed: {e}")
            spatial_details = {"error": "failed to inspect spatial bundle"}
            components["spatial_bundle"] = "unhealthy"

        try:
            email_status, email_details = _email_delivery_health()
            components["email"] = email_status
        except Exception as e:
            logger.error(f"Email health check failed: {e}")
            email_details = {"error": "failed to inspect email configuration"}
            components["email"] = "unhealthy"
        details["email"] = email_details

        try:
            auth_status, auth_details = _auth_service_health()
            components["auth"] = auth_status
            details["auth"] = auth_details
        except Exception as e:
            logger.error("Auth configuration health check failed: %s", e, exc_info=True)
            components["auth"] = "unhealthy"
            details["auth"] = {"error": "failed to inspect auth configuration"}

        try:
            deployment_status, deployment_details = _deployment_configuration_health()
            components["deployment"] = deployment_status
            details["deployment"] = deployment_details
        except Exception as e:
            logger.error("Deployment configuration health check failed: %s", e, exc_info=True)
            components["deployment"] = "unhealthy"
            details["deployment"] = {"error": "failed to inspect deployment configuration"}

        try:
            backup_status, backup_details = _backup_health()
            components["backup"] = backup_status
            details["backup"] = backup_details
        except Exception as e:
            logger.error("Backup health check failed: %s", e, exc_info=True)
            components["backup"] = "unhealthy"
            details["backup"] = {"error": "failed to inspect scheduled backups"}

        details["spatial_bundle"] = spatial_details
        allowed_statuses = {
            "healthy",
            "demo",
            "dev",
            "disabled",
            "skipped",
            "pending",
        }
        readiness_components = {
            name: status
            for name, status in components.items()
            if name != "spatial_bundle" or REQUIRE_OFFICIAL_SPATIAL
        }
        components_ready = all(
            status in allowed_statuses for status in readiness_components.values()
        )
        production_controls_ready = not PUBLIC_DEPLOYMENT or (
            components.get("auth") == "healthy"
            and components.get("backup") in {"healthy", "pending"}
        )
        spatial_ready = not REQUIRE_OFFICIAL_SPATIAL or components.get("spatial_bundle") == "healthy"
        overall_status = (
            "healthy"
            if components_ready and production_controls_ready and spatial_ready
            else "degraded"
        )
        grading_mode = "deterministic_jsonl_exact"

        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
            grading=grading_mode,
            details=details,
        )

        status_code = 200 if overall_status == "healthy" else 503
        payload = response.model_dump()
        if cache_enabled:
            with _health_cache_lock:
                _health_cache.update({
                    "expires_at": time.monotonic() + HEALTH_CACHE_SECONDS,
                    "payload": payload,
                    "status_code": status_code,
                })
        return jsonify(payload), status_code

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Health check failed"
        }), 500


@app.route("/api/health/live", methods=["GET"])
@limiter.exempt
def liveness_check():
    """Process liveness check that does not touch storage or external services."""
    return jsonify({
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": getattr(g, "request_id", None),
    }), 200

@app.route("/api/submit", methods=["POST"])
@limiter.limit("10 per minute", key_func=_identity_key)
@submission_auth_required
def submit_prediction():
    """Retired legacy submit endpoint.

    The production path is /api/tasks/<task_id>/submit with JSONL final-answer
    rows. Keeping the older endpoint writable would preserve a second upload
    contract and raw-file save path.
    """
    return _error_response(
        "This legacy submission endpoint has been retired. Submit a JSONL response file from the benchmark card on the Submit page.",
        "legacy_submission_endpoint",
        410,
    )

@app.route("/api/leaderboard", methods=["GET"])
@limiter.limit("60 per minute")
def get_leaderboard():
    """Compatibility route backed by the current persistent leaderboard."""
    request_id = getattr(g, "request_id", None)

    try:
        limit, error_response = _query_limit()
        if error_response is not None:
            return error_response

        benchmark = request.args.get("benchmark", "").strip()
        task = request.args.get("task", "").strip()
        if benchmark and task and benchmark != task:
            return _error_response(
                "The benchmark and task filters conflict. Provide only one filter or use the same benchmark identifier for both.",
                "invalid_leaderboard_request",
                400,
            )

        scope = task or benchmark or "visual_cognition"
        valid_scopes = {
            "visual_cognition",
            "do_you_see_me",
            "minds_eye",
            "spatial",
        }
        if scope not in valid_scopes:
            return _error_response(
                f"Unknown leaderboard scope '{scope}'. Use visual_cognition, do_you_see_me, minds_eye, or spatial.",
                "invalid_leaderboard_request",
                400,
            )

        if scope == "spatial":
            rows = leaderboard_store.spatial_leaderboard(limit=limit)
        else:
            rows = leaderboard_store.visual_cognition_leaderboard(
                limit=MAX_LEADERBOARD_LIMIT
            )
            if scope == "do_you_see_me":
                rows = [row for row in rows if row.get("has_perception")]
                rows.sort(
                    key=lambda row: row.get("perception_accuracy")
                    if row.get("perception_accuracy") is not None
                    else -1,
                    reverse=True,
                )
            elif scope == "minds_eye":
                rows = [row for row in rows if row.get("has_cognition")]
                rows.sort(
                    key=lambda row: row.get("cognition_accuracy")
                    if row.get("cognition_accuracy") is not None
                    else -1,
                    reverse=True,
                )
            rows = rows[:limit]
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank

        return jsonify({
            "leaderboard": rows,
            "count": len(rows),
            "scope": scope,
            "request_id": request_id,
        }), 200

    except Exception as e:
        logger.error(f"Leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The leaderboard could not be loaded from storage. Existing rankings were not changed; retry shortly.",
            "leaderboard_unavailable",
            503,
            retryable=True,
        )

@app.route("/api/submission/<submission_id>", methods=["GET"])
@limiter.limit("60 per minute")
def get_submission_details(submission_id):
    """Retired public detail route; submission exports require ownership."""
    return _error_response(
        "Public submission details are no longer exposed. Sign in and open Submission history to view or export a submission you own.",
        "legacy_submission_details_endpoint",
        410,
    )

@app.route("/api/tasks", methods=["GET"])
@limiter.limit("60 per minute")
def get_available_tasks():
    """Get list of available tasks."""
    request_id = getattr(g, 'request_id', None)

    try:
        benchmark_str = request.args.get("benchmark", "").strip()
        valid_filters = {"", "do_you_see_me", "minds_eye", "spatial"}
        if benchmark_str not in valid_filters:
            logger.warning(f"Invalid benchmark: {benchmark_str}", extra={"request_id": request_id})
            return _error_response(
                f"Unknown benchmark filter '{benchmark_str}'. Use do_you_see_me, minds_eye, spatial, or omit the filter.",
                "invalid_benchmark",
                400,
            )

        legacy_groups = gt_manager.list_available_tasks()
        if benchmark_str == "do_you_see_me":
            legacy_groups = {"do_you_see_me": legacy_groups["do_you_see_me"]}
        elif benchmark_str == "minds_eye":
            legacy_groups = {"minds_eye": legacy_groups["minds_eye"]}
        elif benchmark_str == "spatial":
            legacy_groups = {}

        submission_tasks = []
        for task in sorted(TASKS.values(), key=lambda item: item.get("order", 999)):
            if benchmark_str and task["task_id"] != benchmark_str:
                continue
            submission_tasks.append({
                "task_id": task["task_id"],
                "label": task["label"],
                "section": task["section"],
                "layer": task["layer"],
                "order": task["order"],
                "supports_diagnostics": task["supports_diagnostics"],
                "description": task["description"],
                "submission_format": "jsonl",
                "score_method": task.get("score_method"),
                "score_description": task.get("score_description"),
            })
        logger.info(f"Available tasks retrieved", extra={"request_id": request_id})

        return jsonify({
            **legacy_groups,
            "tasks": submission_tasks,
            "task_ids": [task["task_id"] for task in submission_tasks],
            "request_id": request_id,
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving tasks: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The benchmark task list could not be loaded from configuration. Retry shortly or contact the administrator with the request reference.",
            "tasks_unavailable",
            503,
            retryable=True,
        )

@app.route("/api/statistics", methods=["GET"])
@limiter.limit("60 per minute")
def get_statistics():
    """Compatibility statistics route backed by the current store."""
    request_id = getattr(g, "request_id", None)

    try:
        benchmark_str = request.args.get("benchmark", "").strip()
        valid_filters = {"", "do_you_see_me", "minds_eye", "spatial"}
        if benchmark_str not in valid_filters:
            return _error_response(
                f"Unknown benchmark filter '{benchmark_str}'. Use do_you_see_me, minds_eye, spatial, or omit the filter.",
                "invalid_benchmark",
                400,
            )

        stats = leaderboard_store.statistics()
        if benchmark_str == "spatial":
            rows = leaderboard_store.spatial_leaderboard(
                limit=MAX_LEADERBOARD_LIMIT
            )
            best_accuracy = max(
                (
                    row.get("macro_accuracy")
                    if row.get("macro_accuracy") is not None
                    else row.get("accuracy", 0.0)
                    for row in rows
                ),
                default=0.0,
            )
        elif benchmark_str:
            field = (
                "perception_accuracy"
                if benchmark_str == "do_you_see_me"
                else "cognition_accuracy"
            )
            rows = [
                row
                for row in leaderboard_store.visual_cognition_leaderboard(
                    limit=MAX_LEADERBOARD_LIMIT
                )
                if row.get(field) is not None
            ]
            best_accuracy = max(
                (row.get(field, 0.0) for row in rows),
                default=0.0,
            )
        else:
            rows = []
            best_accuracy = None

        return jsonify({
            **stats,
            "benchmark": benchmark_str or None,
            "benchmark_models": len(rows) if benchmark_str else None,
            "best_accuracy": round(best_accuracy, 4)
            if best_accuracy is not None
            else None,
            "request_id": request_id,
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Benchmark statistics could not be loaded. Retry shortly; existing results were not changed.",
            "statistics_unavailable",
            503,
            retryable=True,
        )

# ---------------------------------------------------------------------------
# Three-task Visual Cognition / Spatial Reasoning endpoints
# ---------------------------------------------------------------------------

def _task_or_404(task_id):
    return TASKS.get(task_id)


@app.route("/api/sections", methods=["GET"])
@limiter.limit("60 per minute")
def api_sections():
    """UI layout: the two sections, their tasks, layers and VCI weights."""
    request_id = getattr(g, "request_id", None)
    try:
        sections = []
        for sec in SECTIONS.values():
            sections.append({
                "id": sec["id"],
                "label": sec["label"],
                "primary_metric": sec["primary_metric"],
                "tasks": [
                    {
                        "task_id": TASKS[t]["task_id"],
                        "label": TASKS[t]["label"],
                        "layer": TASKS[t]["layer"],
                        "order": TASKS[t]["order"],
                        "supports_diagnostics": TASKS[t]["supports_diagnostics"],
                        "description": TASKS[t]["description"],
                    }
                    for t in sec["tasks"]
                ],
            })
        return jsonify({
            "sections": sections,
            "layer_labels": LAYER_LABELS,
            "vci_weights": VCI_LAYER_WEIGHTS,
            "eval_conditions": EVAL_CONDITIONS,
            "request_id": request_id,
        }), 200
    except Exception as e:
        logger.error(f"Sections error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The benchmark section configuration could not be loaded. Refresh the page or contact the administrator with the request reference.",
            "sections_unavailable",
            503,
            retryable=True,
        )


@app.route("/api/tasks/<task_id>/info", methods=["GET"])
@limiter.limit("60 per minute")
def task_info(task_id):
    """Metadata + sample count for a single task."""
    request_id = getattr(g, "request_id", None)
    task = _task_or_404(task_id)
    if not task:
        return _error_response(
            f"The benchmark '{task_id}' does not exist.",
            "unknown_task",
            404,
        )
    try:
        total = 0
        spatial_status = spatial_details = None
        if task_id == "spatial":
            spatial_status, spatial_details = _spatial_bundle_health()
            total = int(spatial_details.get("samples") or 0)
        elif task.get("ground_truth_sources"):
            total = len(task_scorers[task_id].ground_truth)
        qfile = task["paths"]["questions"]
        if task_id != "spatial" and not total and qfile.exists():
            with open(qfile, "r", encoding="utf-8") as f:
                total = json.load(f).get("total_samples", 0)
        info = {
            "task_id": task["task_id"],
            "label": task["label"],
            "section": task["section"],
            "layer": task["layer"],
            "group_by": task["group_by"],
            "supports_diagnostics": task["supports_diagnostics"],
            "description": task["description"],
            "total_samples": total,
            "paper_total_samples": task.get("paper_total_samples"),
            "score_method": task.get("score_method"),
            "score_description": task.get("score_description"),
            "score_provenance": (
                "leaderboard_release_suite"
                if task_id != "spatial"
                else "harness_report_with_public_evidence"
            ),
        }
        # Advertise the track-specific verification contract before upload.
        # Visual tasks are server-scored; Spatial publishes harness evidence.
        gcfg = GRADING.get(task_id, {})
        if gcfg:
            info["grading"] = {
                "method": (
                    "harness_reported_public_evidence"
                    if task_id == "spatial"
                    else gcfg.get("method")
                ),
                "paper": gcfg.get("paper"),
                "random_baseline": task_scorers[task_id].random_baseline(),
                "submission_format": (
                    "spatial_evidence_zip" if task_id == "spatial" else "jsonl"
                ),
                "server_ground_truth_evaluation": task_id != "spatial",
            }
        if task_id == "spatial":
            info["datasets"] = SPATIAL_DATASETS
            info["conditions"] = EVAL_CONDITIONS
            info["no_image_plus_option"] = NO_IMAGE_PLUS_OPTION
            info["submission_ready"] = spatial_status == "healthy"
            info["bundle_status"] = spatial_status
            info["bundle_details"] = spatial_details
            info["required_uploads"] = [SPATIAL_SUBMISSION_ARCHIVE_NAME]
            info["upload_processing"] = "in_memory"
            info["max_upload_bytes"] = MAX_SPATIAL_ARCHIVE_BYTES
            info["public_evidence"] = True
            info["archive_members"] = list(SPATIAL_ARCHIVE_MEMBERS)
            info["harness_url"] = "/api/spatial/harness"
            info["manifest_url"] = "/api/spatial/manifest"
        return jsonify({**info, "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Task info error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Benchmark details could not be loaded because the server could not read its configuration. Try again or contact the administrator with the request reference.",
            "task_info_unavailable",
            500,
            retryable=True,
        )


@app.route("/api/tasks/<task_id>/questions", methods=["GET"])
@limiter.limit("30 per minute")
def task_questions(task_id):
    """Download a task's public sample set (no answers)."""
    task = _task_or_404(task_id)
    if not task:
        return _error_response(f"The benchmark '{task_id}' does not exist.", "unknown_task", 404)
    if task_id == "spatial" and _spatial_bundle_health()[0] != "healthy":
        return _error_response(
            "The official spatial question identifiers are not available until the verified 13-dataset bundle is published.",
            "spatial_task_bundle_unavailable",
            503,
            retryable=True,
        )
    qjsonl = task["paths"].get("questions_jsonl")
    if qjsonl and qjsonl.exists():
        return send_file(str(qjsonl), as_attachment=True,
                         download_name=f"{task_id}_questions.jsonl",
                         mimetype="application/x-ndjson")
    qfile = task["paths"]["questions"]
    if not qfile.exists():
        return _error_response(
            "The question set for this benchmark has not been published yet. Contact the leaderboard administrator before preparing a submission.",
            "questions_unavailable",
            404,
        )
    return send_file(str(qfile), as_attachment=True,
                     download_name=f"{task_id}_questions.json", mimetype="application/json")


@app.route("/api/tasks/<task_id>/template.<fmt>", methods=["GET"])
@limiter.limit("30 per minute")
def task_template(task_id, fmt):
    """Download a task's JSONL submission template."""
    task = _task_or_404(task_id)
    if not task:
        return _error_response(f"The benchmark '{task_id}' does not exist.", "unknown_task", 404)
    if task_id == "spatial" and _spatial_bundle_health()[0] != "healthy":
        return _error_response(
            "The official spatial submission template is not available until the verified 13-dataset bundle is published.",
            "spatial_task_bundle_unavailable",
            503,
            retryable=True,
        )
    if fmt == "jsonl" and task["paths"]["template_jsonl"].exists():
        return send_file(str(task["paths"]["template_jsonl"]), as_attachment=True,
                         download_name=f"{task_id}_template.jsonl", mimetype="application/x-ndjson")
    if fmt in {"json", "csv"}:
        return _error_response(
            "JSON and CSV templates are no longer supported. Download and submit the JSONL template instead.",
            "template_format_retired",
            410,
        )
    return _error_response(
        "The JSONL template for this benchmark is unavailable. Contact the leaderboard administrator before preparing a submission.",
        "template_unavailable",
        404,
    )


@app.route("/api/spatial/manifest", methods=["GET"])
@limiter.limit("30 per minute")
def spatial_manifest():
    """Download the Task-3 dataset manifest (the public spec for the harness)."""
    status, details = _spatial_bundle_health()
    if status != "healthy":
        return _error_response(
            "The official spatial benchmark manifest has not been published yet. The checked-in demo bundle cannot be used for leaderboard submissions.",
            "spatial_manifest_unavailable",
            503,
            retryable=True,
            extra={"spatial_bundle": details},
        )
    return send_file(str(SPATIAL_MANIFEST_FILE), as_attachment=True,
                     download_name="spatial_manifest.json", mimetype="application/json")


@app.route("/api/spatial/harness", methods=["GET"])
@limiter.limit("10 per minute")
def spatial_harness():
    """Download the source-only spatial evaluation harness."""
    if not SPATIAL_HARNESS_DIR.is_dir():
        return _error_response(
            "The spatial evaluation harness is not installed on this server.",
            "spatial_harness_unavailable",
            404,
        )
    excluded_dirs = {"LMUData", "results", "__pycache__", ".git", ".pytest_cache"}
    excluded_suffixes = {".pyc", ".pyo"}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(SPATIAL_HARNESS_DIR.rglob("*")):
            relative = path.relative_to(SPATIAL_HARNESS_DIR)
            if path.is_symlink() or path.is_dir() or any(part in excluded_dirs for part in relative.parts):
                continue
            if path.suffix.lower() in excluded_suffixes or path.name == ".DS_Store":
                continue
            bundle.write(path, Path("spatial_reasoning") / relative)
    archive.seek(0)
    return send_file(
        archive,
        as_attachment=True,
        download_name="spatial_reasoning_evaluation.zip",
        mimetype="application/zip",
    )


@app.route("/api/models/mine", methods=["GET"])
@limiter.limit("60 per minute", key_func=_identity_key)
@submission_auth_required
def my_registered_models():
    """List model identities owned by the signed-in account."""
    request_id = getattr(g, "request_id", None)
    try:
        rows = list_registered_models(getattr(g, "user_id", None))
        return jsonify({
            "models": rows,
            "count": len(rows),
            "request_id": request_id,
        }), 200
    except Exception as exc:
        logger.error(
            "Registered model lookup failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "Your registered models could not be loaded. Existing model records were not changed; refresh and try again.",
            "model_registry_unavailable",
            503,
            retryable=True,
        )


@app.route("/api/models", methods=["POST"])
@limiter.limit("10 per hour", key_func=_identity_key)
@submission_auth_required
def register_model():
    """Create one canonical model identity for future benchmark submissions."""
    request_id = getattr(g, "request_id", None)
    if request.content_length and request.content_length > MAX_AUTH_REQUEST_BYTES:
        return _error_response(
            "The model registration request is too large. Shorten the model details and try again.",
            "model_request_too_large",
            413,
        )
    if not request.is_json:
        return _error_response(
            "Model registration requires a JSON request body.",
            "model_json_required",
            415,
        )
    data = request.get_json(silent=True)
    if data is None:
        return _error_response(
            "The model registration request contains malformed JSON.",
            "invalid_model_json",
            400,
        )
    if not isinstance(data, dict):
        return _error_response(
            "The model registration request must be a JSON object with named fields.",
            "invalid_model_json_object",
            400,
        )
    values, field_errors = _registered_model_payload(data)
    if field_errors:
        return _error_response(
            "Review the highlighted model details and correct them before registering this model.",
            "invalid_model_registration",
            400,
            field_errors=field_errors,
        )
    try:
        row = create_registered_model(
            getattr(g, "user_id", None),
            values.pop("model_name"),
            values,
        )
        return jsonify({
            "success": True,
            "model": row,
            "request_id": request_id,
        }), 201
    except ModelNameConflictError:
        return _error_response(
            "A model with this name is already registered. Select the existing model from your workspace, or contact an administrator if another account controls it.",
            "model_name_conflict",
            409,
            field_errors={"model_name": "This canonical model name is already registered."},
        )
    except Exception as exc:
        logger.error(
            "Model registration failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The model could not be registered. No model record was created; refresh and try again.",
            "model_registration_failed",
            500,
            retryable=True,
        )


@app.route("/api/tasks/<task_id>/submit", methods=["POST"])
@limiter.limit("10 per minute", key_func=_identity_key)
@submission_auth_required
def submit_task(task_id):
    """Submit one task's predictions, score them, and update the model entry."""
    request_id = getattr(g, "request_id", None)
    submission_id = None
    submission_finalized = False
    task = _task_or_404(task_id)
    if not task:
        return _error_response(
            f"The benchmark '{task_id}' does not exist. Choose one of: {', '.join(sorted(TASKS))}.",
            "unknown_task",
            404,
        )
    try:
        if task_id == "spatial":
            spatial_status, spatial_details = _spatial_bundle_health()
            if spatial_status != "healthy":
                return _error_response(
                    "Spatial submissions are not open because the server does not have the complete official 13-dataset grading bundle. The demo bundle cannot produce publishable scores.",
                    "spatial_benchmark_not_ready",
                    503,
                    retryable=True,
                    extra={"spatial_bundle": spatial_details},
                )
        if "file" not in request.files:
            expected_file = (
                f"{SPATIAL_SUBMISSION_ARCHIVE_NAME} package"
                if task_id == "spatial"
                else "JSONL response file"
            )
            return _error_response(
                f"Select a {expected_file} before submitting.",
                "missing_submission_file",
                400,
                field_errors={"file": f"A {expected_file} is required."},
            )
        if task_id == "spatial" and (
            set(request.files.keys()) != {"file"}
            or len(request.files.getlist("file")) != 1
        ):
            return _error_response(
                f"Upload one {SPATIAL_SUBMISSION_ARCHIVE_NAME} package only. Do not upload submission.jsonl or run_manifest.json separately.",
                "invalid_spatial_upload_parts",
                400,
                field_errors={"file": "Select the single ZIP package produced by the harness."},
            )
        file = request.files["file"]
        if task_id == "spatial":
            archive_name = str(file.filename or "")
            if (
                not archive_name
                or len(archive_name) > 255
                or Path(archive_name).suffix.lower() != ".zip"
                or "\x00" in archive_name
                or "/" in archive_name
                or "\\" in archive_name
            ):
                return _error_response(
                    f"The spatial submission filename is invalid. Select {SPATIAL_SUBMISSION_ARCHIVE_NAME} produced by the current harness.",
                    "invalid_spatial_archive_file",
                    400,
                    field_errors={"file": "Use the harness-generated .zip package."},
                )
            file.stream.seek(0, 2)
            archive_size = file.stream.tell()
            file.stream.seek(0)
            if archive_size <= 0 or archive_size > MAX_SPATIAL_ARCHIVE_BYTES:
                return _error_response(
                    f"The spatial ZIP package is empty or larger than {MAX_SPATIAL_ARCHIVE_BYTES // (1024 * 1024)} MB. Upload the unchanged package produced by the harness.",
                    "invalid_spatial_archive_size",
                    400,
                    field_errors={"file": f"Package must be a non-empty ZIP file under {MAX_SPATIAL_ARCHIVE_BYTES // (1024 * 1024)} MB."},
                )
        else:
            is_valid, error_msg, _safe_filename = FileSecurityValidator.validate_and_secure_upload(
                file.stream, file.filename
            )
            if not is_valid:
                return _error_response(
                    f"The response file could not be accepted: {error_msg}. Choose a non-empty UTF-8 JSONL file and try again.",
                    "invalid_submission_file",
                    400,
                    field_errors={"file": str(error_msg)},
                )

        owner_email = getattr(g, "user_id", None)
        model_id = _form_value("model_id")
        if model_id and not re.fullmatch(r"mdl_[a-f0-9]{32}", model_id):
            return _error_response(
                "The selected model identifier is malformed. Refresh the model workspace and select the model again.",
                "invalid_model_id",
                400,
                field_errors={"model_id": "Refresh and select a registered model again."},
            )
        registered_model = get_owned_model(model_id, owner_email) if model_id else None
        if registered_model is None and SUBMISSION_AUTH_DISABLED:
            legacy_name = _form_value("model_name")
            if legacy_name and _valid_model_name(legacy_name):
                registered_model = find_owned_model_by_name(legacy_name, owner_email) or {
                    "model_id": "",
                    "model_name": legacy_name,
                    "model_meta": {},
                }
        if registered_model is None:
            return _error_response(
                "Select one of the models registered to your account before uploading benchmark responses.",
                "model_selection_required" if not model_id else "model_not_found",
                400 if not model_id else 404,
                field_errors={
                    "model_id": (
                        "Select a registered model."
                        if not model_id
                        else "This model is unavailable or belongs to a different account."
                    )
                },
            )
        model_id = registered_model["model_id"]
        model_name = registered_model["model_name"]
        model_meta, meta_error = _submission_model_meta(
            task_id,
            registered_model.get("model_meta") or {},
        )
        if meta_error:
            return _error_response(
                str(meta_error),
                "invalid_submission_metadata",
                400,
                field_errors=_metadata_field_errors(meta_error),
            )

        submission_id, quota_error = _enforce_quota(task_id, model_name, model_id)
        if quota_error is not None:
            return quota_error

        uploaded_bytes = file.stream.read(
            MAX_SPATIAL_ARCHIVE_BYTES + 1
            if task_id == "spatial"
            else -1
        )
        if task_id == "spatial" and len(uploaded_bytes) > MAX_SPATIAL_ARCHIVE_BYTES:
            _finalize_submission_safely(submission_id, False)
            return _error_response(
                f"The spatial ZIP package is larger than the allowed {MAX_SPATIAL_ARCHIVE_BYTES // (1024 * 1024)} MB limit.",
                "invalid_spatial_archive_size",
                400,
                field_errors={"file": "Upload the unchanged package produced by the harness."},
            )
        file_sha256 = hashlib.sha256(uploaded_bytes).hexdigest()
        run_manifest_bytes = None
        report_bytes = None
        submission_artifacts = None
        if task_id == "spatial":
            try:
                (
                    submission_bytes,
                    run_manifest_bytes,
                    report_bytes,
                ) = read_spatial_submission_archive(uploaded_bytes)
                submission_artifacts = [
                    {
                        "artifact_name": SPATIAL_SUBMISSION_ARCHIVE_NAME,
                        "media_type": "application/zip",
                        "content": uploaded_bytes,
                    },
                    {
                        "artifact_name": SPATIAL_SUBMISSION_MEMBER,
                        "media_type": "application/x-ndjson",
                        "content": submission_bytes,
                    },
                    {
                        "artifact_name": SPATIAL_MANIFEST_MEMBER,
                        "media_type": "application/json",
                        "content": run_manifest_bytes,
                    },
                    {
                        "artifact_name": SPATIAL_REPORT_MEMBER,
                        "media_type": "application/json",
                        "content": report_bytes,
                    },
                ]
            except SubmissionValidationError as exc:
                logger.info(
                    "Spatial ZIP validation failed (%s): %s",
                    exc.code,
                    exc,
                    extra={"request_id": request_id},
                )
                _finalize_submission_safely(submission_id, False)
                return _submission_validation_response(exc, spatial_archive=True)
        else:
            submission_bytes = uploaded_bytes
        scorer = task_scorers[task_id]
        predictions = None
        parsed_meta = {}
        computed_spatial_report = None
        spatial_contract = None
        try:
            if task_id == "spatial":
                spatial_contract = {
                    "manifest": SPATIAL_MANIFEST_FILE.read_bytes(),
                    "template": TASKS["spatial"]["paths"]["template_jsonl"].read_bytes(),
                    "questions": TASKS["spatial"]["paths"]["questions_jsonl"].read_bytes(),
                }
                spatial_contract["manifest_sha256"] = hashlib.sha256(
                    spatial_contract["manifest"]
                ).hexdigest()
                (
                    answer_records,
                    computed_spatial_report,
                    _benchmark_manifest,
                ) = parse_spatial_evidence(
                    submission_bytes,
                    spatial_contract["manifest"],
                    spatial_contract["template"],
                    spatial_contract["questions"],
                )
            else:
                try:
                    submission_text = submission_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    _finalize_submission_safely(submission_id, False)
                    return _error_response(
                        "The response file is not valid UTF-8 text. Export UTF-8 JSONL without binary content and try again.",
                        "invalid_file_encoding",
                        400,
                        field_errors={"file": "File must use UTF-8 encoding."},
                    )
                predictions, parsed_meta, answer_records = (
                    scorer.parse_submission_text_with_records(submission_text)
                )
        except SubmissionValidationError as exc:
            logger.info(
                "Submission JSONL validation failed (%s): %s",
                exc.code,
                exc,
                extra={"request_id": request_id},
            )
            _finalize_submission_safely(submission_id, False)
            return _submission_validation_response(
                exc,
                spatial_archive=task_id == "spatial",
            )
        except (OSError, ValueError) as exc:
            logger.info(
                "Submission JSONL validation failed: %s",
                exc,
                extra={"request_id": request_id},
            )
            _finalize_submission_safely(submission_id, False)
            if task_id == "spatial":
                return _error_response(
                    "Spatial verification is temporarily unavailable because the server's public benchmark contract could not be validated. Your package was not stored or published; contact the administrator with the request reference.",
                    "spatial_contract_verification_failed",
                    503,
                    retryable=True,
                )
            return _error_response(
                f"The response file is invalid: {exc}",
                "invalid_submission_jsonl",
                400,
                field_errors={"file": str(exc)},
            )

        spatial_run_metadata = None
        if task_id == "spatial":
            try:
                spatial_run_metadata = validate_run_manifest(
                    run_manifest_bytes,
                    submission_bytes,
                    report_bytes,
                    model_name,
                    answer_records,
                    spatial_contract["manifest"],
                )
                validated_spatial_report = validate_spatial_report(
                    report_bytes,
                    model_name,
                    computed_spatial_report,
                )
            except SubmissionValidationError as exc:
                logger.info(
                    "Spatial run manifest validation failed (%s): %s",
                    exc.code,
                    exc,
                    extra={"request_id": request_id},
                )
                _finalize_submission_safely(submission_id, False)
                return _submission_validation_response(exc, spatial_archive=True)
            except (OSError, ValueError) as exc:
                logger.error(
                    "Official spatial manifest validation failed: %s",
                    exc,
                    extra={"request_id": request_id},
                    exc_info=True,
                )
                _finalize_submission_safely(submission_id, False)
                return _error_response(
                    "Spatial verification is temporarily unavailable because the server's official public benchmark contract could not be verified. Your submission was not published; contact the administrator with the request reference.",
                    "spatial_manifest_verification_failed",
                    503,
                    retryable=True,
                )

        try:
            if task_id == "spatial":
                score = build_spatial_task_score(
                    validated_spatial_report,
                    model_name,
                    model_meta,
                    spatial_run_metadata,
                )
            else:
                score = scorer.score_predictions(
                    predictions,
                    model_name=model_name,
                    parsed_meta=parsed_meta,
                    model_meta=model_meta,
                )
            score.model_id = model_id or None
        except SubmissionValidationError as exc:
            logger.info(
                "Submission coverage validation failed (%s): %s",
                exc.code,
                exc,
                extra={"request_id": request_id},
            )
            _finalize_submission_safely(submission_id, False)
            return _submission_validation_response(
                exc,
                spatial_archive=task_id == "spatial",
            )
        except (FileNotFoundError, PermissionError) as exc:
            logger.error(
                "Task grading resources unavailable: %s",
                exc,
                extra={"request_id": request_id},
                exc_info=True,
            )
            _finalize_submission_safely(submission_id, False)
            return _error_response(
                "Scoring is temporarily unavailable because the benchmark grading data could not be loaded. Your file was not published; retry later or contact the administrator with the request reference.",
                "grading_unavailable",
                503,
                retryable=True,
            )
        except ValueError as exc:
            message = str(exc)
            internal_gt_error = "ground truth" in message.lower()
            logger.log(
                logging.ERROR if internal_gt_error else logging.INFO,
                "Task scoring validation failed: %s",
                exc,
                extra={"request_id": request_id},
                exc_info=internal_gt_error,
            )
            _finalize_submission_safely(submission_id, False)
            if internal_gt_error:
                return _error_response(
                    "Scoring is temporarily unavailable because the benchmark grading data is invalid. Your file was not published; contact the administrator with the request reference.",
                    "grading_unavailable",
                    503,
                    retryable=True,
                )
            return _error_response(
                f"The response file does not match this benchmark: {message}",
                "submission_coverage_error",
                400,
                field_errors={"file": message},
            )

        store_submission_answers(
            submission_id,
            score_submission_id=score.submission_id,
            file_sha256=file_sha256,
            records=answer_records,
            model_meta=score.model_meta,
            score_json=score.to_dict(),
            artifacts=submission_artifacts,
            spatial_contract=spatial_contract,
        )
        finalize_submission(submission_id, True)
        submission_finalized = True
        try:
            latest_submission_id = latest_visible_scored_submission_id(
                model_id,
                task_id,
            )
            if not model_id or latest_submission_id == score.submission_id:
                record = leaderboard_store.add_result(
                    score,
                    submitted_by=getattr(g, "user_id", None),
                )
            else:
                _refresh_public_model_task(model_id, task_id)
                record = score.to_dict()
        except Exception as publish_error:
            logger.error(
                "Submission %s was scored but public leaderboard publication failed: %s",
                score.submission_id,
                publish_error,
                extra={"request_id": request_id},
                exc_info=True,
            )
            return jsonify({
                **score.to_dict(),
                "success": True,
                "status": "publication_pending",
                "code": "leaderboard_publication_pending",
                "message": "Your submission was scored and safely stored, but the public leaderboard could not be updated. Do not upload the same file again. An administrator can republish it using Rebuild leaderboard.",
                "retryable": False,
                "request_id": request_id,
                "submission_id": score.submission_id,
                "submission_export_url": f"/api/submissions/{score.submission_id}/export.jsonl",
                "public_evidence_url": (
                    f"/api/public/submissions/{score.submission_id}/evidence"
                    if task_id == "spatial"
                    else None
                ),
                "stored": True,
                "published": False,
            }), 202
        logger.info(
            f"Task '{task_id}' scored for {model_name}: acc={score.accuracy:.4f}",
            extra={"request_id": request_id},
        )
        return jsonify({
            **record,
            "success": True,
            "request_id": request_id,
            "submission_export_url": f"/api/submissions/{score.submission_id}/export.jsonl",
            "public_evidence_url": (
                f"/api/public/submissions/{score.submission_id}/evidence"
                if task_id == "spatial"
                else None
            ),
        }), 200
    except Exception as e:
        if not submission_finalized:
            _finalize_submission_safely(submission_id, False)
        logger.error(f"Task submission error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "We could not store and publish this submission because of an unexpected server error. The submission was not counted as successful; retry once, then contact the administrator with the request reference if it continues.",
            "submission_processing_failed",
            500,
            retryable=True,
        )


@app.route("/api/submissions/<submission_id>/export.jsonl", methods=["GET"])
@limiter.limit("30 per minute")
@submission_auth_required
def export_submission_jsonl(submission_id):
    """Reconstruct a submitted final-answer JSONL from stored DB rows."""
    request_id = getattr(g, "request_id", None)
    if not submission_id or len(submission_id) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", submission_id):
        return _error_response(
            "The submission identifier is malformed. Open the export from your submission history and try again.",
            "invalid_submission_id",
            400,
        )
    try:
        signed_email = getattr(g, "user_id", None)
        user_email = None if (SUBMISSION_AUTH_DISABLED or _is_admin_email(signed_email)) else signed_email
        export = get_submission_export(submission_id, user_email=user_email)
        if export is None:
            return _error_response(
                "This submission does not exist or belongs to a different account.",
                "submission_not_found",
                404,
            )

        def generate():
            for row in export["rows"]:
                yield json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"

        filename = f"{export['task_id']}_{submission_id}.jsonl"
        return Response(
            generate(),
            mimetype="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Submission-Id": submission_id,
                "X-Submission-Row-Count": str(export.get("row_count") or len(export["rows"])),
            },
        )
    except Exception as e:
        logger.error(
            f"Submission export error: {e}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The stored JSONL export could not be generated. Try again; if it continues, contact the administrator with the request reference.",
            "submission_export_failed",
            500,
            retryable=True,
        )


def _valid_public_submission_id(submission_id: str) -> bool:
    return bool(
        submission_id
        and len(submission_id) <= 64
        and re.fullmatch(r"[A-Za-z0-9_-]+", submission_id)
    )


@app.route("/api/public/submissions/<submission_id>/evidence", methods=["GET"])
@limiter.limit("60 per minute")
def public_spatial_evidence(submission_id):
    """Describe retained evidence for one visible spatial submission."""
    request_id = getattr(g, "request_id", None)
    if not _valid_public_submission_id(submission_id):
        return _error_response(
            "The public evidence submission identifier is malformed.",
            "invalid_submission_id",
            400,
        )
    try:
        evidence = get_public_spatial_evidence(submission_id)
        if evidence is None:
            return _error_response(
                "Public evidence was not found. The spatial submission may be hidden, deleted, or unavailable.",
                "public_evidence_not_found",
                404,
            )
        artifact_names = {item["name"] for item in evidence.get("artifacts") or []}
        if artifact_names != set(SPATIAL_PUBLIC_ARTIFACT_NAMES):
            return _error_response(
                "This spatial submission predates public evidence retention or its evidence record is incomplete.",
                "public_evidence_incomplete",
                409,
            )
        response = jsonify({**evidence, "request_id": request_id})
        response.headers["Cache-Control"] = "public, max-age=60, must-revalidate"
        return response, 200
    except Exception as exc:
        logger.error(
            "Public spatial evidence lookup failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "Public spatial evidence could not be loaded. Retry shortly or contact the administrator with the request reference.",
            "public_evidence_unavailable",
            503,
            retryable=True,
        )


def _public_spatial_artifact_response(submission_id: str, artifact_name: str):
    request_id = getattr(g, "request_id", None)
    if not _valid_public_submission_id(submission_id):
        return _error_response(
            "The public evidence submission identifier is malformed.",
            "invalid_submission_id",
            400,
        )
    if artifact_name not in SPATIAL_PUBLIC_ARTIFACT_NAMES:
        return _error_response(
            "The requested public evidence artifact is not part of the spatial submission contract.",
            "invalid_evidence_artifact",
            400,
        )
    try:
        artifact = get_public_spatial_artifact(submission_id, artifact_name)
        if artifact is None:
            return _error_response(
                "The public evidence artifact was not found. The spatial submission may be hidden, deleted, or incomplete.",
                "public_evidence_not_found",
                404,
            )
        etag = artifact["sha256"]
        common_headers = {
            "Cache-Control": "public, max-age=60, must-revalidate, no-transform",
            "ETag": f'"{etag}"',
            "X-Evidence-SHA256": etag,
            "X-Submission-Id": submission_id,
        }
        if request.if_none_match.contains(etag):
            return Response(status=304, headers=common_headers)
        disposition = "attachment" if artifact_name.endswith(".zip") else "inline"
        return Response(
            artifact["content"],
            mimetype=artifact["media_type"],
            headers={
                **common_headers,
                "Content-Disposition": (
                    f'{disposition}; filename="{artifact_name}"'
                ),
            },
        )
    except Exception as exc:
        logger.error(
            "Public spatial artifact download failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The public evidence artifact could not be read. Retry shortly or contact the administrator with the request reference.",
            "public_evidence_unavailable",
            503,
            retryable=True,
        )


@app.route(
    "/api/public/submissions/<submission_id>/artifacts/<artifact_name>",
    methods=["GET"],
)
@limiter.limit("60 per minute")
def public_spatial_artifact(submission_id, artifact_name):
    return _public_spatial_artifact_response(submission_id, artifact_name)


@app.route("/api/public/submissions/<submission_id>/answers.jsonl", methods=["GET"])
@limiter.limit("60 per minute")
def public_spatial_answers(submission_id):
    return _public_spatial_artifact_response(
        submission_id,
        SPATIAL_SUBMISSION_MEMBER,
    )


@app.route("/api/submissions/mine", methods=["GET"])
@limiter.limit("60 per minute", key_func=_identity_key)
@submission_auth_required
def my_submissions():
    """List submissions owned by the signed-in account."""
    request_id = getattr(g, "request_id", None)
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    except (TypeError, ValueError):
        return _error_response("The history limit must be a whole number from 1 to 500.", "invalid_limit", 400)
    user_email = getattr(g, "user_id", None)
    try:
        rows = list_submissions(user_email=user_email, limit=limit, include_deleted=False)
    except Exception as exc:
        logger.error(
            "Submission history query failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "Your submission history could not be loaded. Your submissions are still stored; retry shortly or contact the administrator with the request reference.",
            "submission_history_unavailable",
            503,
            retryable=True,
        )
    return jsonify({"submissions": rows, "count": len(rows), "request_id": request_id}), 200


@app.route("/api/submissions/<submission_id>", methods=["DELETE"])
@app.route("/api/submissions/<submission_id>/delete", methods=["POST"])
@limiter.limit("20 per minute", key_func=_identity_key)
@submission_auth_required
def delete_my_submission(submission_id):
    """Delete one submission owned by the signed-in account."""
    request_id = getattr(g, "request_id", None)
    if not submission_id or len(submission_id) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", submission_id):
        return _error_response(
            "The submission identifier is malformed. Refresh your submission history and try again.",
            "invalid_submission_id",
            400,
        )
    user_email = getattr(g, "user_id", None)
    try:
        row = delete_owned_submission(submission_id, user_email)
        if row is None:
            return _error_response(
                "This submission does not exist, has already been deleted, or belongs to a different account.",
                "submission_not_found",
                404,
            )
        previous = row.pop("previous_moderation", {})
        try:
            refresh = _refresh_public_model_task(
                row.get("model_id"),
                row.get("task_id"),
            )
        except Exception:
            _rollback_moderation(submission_id, previous)
            try:
                _refresh_public_model_task(
                    row.get("model_id"),
                    row.get("task_id"),
                )
            except Exception as rollback_error:
                logger.critical(
                    "Member deletion rollback for %s could not republish the prior state: %s",
                    submission_id,
                    rollback_error,
                    extra={"request_id": request_id},
                    exc_info=True,
                )
            raise
        return jsonify({
            "success": True,
            "submission": row,
            "leaderboard_updated": True,
            "active_submission_id": refresh.get("submission_id"),
            "request_id": request_id,
        }), 200
    except Exception as exc:
        logger.error(
            "Member submission deletion failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The submission could not be deleted. It remains in your history; refresh the page and retry.",
            "submission_delete_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/submissions", methods=["GET"])
@limiter.limit("60 per minute")
@admin_required
def admin_submissions():
    """Admin submission audit list."""
    request_id = getattr(g, "request_id", None)
    try:
        limit = min(max(int(request.args.get("limit", 200)), 1), 500)
    except (TypeError, ValueError):
        return _error_response("The audit limit must be a whole number from 1 to 500.", "invalid_limit", 400)
    try:
        rows = list_submissions(limit=limit)
    except Exception as exc:
        logger.error("Admin submission query failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The submission audit list could not be loaded. Retry shortly or use the request reference when contacting support.",
            "admin_submissions_unavailable",
            503,
            retryable=True,
        )
    return jsonify({"submissions": rows, "count": len(rows), "request_id": request_id}), 200


@app.route("/api/admin/submissions/<submission_id>/hide", methods=["POST"])
@limiter.limit("30 per minute")
@admin_required
def admin_hide_submission(submission_id):
    """Hide a submission from public leaderboards without deleting audit data."""
    request_id = getattr(g, "request_id", None)
    data, body_error = _optional_json_object()
    if body_error is not None:
        return body_error
    try:
        row, refresh = _set_moderation_and_refresh(
            submission_id,
            "hidden",
            reason=str(data.get("reason") or "").strip(),
            moderated_by=getattr(g, "user_id", None),
        )
        if row is None:
            return _error_response("The submission could not be found; refresh the audit list.", "submission_not_found", 404)
        return jsonify({
            "submission": row,
            "leaderboard_updated": True,
            "active_submission_id": refresh.get("submission_id"),
            "request_id": request_id,
        }), 200
    except ValueError as exc:
        return _error_response(str(exc), "invalid_moderation_request", 400)
    except Exception as exc:
        logger.error("Hide submission failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The submission could not be hidden. Its existing visibility was preserved; refresh the list and retry.",
            "moderation_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/submissions/<submission_id>/delete", methods=["POST", "DELETE"])
@limiter.limit("30 per minute")
@admin_required
def admin_delete_submission(submission_id):
    """Soft-delete a submission and remove it from public leaderboards."""
    request_id = getattr(g, "request_id", None)
    data, body_error = _optional_json_object()
    if body_error is not None:
        return body_error
    try:
        row, refresh = _set_moderation_and_refresh(
            submission_id,
            "deleted",
            reason=str(data.get("reason") or "").strip(),
            moderated_by=getattr(g, "user_id", None),
        )
        if row is None:
            return _error_response("The submission could not be found; refresh the audit list.", "submission_not_found", 404)
        return jsonify({
            "submission": row,
            "leaderboard_updated": True,
            "active_submission_id": refresh.get("submission_id"),
            "request_id": request_id,
        }), 200
    except ValueError as exc:
        return _error_response(str(exc), "invalid_moderation_request", 400)
    except Exception as exc:
        logger.error("Delete submission failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The submission could not be deleted. Its existing status was preserved; refresh the list and retry.",
            "moderation_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/submissions/<submission_id>/rescore", methods=["POST"])
@limiter.limit("20 per minute")
@admin_required
def admin_rescore_submission(submission_id):
    """Re-score one stored submission from persisted final-answer rows."""
    request_id = getattr(g, "request_id", None)
    result, error = _rescore_stored_submission(submission_id)
    if error:
        status = 404 if error.get("code") == "submission_not_found" else 503
        return _error_response(error["error"], error.get("code", "rescore_failed"), status, retryable=status >= 500)
    try:
        score, stored = result
        published = False
        active_submission_id = None
        if stored.get("moderation_status") == "visible":
            refresh = _refresh_public_model_task(
                stored.get("model_id"),
                stored.get("task_id"),
            )
            active_submission_id = refresh.get("submission_id")
            published = active_submission_id == submission_id
        return jsonify({
            **score.to_dict(),
            "success": True,
            "published": published,
            "active_submission_id": active_submission_id,
            "request_id": request_id,
        }), 200
    except Exception as exc:
        logger.error("Publishing rescored submission failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The submission was rescored but could not be republished. Rebuild the leaderboard before relying on public rankings.",
            "rescore_publish_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/submissions/<submission_id>/restore", methods=["POST"])
@limiter.limit("20 per minute")
@admin_required
def admin_restore_submission(submission_id):
    """Restore a hidden/deleted submission and publish its freshly rescored result."""
    request_id = getattr(g, "request_id", None)
    try:
        row, refresh = _set_moderation_and_refresh(
            submission_id,
            "visible",
            reason="Restored by admin",
            moderated_by=getattr(g, "user_id", None),
        )
        if row is None:
            return _error_response("The submission could not be found; refresh the audit list.", "submission_not_found", 404)
        score = refresh.get("score")
        return jsonify({
            "submission": row,
            "score": score.to_dict() if score is not None else None,
            "published": refresh.get("published", False),
            "active_submission_id": refresh.get("submission_id"),
            "request_id": request_id,
        }), 200
    except Exception as exc:
        logger.error("Restore submission failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The submission could not be fully restored. Refresh the audit list to confirm its current state before retrying.",
            "restore_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/rescore", methods=["POST"])
@limiter.limit("5 per hour")
@admin_required
def admin_rescore_all():
    """Re-score all visible scored submissions and rebuild public leaderboards."""
    request_id = getattr(g, "request_id", None)
    data, body_error = _optional_json_object()
    if body_error is not None:
        return body_error
    try:
        limit = min(max(int(data.get("limit") or request.args.get("limit") or 1000), 1), 10_000)
    except (TypeError, ValueError):
        return _error_response("The rebuild limit must be a whole number from 1 to 10000.", "invalid_limit", 400)
    scored = []
    errors = []
    try:
        submission_ids = latest_visible_scored_submission_ids(limit=limit + 1)
    except Exception as exc:
        logger.error("Visible submission lookup failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The leaderboard rebuild could not start because stored submissions are unavailable. Existing rankings were left unchanged.",
            "submission_storage_unavailable",
            503,
            retryable=True,
        )
    if len(submission_ids) > limit:
        return _error_response(
            f"The rebuild limit of {limit} would omit current model results. Increase the limit to at least {limit + 1} and run the rebuild again; the public leaderboard was not changed.",
            "leaderboard_rebuild_limit_too_low",
            409,
            field_errors={"limit": f"Use a value of at least {limit + 1}."},
            extra={"limit": limit, "minimum_required": limit + 1},
        )
    for score_submission_id in submission_ids:
        result, error = _rescore_stored_submission(score_submission_id)
        if error:
            errors.append({"submission_id": score_submission_id, **error})
            continue
        score, stored = result
        scored.append((score, stored.get("user_email")))
    if errors:
        return _error_response(
            f"The leaderboard was not rebuilt because {len(errors)} current submission(s) could not be rescored. Existing public rankings were preserved. Resolve the listed audit records, then rebuild again.",
            "leaderboard_rebuild_incomplete",
            409,
            extra={
                "rescored": len(scored),
                "rescore_errors": len(errors),
                "errors": errors[:20],
            },
        )
    try:
        published = leaderboard_store.replace_all_results(scored)
    except Exception as exc:
        logger.error("Leaderboard rebuild publish failed: %s", exc, extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Submissions were rescored, but the rebuilt leaderboard could not be saved. Existing public rankings were left in place; retry after checking storage.",
            "leaderboard_rebuild_failed",
            500,
            retryable=True,
            extra={"rescored": len(scored), "rescore_errors": len(errors)},
        )
    return jsonify({
        "success": True,
        "rescored": len(scored),
        "published": published,
        "errors": [],
        "request_id": request_id,
    }), 200


@app.route("/api/admin/backups/status", methods=["GET"])
@limiter.limit("30 per hour")
@admin_required
def admin_backup_status():
    """Return scheduled-backup status without exposing filesystem paths."""
    status, details = backup_scheduler.status(include_error=True)
    return jsonify({
        "status": status,
        "backup": details,
        "request_id": getattr(g, "request_id", None),
    }), 200


@app.route("/api/admin/backups/run", methods=["POST"])
@limiter.limit("5 per hour")
@admin_required
def admin_run_backup():
    """Create and retain a verified server-side backup immediately."""
    request_id = getattr(g, "request_id", None)
    try:
        destination = backup_scheduler.run_if_due(force=True)
        status, details = backup_scheduler.status(include_error=True)
        return jsonify({
            "success": True,
            "filename": destination.name if destination is not None else None,
            "status": status,
            "backup": details,
            "request_id": request_id,
        }), 201
    except Exception as exc:
        logger.error(
            "Server-side backup failed: %s",
            exc,
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The server could not create a verified database backup. Existing backups were not changed; check persistent storage and retry with the request reference.",
            "backup_failed",
            500,
            retryable=True,
        )


@app.route("/api/admin/backups/download", methods=["POST"])
@limiter.limit("10 per hour")
@admin_required
def admin_download_backup():
    """Create and download a CSRF-protected backup archive."""
    request_id = getattr(g, "request_id", None)
    try:
        archive, filename, _manifest = create_backup_archive()
        validation = validate_backup_archive(archive)
        response = send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )
        response.headers["X-Backup-Request-Id"] = request_id or ""
        response.headers["X-Backup-Sqlite-Database-Count"] = str(
            validation["sqlite_snapshots"]
        )
        return response
    except Exception as e:
        logger.error(
            f"Backup download error: {e}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        return _error_response(
            "The backup archive could not be created. No database files were changed; check storage access and retry with the request reference.",
            "backup_failed",
            500,
            retryable=True,
        )


@app.route("/api/leaderboard/visual-cognition", methods=["GET"])
@limiter.limit("60 per minute")
def leaderboard_visual_cognition():
    """Combined Do-You-See-Me + Mind's-Eye ranking (VCI)."""
    request_id = getattr(g, "request_id", None)
    try:
        limit, error_response = _query_limit()
        if error_response is not None:
            return error_response
        rows = leaderboard_store.visual_cognition_leaderboard(limit=limit)
        return jsonify({"leaderboard": rows, "count": len(rows),
                        "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"VC leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Visual-cognition rankings could not be loaded. Existing leaderboard data was not changed; retry shortly.",
            "leaderboard_unavailable",
            503,
            retryable=True,
        )


@app.route("/api/leaderboard/spatial", methods=["GET"])
@limiter.limit("60 per minute")
def leaderboard_spatial():
    """Task-3 spatial ranking with robustness diagnostics."""
    request_id = getattr(g, "request_id", None)
    try:
        limit, error_response = _query_limit()
        if error_response is not None:
            return error_response
        rows = leaderboard_store.spatial_leaderboard(limit=limit)
        return jsonify({"leaderboard": rows, "count": len(rows),
                        "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Spatial leaderboard error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Spatial rankings could not be loaded. Existing leaderboard data was not changed; retry shortly.",
            "leaderboard_unavailable",
            503,
            retryable=True,
        )


@app.route("/api/statistics/overview", methods=["GET"])
@limiter.limit("60 per minute")
def statistics_overview():
    request_id = getattr(g, "request_id", None)
    try:
        return jsonify({**leaderboard_store.statistics(), "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Overview statistics error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "Leaderboard summary statistics could not be loaded. Retry shortly.",
            "statistics_unavailable",
            503,
            retryable=True,
        )


@app.route("/api/model/<path:model_name>/report", methods=["GET"])
@limiter.limit("60 per minute")
def model_report(model_name):
    """Full per-model report across all three tasks."""
    request_id = getattr(g, "request_id", None)
    try:
        if not model_name or len(model_name) > 255:
            return _error_response(
                "The model name in this report link is invalid. Reopen the report from the leaderboard.",
                "invalid_model_name",
                400,
            )
        report = leaderboard_store.get_model(model_name)
        if report is None:
            return _error_response(
                "This model is not present in the current leaderboard. It may have been renamed, hidden, or removed.",
                "model_not_found",
                404,
            )
        return jsonify({**report, "request_id": request_id}), 200
    except Exception as e:
        logger.error(f"Model report error: {e}", extra={"request_id": request_id}, exc_info=True)
        return _error_response(
            "The model report could not be loaded. Close it and retry; use the request reference if the problem continues.",
            "model_report_unavailable",
            503,
            retryable=True,
        )

if __name__ == "__main__":
    host = os.getenv("HOST", os.getenv("FLASK_HOST", "0.0.0.0"))
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5050")))
    use_dev_server = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")

    if use_dev_server:
        # Development server (auto-reload, debugger). Not for production.
        logger.info(f"Starting Flask development server on {host}:{port}")
        app.run(host=host, port=port, debug=True, use_reloader=True)
    else:
        # Production-grade WSGI server. Waitress works on Windows (unlike
        # Gunicorn, which is Unix-only).
        try:
            from waitress import serve
        except ImportError:
            logger.warning(
                "waitress not installed; falling back to the Flask development "
                "server. Install it with `pip install waitress` for production."
            )
            app.run(host=host, port=port, debug=False)
        else:
            logger.info(f"Starting Waitress production server on {host}:{port}")
            serve(
                app,
                host=host,
                port=port,
                threads=8,
                # Waitress otherwise spills request bodies above 512 KB to a
                # temporary file before Flask can apply the spatial route's
                # in-memory stream factory.
                inbuf_overflow=MAX_SPATIAL_MULTIPART_BYTES,
                max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
            )
