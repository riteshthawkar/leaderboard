import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth_db  # noqa: E402


@pytest.fixture()
def isolated_auth(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(auth_db, "_engine", engine)
    monkeypatch.setattr(auth_db, "_Session", session_factory)
    monkeypatch.setattr(auth_db, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(auth_db, "_DB_PATH", None)
    auth_db.Base.metadata.create_all(engine)
    try:
        yield auth_db
    finally:
        engine.dispose()


def test_password_reset_unknown_reuse_and_login_flow(isolated_auth):
    db = isolated_auth

    status, _verify_token = db.register_user("User@Example.com", "old-password-value")
    assert status == "created"
    assert db.login_user("user@example.com", "old-password-value") == "unverified"

    assert db.request_password_reset("not-an-email") == ("invalid", None)
    assert db.request_password_reset("missing@example.com") == ("sent", None)

    status, reset_token = db.request_password_reset("user@example.com")
    assert status == "sent"
    assert reset_token

    assert db.reset_password(reset_token, "short") == "weak"
    assert db.login_user("user@example.com", "old-password-value") == "unverified"

    assert db.reset_password(reset_token, "new-secure-password") == "ok"
    assert db.login_user("user@example.com", "old-password-value") == "invalid"
    assert db.login_user("user@example.com", "new-secure-password") == "ok"
    assert db.reset_password(reset_token, "another-secure-password") == "invalid"


def test_password_reset_expired_token_is_rejected(isolated_auth):
    db = isolated_auth

    status, _verify_token = db.register_user("expired@example.com", "old-password-value")
    assert status == "created"
    status, reset_token = db.request_password_reset("expired@example.com")
    assert status == "sent"
    assert reset_token

    with db._Session() as session:
        user = session.query(db.User).filter_by(email="expired@example.com").one()
        user.password_reset_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert db.reset_password(reset_token, "new-secure-password") == "invalid"
    assert db.login_user("expired@example.com", "new-secure-password") == "invalid"


def test_verification_and_reset_tokens_are_hashed_at_rest(isolated_auth):
    db = isolated_auth

    status, verification_token = db.register_user("hashed@example.com", "secure-password")
    assert status == "created"
    with db._Session() as session:
        user = session.query(db.User).filter_by(email="hashed@example.com").one()
        assert user.verification_token == hashlib.sha256(verification_token.encode()).hexdigest()
        assert user.verification_token != verification_token

    assert db.verify_email(verification_token) == "hashed@example.com"
    status, reset_token = db.request_password_reset("hashed@example.com")
    assert status == "sent"
    with db._Session() as session:
        user = session.query(db.User).filter_by(email="hashed@example.com").one()
        assert user.password_reset_token == hashlib.sha256(reset_token.encode()).hexdigest()
        assert user.password_reset_token != reset_token

    assert db.reset_password(reset_token, "new-secure-password") == "ok"


def test_repeat_registration_replaces_unverified_credentials(isolated_auth):
    db = isolated_auth
    first_password = "violet telescope cedar glacier"
    second_password = "amber orbit meadow lantern"

    first_status, first_token = db.register_user(
        "claimed@example.com",
        first_password,
    )
    second_status, second_token = db.register_user(
        "claimed@example.com",
        second_password,
    )

    assert first_status == "created"
    assert second_status == "resent"
    assert first_token != second_token
    assert db.verify_email(first_token) is None
    assert db.verify_email(second_token) == "claimed@example.com"
    assert db.login_user("claimed@example.com", first_password) == "invalid"
    assert db.login_user("claimed@example.com", second_password) == "ok"


def test_legacy_plaintext_verification_token_remains_consumable(isolated_auth):
    db = isolated_auth
    status, _token = db.register_user("legacy@example.com", "secure-password")
    assert status == "created"
    with db._Session() as session:
        user = session.query(db.User).filter_by(email="legacy@example.com").one()
        user.verification_token = "legacy-plaintext-token"
        session.commit()

    assert db.verify_email("legacy-plaintext-token") == "legacy@example.com"


def test_password_policy_bounds_and_email_length_are_enforced(isolated_auth):
    db = isolated_auth

    assert db.password_policy_status("x" * 14) == "too_short"
    assert db.password_policy_status("x" * 15) == "ok"
    assert db.password_policy_status("x" * 128) == "ok"
    assert db.password_policy_status("x" * 129) == "too_long"
    assert db.password_policy_status("passwordpassword") == "common"
    assert db.register_user("x" * 245 + "@example.com", "x" * 15) == ("invalid", None)
    assert db.is_valid_email("name@example..com") is False
    assert db.is_valid_email(".name@example.com") is False
    assert db.is_valid_email("name@example.com") is True


def test_password_reset_increments_session_version(isolated_auth):
    db = isolated_auth
    status, verification_token = db.register_user(
        "sessions@example.com",
        "original-password-value",
    )
    assert status == "created"
    assert db.verify_email(verification_token) == "sessions@example.com"
    before = db.get_user("sessions@example.com")["session_version"]

    status, reset_token = db.request_password_reset("sessions@example.com")
    assert status == "sent"
    assert db.reset_password(reset_token, "replacement-password-value") == "ok"

    after = db.get_user("sessions@example.com")["session_version"]
    assert after == before + 1


def test_oauth_identity_uses_stable_provider_subject(isolated_auth):
    db = isolated_auth

    assert db.oauth_upsert_user("microsoft", "first@example.com", "subject-1") == "first@example.com"
    assert db.oauth_upsert_user("microsoft", "renamed@example.com", "subject-1") == "first@example.com"
    with pytest.raises(db.OAuthIdentityConflictError):
        db.oauth_upsert_user("microsoft", "first@example.com", "subject-2")
    assert db.oauth_upsert_user("microsoft", "second@example.com", "") is None


def test_oauth_does_not_auto_link_an_existing_password_account(isolated_auth):
    db = isolated_auth
    status, verification_token = db.register_user(
        "existing@example.com",
        "violet telescope cedar glacier",
    )
    assert status == "created"
    assert db.verify_email(verification_token) == "existing@example.com"

    with pytest.raises(db.OAuthIdentityConflictError):
        db.oauth_upsert_user(
            "microsoft",
            "existing@example.com",
            "microsoft-subject",
        )

    assert db.login_user(
        "existing@example.com",
        "violet telescope cedar glacier",
    ) == "ok"
    account = db.get_user("existing@example.com")
    assert account["auth_provider"] == "password"


def test_concurrent_password_reset_consumes_token_once(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent-auth.db'}",
        connect_args={"check_same_thread": False, "timeout": 5},
    )
    monkeypatch.setattr(auth_db, "_engine", engine)
    monkeypatch.setattr(auth_db, "_Session", sessionmaker(bind=engine))
    monkeypatch.setattr(auth_db, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(auth_db, "_DB_PATH", tmp_path / "concurrent-auth.db")
    auth_db.Base.metadata.create_all(engine)
    status, verification_token = auth_db.register_user(
        "concurrent@example.com",
        "violet telescope cedar glacier",
    )
    assert status == "created"
    assert auth_db.verify_email(verification_token) == "concurrent@example.com"
    status, reset_token = auth_db.request_password_reset("concurrent@example.com")
    assert status == "sent"
    barrier = threading.Barrier(2)

    def reset(candidate):
        barrier.wait()
        return auth_db.reset_password(reset_token, candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            reset,
            ["amber orbit meadow lantern", "silver canyon maple horizon"],
        ))

    assert sorted(results) == ["invalid", "ok"]
    engine.dispose()
