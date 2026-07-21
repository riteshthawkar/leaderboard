import importlib
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


import auth_db  # noqa: E402


PASSWORD = "violet telescope cedar glacier"
NEW_PASSWORD = "amber orbit meadow lantern"


@pytest.fixture()
def auth_app(monkeypatch):
    web = importlib.import_module("web.app")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(auth_db, "_engine", engine)
    monkeypatch.setattr(auth_db, "_Session", session_factory)
    monkeypatch.setattr(auth_db, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(auth_db, "_DB_PATH", None)
    auth_db.Base.metadata.create_all(engine)

    monkeypatch.setattr(web, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web, "AUTH_DEV_MODE", True)
    monkeypatch.setattr(web.limiter, "enabled", False)
    monkeypatch.setattr(
        web,
        "_email_delivery_health",
        lambda: ("dev", {"provider": "log", "production_ready": False}),
    )
    monkeypatch.setattr(web, "send_verification_email", lambda _email, _url: "logged")
    monkeypatch.setattr(web, "send_password_reset_email", lambda _email, _url: "logged")
    monkeypatch.setattr(
        web,
        "quota_status",
        lambda _email: {"limit": 3, "used": 0, "remaining": 3, "reset_at": None},
    )
    web.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)
    try:
        yield web
    finally:
        engine.dispose()


def _register(client, email="route-user@example.com", password=PASSWORD):
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201
    return response.get_json()


def _verify(client, verify_url):
    parsed = urlparse(verify_url)
    token = parse_qs(parsed.fragment)["verify_token"][0]
    response = client.post(
        "/api/auth/verify",
        json={"token": token},
    )
    return response


def _reset_token(reset_url):
    return parse_qs(urlparse(reset_url).fragment)["reset_token"][0]


@pytest.mark.parametrize(
    ("body", "content_type", "expected_code", "expected_status"),
    [
        ("{broken", "application/json", "invalid_auth_json", 400),
        ("[]", "application/json", "invalid_auth_json_object", 400),
        ("email=user@example.com", "application/x-www-form-urlencoded", "auth_json_required", 415),
    ],
)
def test_auth_endpoints_reject_malformed_or_non_object_json(
    auth_app,
    body,
    content_type,
    expected_code,
    expected_status,
):
    client = auth_app.app.test_client()
    response = client.post("/api/auth/login", data=body, content_type=content_type)

    assert response.status_code == expected_status
    assert response.get_json()["code"] == expected_code
    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_auth_endpoints_reject_wrong_field_types_and_oversized_bodies(auth_app):
    client = auth_app.app.test_client()

    wrong_type = client.post(
        "/api/auth/login",
        json={"email": 123, "password": PASSWORD},
    )
    oversized = client.post(
        "/api/auth/login",
        data='{"email":"' + ("x" * auth_app.MAX_AUTH_REQUEST_BYTES) + '"}',
        content_type="application/json",
    )

    assert wrong_type.status_code == 400
    assert wrong_type.get_json()["code"] == "invalid_auth_field_type"
    assert wrong_type.get_json()["field_errors"]["email"]
    assert oversized.status_code == 413
    assert oversized.get_json()["code"] == "auth_request_too_large"


def test_registration_and_reset_enforce_new_password_bounds(auth_app):
    client = auth_app.app.test_client()

    short = client.post(
        "/api/auth/register",
        json={"email": "short@example.com", "password": "x" * 14},
    )
    long = client.post(
        "/api/auth/register",
        json={"email": "long@example.com", "password": "x" * 129},
    )
    common = client.post(
        "/api/auth/register",
        json={"email": "common@example.com", "password": "passwordpassword"},
    )

    assert short.status_code == 400
    assert short.get_json()["code"] == "weak_password"
    assert "15" in short.get_json()["field_errors"]["password"]
    assert long.status_code == 400
    assert long.get_json()["code"] == "password_too_long"
    assert "128" in long.get_json()["field_errors"]["password"]
    assert common.status_code == 400
    assert common.get_json()["code"] == "common_password"


def test_signup_verify_login_logout_and_cookie_security_contract(auth_app):
    client = auth_app.app.test_client()
    registration = _register(client)
    verification_url = urlparse(registration["dev_verify_url"])
    assert verification_url.path == "/login"
    assert verification_url.query == ""
    assert parse_qs(verification_url.fragment)["verify_token"]

    unverified = client.post(
        "/api/auth/login",
        json={"email": registration["email"], "password": PASSWORD},
    )
    assert unverified.status_code == 403
    assert unverified.get_json()["code"] == "unverified"

    verified = _verify(client, registration["dev_verify_url"])
    assert verified.status_code == 200
    cookie = verified.headers["Set-Cookie"]
    assert "ms_vista_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    assert "Path=/" in cookie

    me = client.get("/api/auth/me")
    profile = me.get_json()
    assert profile["authenticated"] is True
    assert profile["email_verified"] is True
    assert profile["csrf_token"]
    assert me.headers["Cache-Control"] == "no-store, max-age=0"
    assert me.headers["Referrer-Policy"] == "no-referrer"

    missing_csrf = client.post("/api/auth/logout", json={})
    assert missing_csrf.status_code == 403
    assert missing_csrf.get_json()["code"] == "csrf_required"

    logged_out = client.post(
        "/api/auth/logout",
        json={},
        headers={auth_app.CSRF_HEADER: profile["csrf_token"]},
    )
    assert logged_out.status_code == 200
    assert client.get("/api/auth/me").get_json() == {"authenticated": False}


def test_legacy_get_verification_does_not_log_the_browser_in(auth_app):
    client = auth_app.app.test_client()
    registration = _register(client, "legacy-link@example.com")
    token = parse_qs(urlparse(registration["dev_verify_url"]).fragment)["verify_token"][0]

    response = client.get(
        f"/api/auth/verify?token={token}",
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = urlparse(response.headers["Location"])
    assert location.path == "/login"
    assert parse_qs(location.fragment) == {"verified": ["1"]}
    assert client.get("/api/auth/me").get_json() == {"authenticated": False}
    assert auth_app.login_user("legacy-link@example.com", PASSWORD) == "ok"


def test_resend_rotates_verification_token_without_account_enumeration(auth_app):
    client = auth_app.app.test_client()
    registration = _register(client, "resend@example.com")

    resend = client.post("/api/auth/resend", json={"email": "resend@example.com"})
    assert resend.status_code == 200
    assert resend.get_json()["status"] == "verification_sent"
    assert resend.get_json()["dev_verify_url"]

    old_link = _verify(client, registration["dev_verify_url"])
    assert old_link.status_code == 400
    assert old_link.get_json()["code"] == "invalid_verification_token"
    current_link = _verify(client, resend.get_json()["dev_verify_url"])
    assert current_link.status_code == 200
    assert current_link.get_json()["status"] == "verified"

    unknown = client.post("/api/auth/resend", json={"email": "unknown@example.com"})
    assert unknown.status_code == 200
    assert unknown.get_json()["status"] == "verification_requested"
    assert "dev_verify_url" not in unknown.get_json()


def test_password_reset_is_single_use_and_revokes_other_sessions(auth_app):
    first_client = auth_app.app.test_client()
    second_client = auth_app.app.test_client()
    reset_client = auth_app.app.test_client()
    registration = _register(first_client, "reset@example.com")
    _verify(first_client, registration["dev_verify_url"])

    second_login = second_client.post(
        "/api/auth/login",
        json={"email": "reset@example.com", "password": PASSWORD},
    )
    assert second_login.status_code == 200
    assert first_client.get("/api/auth/me").get_json()["authenticated"] is True
    assert second_client.get("/api/auth/me").get_json()["authenticated"] is True

    requested = reset_client.post(
        "/api/auth/forgot-password",
        json={"email": "reset@example.com"},
    )
    assert requested.status_code == 200
    token = _reset_token(requested.get_json()["dev_reset_url"])
    reset = reset_client.post(
        "/api/auth/reset-password",
        json={"token": token, "password": NEW_PASSWORD},
    )
    assert reset.status_code == 200

    assert first_client.get("/api/auth/me").get_json() == {"authenticated": False}
    assert second_client.get("/api/auth/me").get_json() == {"authenticated": False}
    reused = reset_client.post(
        "/api/auth/reset-password",
        json={"token": token, "password": NEW_PASSWORD},
    )
    assert reused.status_code == 400
    assert reused.get_json()["code"] == "invalid_reset_token"
    old_login = reset_client.post(
        "/api/auth/login",
        json={"email": "reset@example.com", "password": PASSWORD},
    )
    new_login = reset_client.post(
        "/api/auth/login",
        json={"email": "reset@example.com", "password": NEW_PASSWORD},
    )
    assert old_login.status_code == 401
    assert new_login.status_code == 200


def test_forgot_password_response_does_not_disclose_account_existence(auth_app):
    known_client = auth_app.app.test_client()
    registration = _register(known_client, "known@example.com")
    _verify(known_client, registration["dev_verify_url"])
    unknown_client = auth_app.app.test_client()

    known = known_client.post(
        "/api/auth/forgot-password",
        json={"email": "known@example.com"},
    ).get_json()
    unknown = unknown_client.post(
        "/api/auth/forgot-password",
        json={"email": "unknown@example.com"},
    ).get_json()

    assert known["status"] == unknown["status"] == "reset_requested"
    assert set(known) - {"dev_reset_url"} == set(unknown)
    assert "dev_reset_url" in known
    assert "dev_reset_url" not in unknown


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_microsoft_oauth_uses_pkce_stable_subject_and_clears_state(auth_app, monkeypatch):
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-id")
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "client-secret")
    client = auth_app.app.test_client()
    started = client.get(
        "/api/auth/oauth/microsoft?next=//untrusted.example/path",
        follow_redirects=False,
    )
    location = urlparse(started.headers["Location"])
    params = parse_qs(location.query)
    assert started.status_code == 302
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"][0]
    state = params["state"][0]
    decoded_state = auth_app._oauth_serializer().loads(state)
    assert decoded_state["next"] == "/submit"
    assert "HttpOnly" in started.headers["Set-Cookie"]
    assert "SameSite=Lax" in started.headers["Set-Cookie"]

    with client.session_transaction() as session_data:
        verifier = session_data[auth_app.OAUTH_PKCE_SESSION_KEY]
    token_request = {}

    def fake_post(_url, *, data, timeout):
        token_request.update(data)
        assert timeout == 10
        return _FakeResponse({"access_token": "access-token"})

    monkeypatch.setattr(auth_app.requests, "post", fake_post)
    monkeypatch.setattr(
        auth_app.requests,
        "get",
        lambda _url, *, headers, timeout: _FakeResponse({
            "sub": "microsoft-subject-1",
            "preferred_username": "oauth@example.com",
        }),
    )
    callback = client.get(
        f"/api/auth/oauth/microsoft/callback?code=code-1&state={state}",
        follow_redirects=False,
    )

    assert callback.status_code == 302
    assert callback.headers["Location"].endswith("/submit")
    assert token_request["code_verifier"] == verifier
    assert "Max-Age=0" in callback.headers["Set-Cookie"]
    me = client.get("/api/auth/me").get_json()
    assert me["authenticated"] is True
    assert me["auth_provider"] == "microsoft"


def test_oauth_email_collision_preserves_existing_password_account(auth_app, monkeypatch):
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-id")
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "client-secret")
    password_client = auth_app.app.test_client()
    registration = _register(password_client, "existing@example.com")
    assert _verify(password_client, registration["dev_verify_url"]).status_code == 200

    oauth_client = auth_app.app.test_client()
    started = oauth_client.get("/api/auth/oauth/microsoft", follow_redirects=False)
    state = parse_qs(urlparse(started.headers["Location"]).query)["state"][0]
    monkeypatch.setattr(
        auth_app.requests,
        "post",
        lambda _url, *, data, timeout: _FakeResponse({"access_token": "access-token"}),
    )
    monkeypatch.setattr(
        auth_app.requests,
        "get",
        lambda _url, *, headers, timeout: _FakeResponse({
            "sub": "different-microsoft-subject",
            "preferred_username": "existing@example.com",
        }),
    )

    callback = oauth_client.get(
        f"/api/auth/oauth/microsoft/callback?code=code-1&state={state}",
        follow_redirects=False,
    )

    assert callback.status_code == 302
    fragment = parse_qs(urlparse(callback.headers["Location"]).fragment)
    assert fragment["oauth_error"] == [
        "An account already exists with this email address. Sign in using its "
        "existing method; automatic account linking is disabled."
    ]
    assert oauth_client.get("/api/auth/me").get_json() == {"authenticated": False}
    password_profile = password_client.get("/api/auth/me").get_json()
    assert password_profile["authenticated"] is True
    assert password_profile["auth_provider"] == "password"
    assert auth_app.get_user("existing@example.com")["auth_provider"] == "password"


def test_oauth_provider_errors_are_not_reflected_to_the_frontend(auth_app, monkeypatch):
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-id")
    monkeypatch.setenv("MICROSOFT_CLIENT_SECRET", "client-secret")
    client = auth_app.app.test_client()

    response = client.get(
        "/api/auth/oauth/microsoft/callback"
        "?error=server_error&error_description=%3Cscript%3Edanger%3C%2Fscript%3E",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "danger" not in response.headers["Location"]
    assert "identity+provider+could+not+complete" in response.headers["Location"].lower()


def test_google_oauth_rejects_an_unverified_email_claim(auth_app, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    client = auth_app.app.test_client()
    started = client.get("/api/auth/oauth/google", follow_redirects=False)
    state = parse_qs(urlparse(started.headers["Location"]).query)["state"][0]
    monkeypatch.setattr(
        auth_app.requests,
        "post",
        lambda _url, *, data, timeout: _FakeResponse({"access_token": "access-token"}),
    )
    monkeypatch.setattr(
        auth_app.requests,
        "get",
        lambda _url, *, headers, timeout: _FakeResponse({
            "sub": "google-subject-1",
            "email": "unverified@example.com",
            "email_verified": False,
        }),
    )

    callback = client.get(
        f"/api/auth/oauth/google/callback?code=code-1&state={state}",
        follow_redirects=False,
    )

    assert callback.status_code == 302
    assert "did+not+confirm" in callback.headers["Location"]
    assert client.get("/api/auth/me").get_json() == {"authenticated": False}
