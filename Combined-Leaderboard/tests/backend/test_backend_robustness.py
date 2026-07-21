import io
import importlib
import hashlib
import json
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone

import pytest


from file_security import FileSecurityValidator  # noqa: E402
from leaderboard_store import LeaderboardStore  # noqa: E402
from models.tasks import Diagnostics, GroupResult, TaskScore  # noqa: E402


def test_upload_validator_honors_custom_max_size():
    stream = io.BytesIO(b'{"question_id":"s1","answer":"A"}\n')

    is_valid, error, safe_name = FileSecurityValidator.validate_and_secure_upload(
        stream,
        "answers.jsonl",
        max_size=5,
    )

    assert is_valid is False
    assert error == "File exceeds maximum size of 5 bytes"
    assert safe_name is None
    assert stream.tell() == 0


@pytest.mark.parametrize("filename", ["answers\nforged.jsonl", "answers\r.jsonl", "answers\x7f.jsonl"])
def test_upload_validator_rejects_filename_control_characters(filename):
    is_valid, error = FileSecurityValidator.validate_filename(filename)

    assert is_valid is False
    assert error == "Filename contains invalid characters or patterns"


def test_invalid_leaderboard_limit_returns_400():
    from web.app import app  # noqa: E402

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/api/leaderboard/visual-cognition?limit=abc")

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_limit"


def test_compatibility_leaderboard_uses_current_store(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    rows = [
        {
            "model_id": "model-1",
            "model_name": "Model One",
            "has_perception": True,
            "perception_accuracy": 0.75,
            "has_cognition": False,
            "cognition_accuracy": None,
            "complete": False,
            "vci": 0.75,
        },
        {
            "model_id": "model-2",
            "model_name": "Model Two",
            "has_perception": True,
            "perception_accuracy": 0.9,
            "has_cognition": True,
            "cognition_accuracy": 0.5,
            "complete": True,
            "vci": 0.7,
        },
    ]
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "visual_cognition_leaderboard",
        lambda limit: [dict(row) for row in rows],
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/leaderboard?benchmark=do_you_see_me")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["scope"] == "do_you_see_me"
    assert [row["model_id"] for row in payload["leaderboard"]] == [
        "model-2",
        "model-1",
    ]
    assert [row["rank"] for row in payload["leaderboard"]] == [1, 2]


def test_legacy_public_submission_detail_route_is_retired():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/submission/legacy-id")

    assert response.status_code == 410
    assert response.get_json()["code"] == "legacy_submission_details_endpoint"


def test_backend_is_api_only_and_does_not_serve_frontend_routes():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        service = client.get("/")
        api_service = client.get("/api")
        old_frontend_route = client.get("/leaderboard")
        old_static_route = client.get("/static/react-app/index.html")

    assert service.status_code == 200
    assert service.get_json()["api_only"] is True
    assert service.headers["X-Request-Id"]
    assert api_service.status_code == 200
    assert api_service.get_json()["health"] == "/api/health"
    assert api_service.get_json()["liveness"] == "/api/health/live"
    assert api_service.get_json()["readiness"] == "/api/readiness"
    assert old_frontend_route.status_code == 404
    assert old_frontend_route.get_json()["code"] == "not_found"
    assert old_static_route.status_code == 404
    assert old_static_route.get_json()["code"] == "not_found"


def test_api_cors_supports_credentialed_frontend_requests():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True
    origin = web_app_module.CORS_ORIGINS[0]

    with web_app_module.app.test_client() as client:
        response = client.options(
            "/api/auth/me",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-CSRF-Token",
            },
        )
        rejected = client.options(
            "/api/auth/me",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == origin
    assert response.headers["Access-Control-Allow-Credentials"] == "true"
    exposed = response.headers["Access-Control-Expose-Headers"]
    assert "Content-Disposition" in exposed
    assert "X-Request-Id" in exposed
    assert "Access-Control-Allow-Origin" not in rejected.headers


def test_auth_me_reports_test_user_when_submission_auth_disabled(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(web_app_module, "TEST_SUBMISSION_USER", "test@example.com")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 200
    data = response.get_json()
    assert data["authenticated"] is True
    assert data["auth_disabled"] is True
    assert data["email"] == "test@example.com"
    assert data["quota"] is None
    assert data["email_verified"] is True
    assert data["auth_provider"] == "development"


def test_auth_me_issues_csrf_token_for_authenticated_session(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(
        web_app_module,
        "get_user",
        lambda email: {
            "email": email,
            "email_verified": True,
            "auth_provider": "password",
            "session_version": 0,
        },
    )
    monkeypatch.setattr(
        web_app_module,
        "quota_status",
        lambda email: {"limit": 3, "used": 0, "remaining": 3, "reset_at": None},
    )
    web_app_module.app.config["TESTING"] = True
    web_app_module.app.config["SESSION_COOKIE_SECURE"] = False

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "user@example.com"
            sess[web_app_module.AUTH_SESSION_VERSION_KEY] = 0
        response = client.get("/api/auth/me")

    assert response.status_code == 200
    data = response.get_json()
    assert data["authenticated"] is True
    assert data["email_verified"] is True
    assert data["auth_provider"] == "password"
    assert data["csrf_token"]


def test_admin_endpoint_distinguishes_missing_auth_from_missing_role(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    web_app_module.app.config["TESTING"] = True

    monkeypatch.setattr(web_app_module, "current_user_email", lambda: None)
    with web_app_module.app.test_client() as client:
        unauthenticated = client.get("/api/admin/submissions")

    assert unauthenticated.status_code == 401
    assert unauthenticated.get_json()["code"] == "auth_required"

    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    with web_app_module.app.test_client() as client:
        non_admin = client.get("/api/admin/submissions")

    assert non_admin.status_code == 403
    assert non_admin.get_json()["code"] == "admin_required"


def test_admin_actions_reject_non_object_json(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "admin@example.com")
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post("/api/admin/rescore", json=[])

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_json_object"


def test_admin_rebuild_refuses_to_publish_a_truncated_result_set(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "admin@example.com")
    monkeypatch.setattr(
        web_app_module,
        "latest_visible_scored_submission_ids",
        lambda limit: ["submission-1", "submission-2"],
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "replace_all_results",
        lambda *_args: pytest.fail("A truncated rebuild must not be published"),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post("/api/admin/rescore", json={"limit": 1})

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "leaderboard_rebuild_limit_too_low"
    assert payload["minimum_required"] == 2


def test_admin_rebuild_preserves_cache_when_any_current_score_fails(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "admin@example.com")
    monkeypatch.setattr(
        web_app_module,
        "latest_visible_scored_submission_ids",
        lambda limit: ["submission-bad"],
    )
    monkeypatch.setattr(
        web_app_module,
        "_rescore_stored_submission",
        lambda _submission_id: (
            None,
            {"code": "rescore_failed", "error": "Stored answers are unavailable."},
        ),
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "replace_all_results",
        lambda *_args: pytest.fail("An incomplete rebuild must not be published"),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post("/api/admin/rescore", json={})

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "leaderboard_rebuild_incomplete"
    assert payload["rescore_errors"] == 1


def test_admin_can_inspect_and_run_verified_server_backup(monkeypatch, tmp_path):
    web_app_module = importlib.import_module("web.app")

    class FakeBackupScheduler:
        def status(self, *, include_error=False):
            return "healthy", {
                "enabled": True,
                "interval_hours": 48,
                "retention_count": 15,
                "backup_count": 1,
            }

        def run_if_due(self, *, force=False):
            assert force is True
            return tmp_path / "ms-vista-backup-test.zip"

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "admin@example.com")
    monkeypatch.setattr(web_app_module, "backup_scheduler", FakeBackupScheduler())
    web_app_module.app.config["TESTING"] = True
    web_app_module.app.config["SESSION_COOKIE_SECURE"] = False

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "admin@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        status_response = client.get("/api/admin/backups/status")
        run_response = client.post(
            "/api/admin/backups/run",
            json={},
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert status_response.status_code == 200
    assert status_response.get_json()["backup"]["interval_hours"] == 48
    assert run_response.status_code == 201
    assert run_response.get_json()["filename"] == "ms-vista-backup-test.zip"


def test_admin_backup_download_is_post_only_and_csrf_protected(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "admin@example.com")
    monkeypatch.setattr(
        web_app_module,
        "create_backup_archive",
        lambda: (io.BytesIO(b"verified-backup"), "backup.zip", {}),
    )
    monkeypatch.setattr(
        web_app_module,
        "validate_backup_archive",
        lambda _archive: {"sqlite_snapshots": 1},
    )
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "admin@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        get_response = client.get("/api/admin/backups/download")
        missing_csrf = client.post("/api/admin/backups/download")
        download = client.post(
            "/api/admin/backups/download",
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert get_response.status_code == 405
    assert missing_csrf.status_code == 403
    assert missing_csrf.get_json()["code"] == "csrf_required"
    assert download.status_code == 200
    assert download.data == b"verified-backup"


def test_authenticated_mutating_route_requires_csrf(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    web_app_module.app.config["TESTING"] = True
    web_app_module.app.config["SESSION_COOKIE_SECURE"] = False

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "user@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"

        missing = client.post("/api/auth/logout")
        allowed = client.post(
            "/api/auth/logout",
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert missing.status_code == 403
    assert missing.get_json()["code"] == "csrf_required"
    assert allowed.status_code == 200


def test_login_endpoint_is_csrf_exempt(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post("/api/auth/login", json={})

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "missing_credentials"
    assert data["field_errors"] == {
        "email": "Email is required.",
        "password": "Password is required.",
    }


def test_login_endpoint_rejects_malformed_email():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "not-an-email", "password": "valid-length-password"},
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "invalid_email"
    assert data["field_errors"] == {"email": "Enter a valid email address."}


def test_auth_session_reads_do_not_inherit_low_global_rate_limit():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        responses = [
            client.get(
                "/api/auth/me",
                environ_overrides={"REMOTE_ADDR": "203.0.113.55"},
            )
            for _ in range(55)
        ]

    assert all(response.status_code == 200 for response in responses)
    assert all(response.get_json() == {"authenticated": False} for response in responses)


def test_registration_database_failure_is_actionable_and_sanitized(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    def fail_registration(_email, _password):
        raise RuntimeError("database password must never reach the client")

    monkeypatch.setattr(web_app_module, "register_user", fail_registration)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/auth/register",
            json={"email": "valid@example.com", "password": "strong-password"},
        )

    assert response.status_code == 503
    data = response.get_json()
    assert data["code"] == "registration_unavailable"
    assert data["retryable"] is True
    assert data["request_id"]
    assert "database password" not in data["error"]


def test_password_reset_reports_unavailable_email_service(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(
        web_app_module,
        "_email_delivery_health",
        lambda: ("unhealthy", {"production_ready": False}),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/auth/forgot-password",
            json={"email": "valid@example.com"},
        )

    assert response.status_code == 503
    data = response.get_json()
    assert data["code"] == "email_delivery_unavailable"
    assert data["retryable"] is True
    assert "No reset email was sent" in data["error"]


def test_email_health_rejects_an_invalid_send_timeout(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setenv("ACS_CONNECTION_STRING", "endpoint=https://example.invalid/;accesskey=test")
    monkeypatch.setenv("ACS_SENDER_ADDRESS", "sender@example.com")
    monkeypatch.setenv("EMAIL_SEND_TIMEOUT_SECONDS", "0")

    status, details = web_app_module._email_delivery_health()

    assert status == "unhealthy"
    assert "EMAIL_SEND_TIMEOUT_SECONDS must be a positive number" in details["errors"]


def test_submission_missing_file_returns_field_error(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/tasks/do_you_see_me/submit",
            data={"model_name": "Error Contract Model"},
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "missing_submission_file"
    assert data["field_errors"]["file"]


@pytest.mark.parametrize(
    ("content", "expected_code", "expected_validation"),
    [
        (
            b'{"question_id":"s1","answer":"A"}\n',
            "missing_sample_outputs",
            {"count": 1, "question_ids": ["s2"]},
        ),
        (
            b'{"question_id":"s1","answer":""}\n'
            b'{"question_id":"s2","answer":"B"}\n',
            "empty_sample_outputs",
            {"count": 1},
        ),
        (
            b'{"question_id":"s1","answer":"A"}\n{not json\n',
            "invalid_jsonl_syntax",
            {"line_number": 2},
        ),
    ],
)
def test_submission_endpoint_propagates_structured_file_errors(
    monkeypatch, content, expected_code, expected_validation
):
    web_app_module = importlib.import_module("web.app")
    from scoring.task_scorer import TaskScorer  # noqa: E402

    scorer = TaskScorer("do_you_see_me")
    scorer._questions = {}
    scorer._gt = {
        "s1": {"answer": "A", "group": "g"},
        "s2": {"answer": "B", "group": "g"},
    }
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(web_app_module, "_submission_model_meta", lambda _task_id, *_args: ({}, None))
    monkeypatch.setattr(web_app_module, "_enforce_quota", lambda _task_id, _model_name, _model_id: (None, None))
    monkeypatch.setitem(web_app_module.task_scorers, "do_you_see_me", scorer)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/tasks/do_you_see_me/submit",
            data={
                "model_name": "Validation Error Model",
                "file": (io.BytesIO(content), "responses.jsonl"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == expected_code
    assert data["validation"]["code"] == expected_code
    assert data["field_errors"]["file"].startswith("Correct the JSONL issue")
    for key, value in expected_validation.items():
        assert data["validation"][key] == value


def test_submission_history_storage_failure_has_retry_guidance(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    def fail_history(**_kwargs):
        raise RuntimeError("private storage detail")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(web_app_module, "list_submissions", fail_history)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/submissions/mine")

    assert response.status_code == 503
    data = response.get_json()
    assert data["code"] == "submission_history_unavailable"
    assert data["retryable"] is True
    assert "private storage detail" not in data["error"]


def test_member_can_register_and_list_owned_models(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    created = {}
    model = {
        "model_id": "mdl_member",
        "model_name": "Member Model",
        "owner_email": "member@example.com",
        "organization": "Example Lab",
        "access": "open_weights",
        "base_model": "Example Base",
        "training_data": "Public multimodal data.",
        "benchmarks": {},
    }

    def create(owner_email, display_name, meta):
        created.update(owner_email=owner_email, display_name=display_name, meta=meta)
        return model

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(web_app_module, "create_registered_model", create)
    monkeypatch.setattr(web_app_module, "list_registered_models", lambda owner: [model] if owner else [])
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        response = client.post(
            "/api/models",
            json={
                "model_name": "Member Model",
                "organization": "Example Lab",
                "access": "open_weights",
                "parameter_count": "7B",
                "paper_url": "https://example.com/paper",
            },
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )
        listed = client.get("/api/models/mine")

    assert response.status_code == 201
    assert response.get_json()["model"]["model_id"] == "mdl_member"
    assert created["owner_email"] == "member@example.com"
    assert created["display_name"] == "Member Model"
    assert created["meta"]["organization"] == "Example Lab"
    assert "base_model" not in created["meta"]
    assert "training_data" not in created["meta"]
    assert listed.status_code == 200
    assert listed.get_json()["count"] == 1


def test_model_registration_returns_precise_validation_and_conflict_errors(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(
        web_app_module,
        "create_registered_model",
        lambda *_args: (_ for _ in ()).throw(web_app_module.ModelNameConflictError()),
    )
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        invalid = client.post(
            "/api/models",
            json={"model_name": "Member Model"},
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )
        conflict = client.post(
            "/api/models",
            json={
                "model_name": "Member Model",
                "organization": "Example Lab",
                "access": "open_weights",
            },
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert invalid.status_code == 400
    assert invalid.get_json()["code"] == "invalid_model_registration"
    assert set(invalid.get_json()["field_errors"]) >= {
        "organization",
        "access",
    }
    assert "base_model" not in invalid.get_json()["field_errors"]
    assert "training_data" not in invalid.get_json()["field_errors"]
    assert conflict.status_code == 409
    assert conflict.get_json()["code"] == "model_name_conflict"
    assert conflict.get_json()["field_errors"]["model_name"]


def test_submission_rejects_model_owned_by_another_account(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(web_app_module, "get_owned_model", lambda *_args: None)
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        response = client.post(
            "/api/tasks/do_you_see_me/submit",
            data={
                "model_id": "mdl_11111111111111111111111111111111",
                "file": (io.BytesIO(b'{"question_id":"s1","answer":"A"}\n'), "responses.jsonl"),
            },
            content_type="multipart/form-data",
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert response.status_code == 404
    assert response.get_json()["code"] == "model_not_found"
    assert "different account" in response.get_json()["field_errors"]["model_id"]


@pytest.mark.parametrize("path", ["/api/models/mine", "/api/admin/submissions"])
def test_protected_routes_report_account_service_outages(monkeypatch, path):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(
        web_app_module,
        "current_user_email",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        response = client.get(path)

    assert response.status_code == 503
    assert response.get_json()["code"] == "session_check_unavailable"
    assert response.get_json()["retryable"] is True


def test_member_can_delete_only_an_owned_submission(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    calls = {}

    def delete_owned(submission_id, user_email):
        calls["delete"] = (submission_id, user_email)
        return {
            "submission_id": submission_id,
            "model_id": "mdl_member",
            "task_id": "do_you_see_me",
            "model_name": "Member Model",
            "moderation_status": "deleted",
            "previous_moderation": {
                "moderation_status": "visible",
                "moderation_reason": None,
                "moderated_by": None,
            },
        }

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(web_app_module, "delete_owned_submission", delete_owned)
    monkeypatch.setattr(
        web_app_module,
        "_refresh_public_model_task",
        lambda model_id, task_id: {
            "submission_id": calls.setdefault("refreshed", (model_id, task_id)) and None,
        },
    )
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        response = client.post(
            "/api/submissions/submission-owned-1/delete",
            json={},
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert response.status_code == 200
    assert response.get_json()["submission"]["moderation_status"] == "deleted"
    assert calls["delete"] == ("submission-owned-1", "member@example.com")
    assert calls["refreshed"] == ("mdl_member", "do_you_see_me")


def test_member_deletion_does_not_disclose_other_accounts(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(web_app_module, "delete_owned_submission", lambda *_args: None)
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        response = client.post(
            "/api/submissions/someone-elses-submission/delete",
            json={},
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert response.status_code == 404
    assert response.get_json()["code"] == "submission_not_found"
    assert "different account" in response.get_json()["error"]


def test_member_deletion_restores_visibility_when_leaderboard_write_fails(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    restored = {}
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "current_user_email", lambda: "member@example.com")
    monkeypatch.setattr(
        web_app_module,
        "delete_owned_submission",
        lambda *_args: {
            "submission_id": "submission-owned-1",
            "model_id": "mdl_member",
            "task_id": "do_you_see_me",
            "moderation_status": "deleted",
            "previous_moderation": {
                "moderation_status": "visible",
                "moderation_reason": "previous reason",
                "moderated_by": "admin@example.com",
            },
        },
    )
    refresh_calls = []

    def refresh_model_task(model_id, task_id):
        refresh_calls.append((model_id, task_id))
        if len(refresh_calls) == 1:
            raise RuntimeError("disk failure")
        return {"submission_id": "submission-owned-1"}

    monkeypatch.setattr(web_app_module, "_refresh_public_model_task", refresh_model_task)
    monkeypatch.setattr(
        web_app_module,
        "set_moderation_status",
        lambda submission_id, status, reason=None, moderated_by=None: restored.update({
            "submission_id": submission_id,
            "status": status,
            "reason": reason,
            "moderated_by": moderated_by,
        }),
    )
    monkeypatch.setattr(web_app_module.limiter, "enabled", False)
    web_app_module.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)

    with web_app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_email"] = "member@example.com"
            sess[web_app_module.CSRF_SESSION_KEY] = "known-token"
        response = client.post(
            "/api/submissions/submission-owned-1/delete",
            json={},
            headers={web_app_module.CSRF_HEADER: "known-token"},
        )

    assert response.status_code == 500
    assert response.get_json()["code"] == "submission_delete_failed"
    assert restored == {
        "submission_id": "submission-owned-1",
        "status": "visible",
        "reason": "previous reason",
        "moderated_by": "admin@example.com",
    }
    assert refresh_calls == [
        ("mdl_member", "do_you_see_me"),
        ("mdl_member", "do_you_see_me"),
    ]


def test_public_refresh_publishes_latest_remaining_visible_submission(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    published = {}
    score = TaskScore(
        task_id="do_you_see_me",
        submission_id="submission-older-visible",
        model_id="mdl_member",
        model_name="Member Model",
        submitted_at=datetime.now(timezone.utc),
        accuracy=0.4,
        total_samples=10,
        correct_samples=4,
    )
    monkeypatch.setattr(
        web_app_module,
        "latest_visible_scored_submission_id",
        lambda model_id, task_id: "submission-older-visible",
    )
    monkeypatch.setattr(
        web_app_module,
        "_rescore_stored_submission",
        lambda submission_id: ((score, {"user_email": "member@example.com"}), None),
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "add_result",
        lambda result, submitted_by=None: published.update({
            "submission_id": result.submission_id,
            "submitted_by": submitted_by,
        }),
    )

    refresh = web_app_module._refresh_public_model_task(
        "mdl_member",
        "do_you_see_me",
    )

    assert refresh["submission_id"] == "submission-older-visible"
    assert published == {
        "submission_id": "submission-older-visible",
        "submitted_by": "member@example.com",
    }


def test_spatial_submit_rejects_demo_bundle_before_accepting_files(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(
        web_app_module,
        "_spatial_bundle_health",
        lambda: ("unhealthy", {"error": "demo bundle"}),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post("/api/tasks/spatial/submit")

    assert response.status_code == 503
    assert response.get_json()["code"] == "spatial_benchmark_not_ready"


def test_spatial_demo_bundle_does_not_publish_question_or_template_files(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(
        web_app_module,
        "_spatial_bundle_health",
        lambda: ("unhealthy", {"error": "demo bundle"}),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        questions = client.get("/api/tasks/spatial/questions")
        template = client.get("/api/tasks/spatial/template.jsonl")

    assert questions.status_code == 503
    assert template.status_code == 503
    assert questions.get_json()["code"] == "spatial_task_bundle_unavailable"
    assert template.get_json()["code"] == "spatial_task_bundle_unavailable"


def test_spatial_info_remains_available_when_private_bundle_is_missing(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(
        web_app_module,
        "_spatial_bundle_health",
        lambda: (
            "unhealthy",
            {
                "production_ready": False,
                "samples": 0,
                "error": "Private spatial ground truth is unavailable.",
            },
        ),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/tasks/spatial/info")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["submission_ready"] is False
    assert payload["bundle_status"] == "unhealthy"
    assert payload["total_samples"] == 0
    assert len(payload["conditions"]) == 6
    assert payload["required_uploads"] == ["spatial_reasoning_submission.zip"]
    assert payload["upload_processing"] == "in_memory"
    assert payload["max_upload_bytes"] == web_app_module.MAX_SPATIAL_ARCHIVE_BYTES


def test_visual_task_info_separates_paper_and_release_suite_counts():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        dysm = client.get("/api/tasks/do_you_see_me/info").get_json()
        minds_eye = client.get("/api/tasks/minds_eye/info").get_json()

    assert dysm["paper_total_samples"] == 2612
    assert dysm["total_samples"] == 4500
    assert dysm["score_method"] == "dimension_balanced_task_macro"
    assert minds_eye["paper_total_samples"] == 800
    assert minds_eye["total_samples"] == 799
    assert minds_eye["grading"]["random_baseline"] == pytest.approx(11 / 48)


def _make_spatial_package(
    submission: bytes,
    manifest: bytes = b"{}",
    report: bytes = b"{}",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("submission.jsonl", submission)
        package.writestr("run_manifest.json", manifest)
        package.writestr("leaderboard.json", report)
    return output.getvalue()


def test_spatial_submit_rejects_separate_files(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(
        web_app_module,
        "_spatial_bundle_health",
        lambda: ("healthy", {"production_ready": True}),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/tasks/spatial/submit",
            data={
                "file": (io.BytesIO(b'{"question_id":"q1","answer":"A"}\n'), "submission.jsonl"),
                "run_manifest": (io.BytesIO(b"{}"), "run_manifest.json"),
                "model_name": "Test Model",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "invalid_spatial_upload_parts"
    assert set(data["field_errors"]) == {"file"}


def test_spatial_upload_stream_is_memory_only():
    web_app_module = importlib.import_module("web.app")
    from flask import request

    package = _make_spatial_package(b'{"question_id":"q1","answer":"A"}\n')
    with web_app_module.app.test_request_context(
        "/api/tasks/spatial/submit",
        method="POST",
        data={"file": (io.BytesIO(package), "spatial_reasoning_submission.zip")},
    ):
        assert web_app_module.app.preprocess_request() is None
        assert isinstance(request.files["file"].stream, io.BytesIO)
        assert request.max_content_length == web_app_module.MAX_SPATIAL_MULTIPART_BYTES


def test_spatial_harness_download_excludes_runtime_data():
    from web.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/api/spatial/harness")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        names = archive.namelist()
    assert "spatial_reasoning/run_eval.sh" in names
    assert "spatial_reasoning/spatial_contract.py" in names
    assert all("__pycache__" not in name for name in names)
    assert all("/LMUData/" not in name and "/results/" not in name for name in names)


def test_spatial_single_zip_submission_reaches_scoring_and_storage(monkeypatch, tmp_path):
    web_app_module = importlib.import_module("web.app")

    manifest_path = tmp_path / "manifest.json"
    template_path = tmp_path / "submission_template.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    manifest_path.write_bytes(b'{"benchmark_version":"test-v1"}\n')
    template_path.write_bytes(b'{"question_id":"q1","answer":""}\n')
    questions_path.write_bytes(b'{"question_id":"q1"}\n')
    monkeypatch.setattr(web_app_module, "SPATIAL_MANIFEST_FILE", manifest_path)
    monkeypatch.setitem(
        web_app_module.TASKS["spatial"]["paths"],
        "template_jsonl",
        template_path,
    )
    monkeypatch.setitem(
        web_app_module.TASKS["spatial"]["paths"],
        "questions_jsonl",
        questions_path,
    )

    conditions = list(web_app_module.EVAL_CONDITIONS)
    answer_records = [
        {
            "row_index": index,
            "line_number": index,
            "dataset": "BLINK",
            "question_id": "q1",
            "evaluation_group": "q1",
            "condition": condition,
            "answer": "E" if condition.startswith("no_image_plus_") else "A",
            "correct": True,
            "judge_method": "qwen_llm_judge",
            "judge_attempts": 1,
        }
        for index, condition in enumerate(conditions, start=1)
    ]
    submission = ("\n".join(json.dumps(row) for row in answer_records) + "\n").encode("utf-8")
    upload_package = _make_spatial_package(submission)
    stored = {}
    operations = []

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(web_app_module, "_spatial_bundle_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_submission_model_meta", lambda _task_id, *_args: ({}, None))
    monkeypatch.setattr(web_app_module, "_enforce_quota", lambda _task_id, _model_name, _model_id: (None, None))
    monkeypatch.setattr(
        web_app_module,
        "parse_spatial_evidence",
        lambda *_args, **_kwargs: (answer_records, {"summary": {}}, {}),
    )
    monkeypatch.setattr(
        web_app_module,
        "validate_run_manifest",
        lambda *_args, **_kwargs: {
            "schema_version": "ms-vista-spatial-run/v2",
            "benchmark_version": "test-v1",
        },
    )
    monkeypatch.setattr(
        web_app_module,
        "validate_spatial_report",
        lambda *_args, **_kwargs: {"summary": {"main_noncot": 1.0}},
    )
    monkeypatch.setattr(
        web_app_module,
        "build_spatial_task_score",
        lambda _report, model_name, model_meta, run_metadata: TaskScore(
            task_id="spatial",
            submission_id="spatial-public-evidence-1",
            model_name=model_name,
            submitted_at=datetime.now(timezone.utc),
            accuracy=1.0,
            macro_accuracy=1.0,
            total_samples=1,
            correct_samples=1,
            groups={
                "BLINK": GroupResult(
                    name="BLINK",
                    total_samples=1,
                    correct_samples=1,
                    accuracy=1.0,
                )
            },
            diagnostics=Diagnostics(
                conditions_present=conditions,
                standard_accuracy=1.0,
                cot_accuracy=1.0,
                cot_delta=0.0,
                shortcut_score=1.0,
                hallucination_resistance=1.0,
            ),
            model_meta=model_meta,
            metadata={"spatial_run": run_metadata},
        ),
    )
    monkeypatch.setattr(
        web_app_module,
        "store_submission_answers",
        lambda *args, **kwargs: (operations.append("store"), stored.update(kwargs)),
    )
    monkeypatch.setattr(
        web_app_module,
        "finalize_submission",
        lambda *_args, **_kwargs: operations.append("finalize"),
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "add_result",
        lambda score, submitted_by=None: (operations.append("publish"), score.to_dict())[1],
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/tasks/spatial/submit",
            data={
                "model_name": "Spatial Test Model",
                "file": (
                    io.BytesIO(upload_package),
                    "spatial_reasoning_submission.zip",
                ),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    assert body["success"] is True
    assert body["macro_accuracy"] == 1.0
    assert body["diagnostics"]["conditions_present"] == conditions
    assert body["metadata"]["spatial_run"]["benchmark_version"] == "test-v1"
    assert len(stored["records"]) == len(conditions)
    assert stored["file_sha256"] == hashlib.sha256(upload_package).hexdigest()
    assert {artifact["artifact_name"] for artifact in stored["artifacts"]} == {
        "spatial_reasoning_submission.zip",
        "submission.jsonl",
        "run_manifest.json",
        "leaderboard.json",
    }
    assert set(stored["spatial_contract"]) == {
        "manifest",
        "template",
        "questions",
        "manifest_sha256",
    }
    assert hashlib.sha256(stored["spatial_contract"]["manifest"]).hexdigest() == (
        stored["spatial_contract"]["manifest_sha256"]
    )
    assert operations == ["store", "finalize", "publish"]


def test_public_spatial_evidence_routes_require_visibility_and_preserve_hashes(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    submission_id = "spatial-public-1"
    artifact_names = list(web_app_module.SPATIAL_PUBLIC_ARTIFACT_NAMES)
    content = b'{"question_id":"q1","answer":"A"}\n'
    digest = hashlib.sha256(content).hexdigest()
    visible = {"value": True}

    def evidence_lookup(_submission_id):
        if not visible["value"]:
            return None
        return {
            "submission_id": submission_id,
            "model_name": "Public Spatial Model",
            "task_id": "spatial",
            "artifacts": [
                {"name": name, "url": f"/artifact/{name}"}
                for name in artifact_names
            ],
        }

    def artifact_lookup(_submission_id, artifact_name):
        if not visible["value"]:
            return None
        return {
            "submission_id": submission_id,
            "artifact_name": artifact_name,
            "media_type": "application/x-ndjson",
            "size_bytes": len(content),
            "sha256": digest,
            "content": content,
        }

    monkeypatch.setattr(web_app_module, "get_public_spatial_evidence", evidence_lookup)
    monkeypatch.setattr(web_app_module, "get_public_spatial_artifact", artifact_lookup)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        metadata_response = client.get(
            f"/api/public/submissions/{submission_id}/evidence"
        )
        artifact_response = client.get(
            f"/api/public/submissions/{submission_id}/answers.jsonl"
        )
        cached_response = client.get(
            f"/api/public/submissions/{submission_id}/answers.jsonl",
            headers={"If-None-Match": f'"{digest}"'},
        )
        visible["value"] = False
        hidden_response = client.get(
            f"/api/public/submissions/{submission_id}/evidence"
        )

    assert metadata_response.status_code == 200
    assert metadata_response.get_json()["model_name"] == "Public Spatial Model"
    assert artifact_response.status_code == 200
    assert artifact_response.data == content
    assert artifact_response.headers["X-Evidence-SHA256"] == digest
    assert artifact_response.headers["ETag"] == f'"{digest}"'
    assert cached_response.status_code == 304
    assert hidden_response.status_code == 404
    assert hidden_response.get_json()["code"] == "public_evidence_not_found"


def test_scored_submission_is_not_refunded_when_cache_publication_fails(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    operations = []

    class FakeScorer:
        def parse_submission_text_with_records(self, _text):
            return (
                {"standard": {"q1": "A"}},
                {},
                [{
                    "row_index": 0,
                    "line_number": 1,
                    "question_id": "q1",
                    "condition": "standard",
                    "answer": "A",
                }],
            )

        def score_predictions(self, _predictions, *, model_name, **_kwargs):
            return TaskScore(
                task_id="do_you_see_me",
                submission_id="submission-pending-publication",
                model_name=model_name,
                submitted_at=datetime.now(timezone.utc),
                accuracy=1.0,
                total_samples=1,
                correct_samples=1,
            )

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)
    monkeypatch.setattr(web_app_module, "_submission_model_meta", lambda *_args: ({}, None))
    monkeypatch.setattr(web_app_module, "_enforce_quota", lambda *_args: (123, None))
    monkeypatch.setattr(web_app_module, "find_owned_model_by_name", lambda *_args: None)
    monkeypatch.setitem(web_app_module.task_scorers, "do_you_see_me", FakeScorer())
    monkeypatch.setattr(
        web_app_module,
        "store_submission_answers",
        lambda *_args, **_kwargs: operations.append("store"),
    )
    monkeypatch.setattr(
        web_app_module,
        "finalize_submission",
        lambda _submission_id, success: operations.append(f"finalize:{success}"),
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "add_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.post(
            "/api/tasks/do_you_see_me/submit",
            data={
                "model_name": "Pending Publication Model",
                "file": (io.BytesIO(b'{"question_id":"q1","answer":"A"}\n'), "responses.jsonl"),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["code"] == "leaderboard_publication_pending"
    assert payload["success"] is True
    assert payload["retryable"] is False
    assert payload["stored"] is True
    assert payload["published"] is False
    assert operations == ["store", "finalize:True"]


def test_wrong_http_method_returns_json_error_contract():
    from web.app import app  # noqa: E402

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.put("/api/auth/login", json={})

    assert response.status_code == 405
    data = response.get_json()
    assert data["code"] == "method_not_allowed"
    assert data["request_id"]


def test_cors_preflight_allows_csrf_header():
    from web.app import app  # noqa: E402

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.options(
            "/api/tasks/do_you_see_me/submit",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-CSRF-Token",
            },
        )

    assert response.status_code == 200
    assert "X-CSRF-Token" in response.headers.get("Access-Control-Allow-Headers", "")


def test_production_auth_requires_explicit_public_urls(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.delenv("AUTH_DEV_MODE", raising=False)
    monkeypatch.delenv("ALLOW_INSECURE_SECRET", raising=False)
    monkeypatch.delenv("APP_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTEND_BASE_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("OAUTH_REDIRECT_BASE_URL", raising=False)
    monkeypatch.delenv("PRIVACY_POLICY_URL", raising=False)
    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)

    with pytest.raises(RuntimeError, match="explicit public URL configuration"):
        web_app_module._validate_production_public_urls()


def test_email_health_skipped_when_submission_auth_disabled(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", True)

    status, details = web_app_module._email_delivery_health()

    assert status == "skipped"
    assert details["production_ready"] is False


def test_ground_truth_health_requires_matching_ids_and_pinned_public_hf_revision(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    class FakeScorer:
        def __init__(self, ground_truth_ids, question_ids):
            self.ground_truth = {item: {} for item in ground_truth_ids}
            self.questions = {item: {} for item in question_ids}

    monkeypatch.setattr(
        web_app_module,
        "task_scorers",
        {"do_you_see_me": FakeScorer({"q1"}, {"q1"})},
    )
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(web_app_module, "GROUND_TRUTHS_SOURCE", "hf")
    monkeypatch.setattr(web_app_module, "GROUND_TRUTHS_HF_REVISION", "main")

    status, details = web_app_module._ground_truth_bundle_health()

    assert status == "unhealthy"
    assert details["revision_ready"] is False

    monkeypatch.setattr(web_app_module, "GROUND_TRUTHS_HF_REVISION", "a" * 40)
    status, details = web_app_module._ground_truth_bundle_health()
    assert status == "healthy"
    assert details["hf_revision_pinned"] is True

    monkeypatch.setattr(
        web_app_module,
        "task_scorers",
        {"do_you_see_me": FakeScorer({"q1", "q2"}, {"q1"})},
    )
    status, details = web_app_module._ground_truth_bundle_health()
    assert status == "unhealthy"
    assert details["tasks"]["do_you_see_me"]["missing_public_question_count"] == 1


def test_email_health_rejects_incomplete_smtp_configuration(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setattr(web_app_module, "AUTH_DEV_MODE", False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    monkeypatch.delenv("SMTP_FROM", raising=False)
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("ACS_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("ACS_ENDPOINT", raising=False)

    status, details = web_app_module._email_delivery_health()

    assert status == "unhealthy"
    assert details["production_ready"] is False
    assert "SMTP_PORT" in " ".join(details["errors"])
    assert "SMTP_FROM" in " ".join(details["errors"])


def test_email_health_rejects_invalid_acs_client_configuration(monkeypatch):
    web_app_module = importlib.import_module("web.app")

    monkeypatch.setattr(web_app_module, "SUBMISSION_AUTH_DISABLED", False)
    monkeypatch.setenv("ACS_CONNECTION_STRING", "not-a-connection-string")
    monkeypatch.setenv("ACS_SENDER_ADDRESS", "sender@example.com")
    monkeypatch.delenv("ACS_ENDPOINT", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)

    status, details = web_app_module._email_delivery_health()

    assert status == "unhealthy"
    assert details["client_ready"] is False
    assert details["production_ready"] is False


def test_submission_metadata_allows_missing_optional_canonical_fields():
    from web.app import app, _submission_model_meta  # noqa: E402

    method_description = " ".join(["method"] * 100)
    changes = " ".join(["change"] * 50)
    form_data = {
        "organization": "Example Lab",
        "model_access": "research",
        "method_description": method_description,
        "cot_used": "No",
        "prompt_template": "Answer with the final option only.",
        "changes_from_previous": changes,
    }

    with app.test_request_context("/submit", method="POST", data=form_data):
        meta, error = _submission_model_meta("do_you_see_me")

    assert error is None
    assert meta["organization"] == "Example Lab"
    assert "parameter_count" not in meta
    assert "base_model" not in meta
    assert "training_data" not in meta


@pytest.mark.parametrize(
    "paper_url",
    [
        "javascript:alert(1)",
        "https://user:secret@example.com/paper",
        "https://example.com:invalid/paper",
        "https://example.com/paper\njavascript:alert(1)",
    ],
)
def test_registered_model_rejects_unsafe_or_malformed_paper_urls(paper_url):
    from web.app import _registered_model_payload  # noqa: E402

    values, errors = _registered_model_payload({
        "model_name": "Example Model",
        "organization": "Example Lab",
        "access": "open_weights",
        "paper_url": paper_url,
    })

    assert values is None
    assert "paper_url" in errors


def test_leaderboard_store_backs_up_corrupt_json(tmp_path):
    store_file = tmp_path / "leaderboard_store.json"
    store_file.write_text("{not valid json", encoding="utf-8")
    store = LeaderboardStore(store_file)

    assert store._read() == {"models": {}}
    backups = list(tmp_path.glob("leaderboard_store.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are unavailable")
def test_leaderboard_store_restricts_cache_and_history_permissions(tmp_path):
    store_dir = tmp_path / "results"
    store = LeaderboardStore(store_dir / "leaderboard_store.json")
    score = TaskScore(
        task_id="do_you_see_me",
        submission_id="private-submission",
        model_id="mdl_private",
        model_name="Private Model",
        submitted_at=datetime.now(timezone.utc),
        accuracy=0.5,
        total_samples=2,
        correct_samples=1,
    )

    store.add_result(score, submitted_by="member@example.com")

    assert store_dir.stat().st_mode & 0o777 == 0o700
    assert store.store_file.stat().st_mode & 0o777 == 0o600
    assert store.lock_file.stat().st_mode & 0o777 == 0o600
    assert (store_dir / "submission_history.jsonl").stat().st_mode & 0o777 == 0o600


def test_leaderboard_aggregates_benchmarks_by_stable_model_id(tmp_path):
    store = LeaderboardStore(tmp_path / "leaderboard_store.json")
    submitted_at = datetime.now(timezone.utc)
    perception = TaskScore(
        task_id="do_you_see_me",
        submission_id="submission-perception",
        model_id="mdl_linked",
        model_name="Model A",
        submitted_at=submitted_at,
        accuracy=0.4,
        total_samples=10,
        correct_samples=4,
        model_meta={
            "organization": "Example Lab",
            "access": "open_weights",
            "method_description": "Perception method",
            "prompt_template": "Perception prompt",
            "cot_used": "no",
        },
    )
    cognition = TaskScore(
        task_id="minds_eye",
        submission_id="submission-cognition",
        model_id="mdl_linked",
        model_name="MODEL A renamed",
        submitted_at=submitted_at,
        accuracy=0.6,
        total_samples=10,
        correct_samples=6,
        model_meta={
            "organization": "Example Lab",
            "access": "open_weights",
            "method_description": "Cognition method",
            "prompt_template": "Cognition prompt",
            "cot_used": "yes",
        },
    )

    store.add_result(perception, submitted_by="member@example.com")
    store.add_result(cognition, submitted_by="member@example.com")
    rows = store.visual_cognition_leaderboard()

    assert len(rows) == 1
    assert rows[0]["model_id"] == "mdl_linked"
    assert rows[0]["model_name"] == "MODEL A renamed"
    assert rows[0]["complete"] is True
    assert rows[0]["perception_accuracy"] == 0.4
    assert rows[0]["cognition_accuracy"] == 0.6
    assert rows[0]["model_meta"] == {
        "organization": "Example Lab",
        "access": "open_weights",
    }
    report = store.get_model("mdl_linked")
    assert "method_description" not in report["model_meta"]
    assert report["tasks"]["do_you_see_me"]["model_meta"]["method_description"] == "Perception method"
    assert report["tasks"]["minds_eye"]["model_meta"]["method_description"] == "Cognition method"
    assert store.public_submission_ids() == [
        "submission-cognition",
        "submission-perception",
    ]
    fingerprints = store.public_submission_fingerprints()
    assert set(fingerprints) == {
        "submission-cognition",
        "submission-perception",
    }
    assert all(len(value) == 64 for value in fingerprints.values())

    assert store.remove_model_task("mdl_linked", "minds_eye") is True
    rows = store.visual_cognition_leaderboard()
    assert rows[0]["complete"] is False
    assert rows[0]["has_cognition"] is False
    assert rows[0]["vci"] is None
    assert store.statistics()["best_vci"] is None
    assert store.public_submission_ids() == ["submission-perception"]


def test_registered_model_archive_requires_deleted_submissions(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import submission_store

    engine = create_engine(f"sqlite:///{tmp_path / 'archive-model.db'}")
    submission_store.Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(submission_store, "_Session", session_factory)

    with session_factory() as session:
        session.add(submission_store.RegisteredModel(
            id="mdl_archive",
            owner_email="owner@example.com",
            display_name="Archived Model",
            normalized_name="archived model",
            organization="Example",
            access="open_weights",
            active=True,
        ))
        session.add(submission_store.Submission(
            user_email="owner@example.com",
            task_id="do_you_see_me",
            model_name="Archived Model",
            model_id="mdl_archive",
            status="scored",
            score_submission_id="submission-archive",
            moderation_status="visible",
        ))
        session.commit()

    with pytest.raises(ValueError, match="Delete every submission"):
        submission_store.archive_registered_model(
            "mdl_archive",
            "owner@example.com",
        )

    with session_factory() as session:
        row = session.query(submission_store.Submission).one()
        row.moderation_status = "deleted"
        session.commit()

    archived = submission_store.archive_registered_model(
        "mdl_archive",
        "owner@example.com",
    )

    assert archived["active"] is False
    assert submission_store.archive_registered_model(
        "mdl_archive",
        "owner@example.com",
    ) is None


def test_statistics_count_unique_ranked_models_across_tracks(tmp_path):
    store = LeaderboardStore(tmp_path / "leaderboard_store.json")
    submitted_at = datetime.now(timezone.utc)
    scores = [
        TaskScore(
            task_id="do_you_see_me",
            submission_id="visual-a",
            model_id="mdl_a",
            model_name="Model A",
            submitted_at=submitted_at,
            accuracy=0.4,
            total_samples=10,
            correct_samples=4,
        ),
        TaskScore(
            task_id="spatial",
            submission_id="spatial-a",
            model_id="mdl_a",
            model_name="Model A",
            submitted_at=submitted_at,
            accuracy=0.5,
            total_samples=10,
            correct_samples=5,
        ),
        TaskScore(
            task_id="minds_eye",
            submission_id="visual-b",
            model_id="mdl_b",
            model_name="Model B",
            submitted_at=submitted_at,
            accuracy=0.6,
            total_samples=10,
            correct_samples=6,
        ),
    ]
    for score in scores:
        store.add_result(score, submitted_by="member@example.com")

    statistics = store.statistics()

    assert statistics["total_models"] == 2
    assert statistics["ranked_models"] == 2
    assert statistics["visual_cognition_models"] == 2
    assert statistics["spatial_models"] == 1


def test_visual_leaderboard_derives_paper_aligned_scores_for_legacy_records(tmp_path):
    store = LeaderboardStore(tmp_path / "leaderboard_store.json")
    submitted_at = datetime.now(timezone.utc)
    perception = TaskScore(
        task_id="do_you_see_me",
        submission_id="legacy-perception",
        model_id="mdl_legacy",
        model_name="Legacy Model",
        submitted_at=submitted_at,
        accuracy=0.8,
        macro_accuracy=0.8,
        total_samples=5,
        correct_samples=4,
        task_spread=0.2,
        groups={
            "shape": GroupResult("shape", 4, 4, 1.0),
            "closure": GroupResult("closure", 1, 0, 0.0),
        },
        analysis={
            "dimension": {
                "2D": GroupResult("2D", 4, 4, 1.0),
                "3D": GroupResult("3D", 1, 0, 0.0),
            },
        },
    )
    cognition = TaskScore(
        task_id="minds_eye",
        submission_id="legacy-cognition",
        model_id="mdl_legacy",
        model_name="Legacy Model",
        submitted_at=submitted_at,
        accuracy=0.6,
        macro_accuracy=0.6,
        total_samples=10,
        correct_samples=6,
        task_spread=0.1,
        groups={
            "analogical_reasoning": GroupResult("analogical_reasoning", 8, 6, 0.75),
            "mental_rotation": GroupResult("mental_rotation", 2, 0, 0.0),
        },
    )

    store.add_result(perception, submitted_by="member@example.com")
    store.add_result(cognition, submitted_by="member@example.com")
    row = store.visual_cognition_leaderboard()[0]

    assert row["perception_accuracy"] == 0.5
    assert row["perception_micro_accuracy"] == 0.8
    assert row["cognition_accuracy"] == 0.375
    assert row["cognition_micro_accuracy"] == 0.6
    assert row["vci"] == 0.4375
    assert row["perception_task_spread"] == 0.2
    assert row["cognition_task_spread"] == 0.1
    assert row["task_spread"] == 0.15
    assert row["perception_dimensions"]["2D"]["meta"]["aggregation"] == "legacy_dimension_accuracy"


def test_sqlite_backup_archive_contains_consistent_db_snapshot(tmp_path):
    from backup import create_backup_archive  # noqa: E402

    source_db = tmp_path / "leaderboard.db"
    with sqlite3.connect(source_db) as con:
        con.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY, model_name TEXT)")
        con.execute("INSERT INTO submissions (model_name) VALUES (?)", ("Smoke",))
        con.commit()
    store_file = tmp_path / "leaderboard_store.json"
    store_file.write_text('{"models":{}}', encoding="utf-8")

    archive, filename, manifest = create_backup_archive(
        database_urls={"main": f"sqlite:///{source_db}"},
        extra_files={"leaderboard_store": store_file},
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    assert filename == "ms-vista-backup-20260102T030405Z.zip"
    assert manifest["sqlite_databases"][0]["exists"] is True
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        db_name = next(name for name in names if name.endswith("leaderboard.db"))
        restored = tmp_path / "restored.db"
        restored.write_bytes(zf.read(db_name))
        assert "files/leaderboard_store.json" in names
        assert "manifest.json" in names

    with sqlite3.connect(restored) as con:
        row = con.execute("SELECT model_name FROM submissions WHERE id = 1").fetchone()
        assert row == ("Smoke",)


def test_backup_scheduler_repairs_missing_mirror_after_restart(tmp_path, monkeypatch):
    import backend.backup as backup_module

    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    primary.mkdir()
    mirror.mkdir()
    latest = primary / "ms-vista-backup-20260101T000000Z.zip"
    latest.write_bytes(b"primary")
    calls = []

    def write_archive(output_dir, **_kwargs):
        destination = output_dir / "ms-vista-backup-20260101T000500Z.zip"
        destination.write_bytes(b"replacement")
        calls.append("write")
        return destination, {}

    def mirror_archive(source, mirror_dir, **_kwargs):
        destination = mirror_dir / source.name
        destination.write_bytes(source.read_bytes())
        calls.append("mirror")
        return destination, {}

    monkeypatch.setattr(backup_module, "write_backup_archive", write_archive)
    monkeypatch.setattr(backup_module, "mirror_backup_archive", mirror_archive)
    scheduler = backup_module.BackupScheduler(
        enabled=True,
        output_dir=primary,
        mirror_dir=mirror,
        interval_hours=48,
        retention_count=3,
        poll_seconds=60,
        run_on_start=False,
    )

    destination = scheduler.run_if_due()

    assert destination is not None
    assert calls == ["write", "mirror"]
    status, details = scheduler.status()
    assert status == "healthy"
    assert details["mirror_synchronized"] is True


def test_scheduled_backup_is_validated_atomic_and_retained(tmp_path):
    from backup import write_backup_archive  # noqa: E402

    source_db = tmp_path / "leaderboard.db"
    with sqlite3.connect(source_db) as con:
        con.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY, model_name TEXT)")
        con.execute("INSERT INTO submissions (model_name) VALUES (?)", ("Retained",))
        con.commit()
    cache_file = tmp_path / "leaderboard_store.json"
    cache_file.write_text('{"models":{}}', encoding="utf-8")
    output_dir = tmp_path / "backups"

    created = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset in range(3):
        destination, manifest = write_backup_archive(
            output_dir,
            retention_count=2,
            database_urls={"main": f"sqlite:///{source_db}"},
            extra_files={"leaderboard_store": cache_file},
            now=start + timedelta(days=offset * 2),
        )
        created.append(destination)
        assert manifest["validation"]["zip_crc"] == "ok"
        assert manifest["validation"]["sqlite_quick_check"] == "ok"
        assert destination.stat().st_mode & 0o777 == 0o600

    retained = sorted(output_dir.glob("ms-vista-backup-*.zip"))
    assert len(retained) == 2
    assert created[0] not in retained
    assert created[1] in retained
    assert created[2] in retained


def test_backup_can_be_mirrored_and_restored_offline(tmp_path):
    from backup import mirror_backup_archive, restore_backup_archive, write_backup_archive  # noqa: E402

    source_db = tmp_path / "leaderboard.db"
    with sqlite3.connect(source_db) as con:
        con.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY, model_name TEXT)")
        con.execute("INSERT INTO submissions (model_name) VALUES (?)", ("Recovered",))
        con.commit()
    primary = tmp_path / "primary"
    mirror = tmp_path / "mirror"
    archive, _manifest = write_backup_archive(
        primary,
        retention_count=2,
        database_urls={"main": f"sqlite:///{source_db}"},
        extra_files={},
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    mirrored, mirror_validation = mirror_backup_archive(
        archive,
        mirror,
        retention_count=2,
    )
    recovery = tmp_path / "recovery"
    restore_result = restore_backup_archive(mirrored, recovery)
    restored_db = next((recovery / "sqlite").glob("*.db"))

    assert mirror_validation["zip_crc"] == "ok"
    assert restore_result["sqlite_quick_check"] == "ok"
    assert mirrored.stat().st_mode & 0o777 == 0o600
    assert restored_db.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(restored_db) as con:
        row = con.execute("SELECT model_name FROM submissions WHERE id = 1").fetchone()
    assert row == ("Recovered",)


def test_required_offsite_backup_is_unhealthy_without_separate_mirror(tmp_path):
    from backup import BackupScheduler  # noqa: E402

    scheduler = BackupScheduler(
        enabled=True,
        output_dir=tmp_path / "backups",
        mirror_dir=None,
        require_mirror=True,
        interval_hours=48,
        retention_count=4,
        poll_seconds=60,
        run_on_start=True,
    )

    status, details = scheduler.status()

    assert status == "unhealthy"
    assert details["mirror_required"] is True
    assert details["mirror_configured"] is False


def test_backup_scheduler_runs_when_due_and_reports_health(tmp_path, monkeypatch):
    backup_module = importlib.import_module("backup")
    calls = []

    def fake_writer(output_dir, *, retention_count, now):
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"ms-vista-backup-{len(calls)}.zip"
        destination.write_bytes(b"verified")
        calls.append((retention_count, now))
        return destination, {"validation": {"sqlite_quick_check": "ok"}}

    monkeypatch.setattr(backup_module, "write_backup_archive", fake_writer)
    scheduler = backup_module.BackupScheduler(
        enabled=True,
        output_dir=tmp_path / "backups",
        interval_hours=48,
        retention_count=4,
        poll_seconds=60,
        run_on_start=True,
    )

    first = scheduler.run_if_due()
    second = scheduler.run_if_due()
    overdue_timestamp = (datetime.now(timezone.utc) - timedelta(hours=49)).timestamp()
    os.utime(first, (overdue_timestamp, overdue_timestamp))
    third = scheduler.run_if_due()
    status, details = scheduler.status()

    assert first is not None
    assert second is None
    assert third is not None
    assert len(calls) == 2
    assert status == "healthy"
    assert details["interval_hours"] == 48
    assert details["retention_count"] == 4
    assert details["backup_count"] == 2


def test_backup_scheduler_retries_after_a_failed_mirror(tmp_path, monkeypatch):
    backup_module = importlib.import_module("backup")
    writes = []
    mirrors = []

    def fake_writer(output_dir, *, retention_count, now):
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"ms-vista-backup-{len(writes)}.zip"
        destination.write_bytes(b"verified")
        writes.append(destination)
        return destination, {"validation": {"sqlite_quick_check": "ok"}}

    def fake_mirror(source, mirror_dir, *, retention_count):
        mirrors.append(source)
        if len(mirrors) == 1:
            raise OSError("temporary mirror outage")
        mirror_dir.mkdir(parents=True, exist_ok=True)
        destination = mirror_dir / source.name
        destination.write_bytes(source.read_bytes())
        return destination, {"sqlite_quick_check": "ok"}

    monkeypatch.setattr(backup_module, "write_backup_archive", fake_writer)
    monkeypatch.setattr(backup_module, "mirror_backup_archive", fake_mirror)
    scheduler = backup_module.BackupScheduler(
        enabled=True,
        output_dir=tmp_path / "primary",
        mirror_dir=tmp_path / "mirror",
        interval_hours=48,
        retention_count=4,
        poll_seconds=60,
        run_on_start=True,
    )

    with pytest.raises(OSError, match="temporary mirror outage"):
        scheduler.run_if_due()
    retried = scheduler.run_if_due()

    assert retried is not None
    assert len(writes) == 2
    assert len(mirrors) == 2
    assert scheduler.status()[0] == "healthy"


def test_sqlite_runtime_enables_wal_foreign_keys_and_busy_timeout(tmp_path):
    from sqlalchemy import create_engine
    from sqlite_runtime import configure_sqlite_engine, sqlite_connect_args  # noqa: E402

    database_url = f"sqlite:///{tmp_path / 'runtime.db'}"
    engine = create_engine(
        database_url,
        connect_args=sqlite_connect_args(database_url, 7000),
    )
    configure_sqlite_engine(engine, database_url, 7000)

    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE smoke (id INTEGER PRIMARY KEY)")
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar()
        foreign_keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar()
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar()

    assert journal_mode == "wal"
    assert foreign_keys == 1
    assert busy_timeout == 7000
    if os.name != "nt":
        assert (tmp_path / "runtime.db").stat().st_mode & 0o777 == 0o600


def test_public_auth_health_requires_a_verified_admin(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(web_app_module, "ADMIN_EMAILS", {"admin@example.com"})
    monkeypatch.setattr(web_app_module, "get_verified_admin_emails", lambda _emails: [])

    status, details = web_app_module._auth_service_health()

    assert status == "unhealthy"
    assert details["admin_addresses_configured"] == 1
    assert details["verified_admin_accounts"] == 0
    assert details["admin_ready"] is False


def test_schema_migrations_are_versioned_and_reject_newer_databases(tmp_path):
    from sqlalchemy import create_engine, text
    from schema_migrations import current_schema_version, run_schema_migrations  # noqa: E402

    engine = create_engine(f"sqlite:///{tmp_path / 'schema.db'}")
    applied = []

    def migration_one(connection):
        connection.execute(text("CREATE TABLE example (id INTEGER PRIMARY KEY)"))
        applied.append(1)

    def migration_two(connection):
        connection.execute(text("ALTER TABLE example ADD COLUMN name VARCHAR(64)"))
        applied.append(2)

    with engine.begin() as connection:
        assert run_schema_migrations(
            connection,
            "example",
            [(1, migration_one), (2, migration_two)],
        ) == 2
        assert current_schema_version(connection, "example") == 2
    with engine.begin() as connection:
        assert run_schema_migrations(
            connection,
            "example",
            [(1, migration_one), (2, migration_two)],
        ) == 2
    assert applied == [1, 2]

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE app_schema_versions SET version = 3 WHERE component = 'example'")
        )
    with engine.begin() as connection:
        with pytest.raises(RuntimeError, match="newer than this service supports"):
            run_schema_migrations(connection, "example", [(1, migration_one), (2, migration_two)])


def test_liveness_endpoint_does_not_depend_on_readiness_components():
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/health/live")

    assert response.status_code == 200
    assert response.get_json()["status"] == "alive"


@pytest.mark.parametrize(
    "path",
    [
        "/api/auth/me",
        "/api/models/mine",
        "/api/submissions/mine",
        "/api/submissions/example/export.jsonl",
        "/api/admin/submissions",
    ],
)
def test_private_api_responses_are_not_cacheable(path):
    web_app_module = importlib.import_module("web.app")
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get(path)

    assert response.headers["Cache-Control"] == "no-store, max-age=0"
    assert response.headers["Pragma"] == "no-cache"


def test_readiness_reuses_short_production_cache(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    calls = {"database": 0}

    def database_health():
        calls["database"] += 1
        return "healthy", {}

    monkeypatch.setattr(web_app_module, "HEALTH_CACHE_SECONDS", 15)
    monkeypatch.setattr(web_app_module, "_health_cache", {
        "expires_at": 0.0,
        "payload": None,
        "status_code": None,
    })
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", False)
    monkeypatch.setattr(web_app_module, "REQUIRE_OFFICIAL_SPATIAL", False)
    monkeypatch.setattr(web_app_module, "_database_storage_health", database_health)
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "visual_cognition_leaderboard",
        lambda limit=1: [],
    )
    monkeypatch.setattr(
        web_app_module,
        "latest_visible_scored_submission_fingerprints",
        lambda: {},
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "public_submission_fingerprints",
        lambda: {},
    )
    monkeypatch.setattr(web_app_module, "_ground_truth_bundle_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_spatial_bundle_health", lambda: ("unhealthy", {}))
    monkeypatch.setattr(web_app_module, "_email_delivery_health", lambda: ("dev", {}))
    monkeypatch.setattr(web_app_module, "_auth_service_health", lambda: ("disabled", {}))
    monkeypatch.setattr(web_app_module, "_deployment_configuration_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_backup_health", lambda: ("disabled", {}))
    monkeypatch.setitem(web_app_module.app.config, "TESTING", False)

    with web_app_module.app.test_client() as client:
        first = client.get("/api/readiness")
        second = client.get("/api/readiness")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json()["timestamp"] == second.get_json()["timestamp"]
    assert calls["database"] == 1


def test_readiness_does_not_fail_for_an_optional_spatial_bundle(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "REQUIRE_OFFICIAL_SPATIAL", False)
    monkeypatch.setattr(
        web_app_module,
        "_database_storage_health",
        lambda: ("healthy", {}),
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "visual_cognition_leaderboard",
        lambda limit=1: [],
    )
    monkeypatch.setattr(
        web_app_module,
        "latest_visible_scored_submission_fingerprints",
        lambda: {},
    )
    monkeypatch.setattr(
        web_app_module.leaderboard_store,
        "public_submission_fingerprints",
        lambda: {},
    )
    monkeypatch.setattr(
        web_app_module,
        "_spatial_bundle_health",
        lambda: ("unhealthy", {"required": False, "production_ready": False}),
    )
    monkeypatch.setattr(web_app_module, "_email_delivery_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_auth_service_health", lambda: ("healthy", {}))
    monkeypatch.setattr(
        web_app_module,
        "_deployment_configuration_health",
        lambda: ("healthy", {}),
    )
    monkeypatch.setattr(web_app_module, "_backup_health", lambda: ("healthy", {}))
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert payload["components"]["spatial_bundle"] == "unhealthy"


def test_readiness_detects_database_and_public_cache_divergence(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", False)
    monkeypatch.setattr(web_app_module, "REQUIRE_OFFICIAL_SPATIAL", False)
    monkeypatch.setattr(web_app_module, "_database_storage_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module.leaderboard_store, "visual_cognition_leaderboard", lambda limit=1: [])
    monkeypatch.setattr(web_app_module, "latest_visible_scored_submission_fingerprints", lambda: {"db-submission": "current"})
    monkeypatch.setattr(web_app_module.leaderboard_store, "public_submission_fingerprints", lambda: {})
    monkeypatch.setattr(web_app_module, "_spatial_bundle_health", lambda: ("unhealthy", {"required": False}))
    monkeypatch.setattr(web_app_module, "_email_delivery_health", lambda: ("dev", {}))
    monkeypatch.setattr(web_app_module, "_auth_service_health", lambda: ("disabled", {}))
    monkeypatch.setattr(web_app_module, "_deployment_configuration_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_backup_health", lambda: ("disabled", {}))
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["components"]["leaderboard_store"] == "unhealthy"
    assert payload["details"]["leaderboard_store"]["missing_submission_count"] == 1


def test_readiness_detects_stale_score_for_same_submission_id(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", False)
    monkeypatch.setattr(web_app_module, "REQUIRE_OFFICIAL_SPATIAL", False)
    monkeypatch.setattr(web_app_module, "_database_storage_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module.leaderboard_store, "visual_cognition_leaderboard", lambda limit=1: [])
    monkeypatch.setattr(web_app_module, "latest_visible_scored_submission_fingerprints", lambda: {"same-id": "new-score"})
    monkeypatch.setattr(web_app_module.leaderboard_store, "public_submission_fingerprints", lambda: {"same-id": "old-score"})
    monkeypatch.setattr(web_app_module, "_spatial_bundle_health", lambda: ("unhealthy", {"required": False}))
    monkeypatch.setattr(web_app_module, "_email_delivery_health", lambda: ("dev", {}))
    monkeypatch.setattr(web_app_module, "_auth_service_health", lambda: ("disabled", {}))
    monkeypatch.setattr(web_app_module, "_deployment_configuration_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_backup_health", lambda: ("disabled", {}))
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    details = response.get_json()["details"]["leaderboard_store"]
    assert details["missing_submission_count"] == 0
    assert details["stale_submission_count"] == 0
    assert details["score_mismatch_count"] == 1


def test_public_readiness_requires_auth_and_backups_even_without_flask_env(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(web_app_module, "REQUIRE_OFFICIAL_SPATIAL", False)
    monkeypatch.setattr(web_app_module, "_database_storage_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module.leaderboard_store, "visual_cognition_leaderboard", lambda limit=1: [])
    monkeypatch.setattr(web_app_module, "latest_visible_scored_submission_fingerprints", lambda: {})
    monkeypatch.setattr(web_app_module.leaderboard_store, "public_submission_fingerprints", lambda: {})
    monkeypatch.setattr(web_app_module, "_spatial_bundle_health", lambda: ("unhealthy", {"required": False}))
    monkeypatch.setattr(web_app_module, "_email_delivery_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_auth_service_health", lambda: ("disabled", {}))
    monkeypatch.setattr(web_app_module, "_deployment_configuration_health", lambda: ("healthy", {}))
    monkeypatch.setattr(web_app_module, "_backup_health", lambda: ("disabled", {}))
    monkeypatch.delenv("FLASK_ENV", raising=False)
    web_app_module.app.config["TESTING"] = True

    with web_app_module.app.test_client() as client:
        response = client.get("/api/health")

    assert response.status_code == 503
    assert response.get_json()["status"] == "degraded"


def test_public_deployment_health_fails_closed_for_local_http(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "DEPLOYMENT_MODE", "public")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(web_app_module, "CORS_ORIGINS", ["http://localhost:5173"])
    monkeypatch.setenv("FRONTEND_BASE_URL", "http://localhost:5173")
    monkeypatch.setenv("API_BASE_URL", "http://localhost:5050")
    monkeypatch.setenv("OAUTH_REDIRECT_BASE_URL", "http://localhost:5050")
    monkeypatch.setitem(web_app_module.app.config, "SESSION_COOKIE_SECURE", False)

    status, details = web_app_module._deployment_configuration_health()

    assert status == "unhealthy"
    assert details["mode"] == "public"
    assert details["detected_mode"] == "local"
    assert details["public_deployment_ready"] is False


def test_local_deployment_health_accepts_explicit_local_configuration(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    monkeypatch.setattr(web_app_module, "DEPLOYMENT_MODE", "local")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", False)
    monkeypatch.setattr(web_app_module, "CORS_ORIGINS", ["http://localhost:5173"])
    monkeypatch.setenv("FRONTEND_BASE_URL", "http://localhost:5173")
    monkeypatch.setenv("API_BASE_URL", "http://localhost:5050")
    monkeypatch.setenv("OAUTH_REDIRECT_BASE_URL", "http://localhost:5050")
    monkeypatch.setitem(web_app_module.app.config, "SESSION_COOKIE_SECURE", False)

    status, details = web_app_module._deployment_configuration_health()

    assert status == "healthy"
    assert details["mode"] == "local"
    assert details["mode_matches_configuration"] is True


def test_cross_site_frontend_requires_samesite_none(monkeypatch):
    web_app_module = importlib.import_module("web.app")
    frontend = "https://org-leaderboard.hf.space"
    api = "https://ms-vista-api.azurewebsites.net"
    monkeypatch.setattr(web_app_module, "DEPLOYMENT_MODE", "public")
    monkeypatch.setattr(web_app_module, "PUBLIC_DEPLOYMENT", True)
    monkeypatch.setattr(web_app_module, "CORS_ORIGINS", [frontend])
    monkeypatch.setenv("FRONTEND_BASE_URL", frontend)
    monkeypatch.setenv("API_BASE_URL", api)
    monkeypatch.setenv("OAUTH_REDIRECT_BASE_URL", api)
    monkeypatch.setenv("PRIVACY_POLICY_URL", f"{frontend}/privacy")
    monkeypatch.setitem(web_app_module.app.config, "SESSION_COOKIE_SECURE", True)
    monkeypatch.setitem(web_app_module.app.config, "SESSION_COOKIE_SAMESITE", "Lax")

    status, details = web_app_module._deployment_configuration_health()

    assert status == "unhealthy"
    assert details["frontend_api_same_site"] is False
    assert details["credential_cookie_ready"] is False

    monkeypatch.setitem(web_app_module.app.config, "SESSION_COOKIE_SAMESITE", "None")
    status, details = web_app_module._deployment_configuration_health()

    assert status == "healthy"
    assert details["credential_cookie_ready"] is True
