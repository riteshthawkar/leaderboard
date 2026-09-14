"""Data subject rights: export and erasure-by-anonymisation on /api/auth/me."""
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import auth_db
import submission_store

from test_auth_routes import auth_app, PASSWORD, _register, _verify  # noqa: F401


@pytest.fixture()
def data_rights_app(auth_app, monkeypatch):
    """Keep submission records in a database separate from authentication."""
    engine = create_engine("sqlite:///:memory:")
    submission_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(submission_store, "_engine", engine)
    monkeypatch.setattr(submission_store, "_Session", submission_session)
    monkeypatch.setattr(submission_store, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(submission_store, "_DB_PATH", None)
    submission_store.Base.metadata.create_all(engine)
    return auth_app


def _signed_in(web):
    client = web.app.test_client()
    payload = _register(client)
    _verify(client, payload["dev_verify_url"])
    client.post("/api/auth/login",
                json={"email": "route-user@example.com", "password": PASSWORD})
    return client


def _add_submission_records():
    with submission_store._Session() as session:
        session.add(submission_store.RegisteredModel(
            id="mdl_privacy_test",
            owner_email="route-user@example.com",
            display_name="Privacy Test Model",
            normalized_name="privacy test model",
            organization="MS VISTA",
            access="closed",
            base_model="",
            training_data="",
        ))
        session.add(submission_store.Submission(
            user_email="route-user@example.com",
            task_id="do_you_see_me",
            model_name="Privacy Test Model",
            model_id="mdl_privacy_test",
            status="scored",
            request_id="privacy-request",
            ip="203.0.113.10",
            score_submission_id="privacy-submission",
            file_sha256="a" * 64,
            row_count=1,
            latest_score_json='{"accuracy": 1.0, "total_samples": 1}',
        ))
        session.commit()


def test_export_returns_account_data_and_no_secrets(data_rights_app):
    client = _signed_in(data_rights_app)
    _add_submission_records()
    assert auth_db.issue_refresh_token("route-user@example.com", 7)
    assert auth_db.issue_oauth_exchange_code("route-user@example.com", 60)

    response = client.get("/api/auth/me/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    body = response.get_json()
    assert body["account"]["email"] == "route-user@example.com"
    assert len(body["records"]["submissions"]) == 1
    assert len(body["records"]["registered_models"]) == 1
    assert len(body["records"]["auth_refresh_tokens"]) == 1
    assert len(body["records"]["auth_oauth_exchange_codes"]) == 1
    encoded = json.dumps(body)
    for secret_field in (
        "password_hash",
        "verification_token",
        "password_reset_token",
        "token_digest",
        "code_digest",
        "replaced_by_digest",
    ):
        assert secret_field not in encoded


def test_export_requires_authentication(auth_app):
    response = auth_app.app.test_client().get("/api/auth/me/export")
    assert response.status_code == 401


def test_delete_anonymises_and_preserves_published_rows(data_rights_app):
    client = _signed_in(data_rights_app)
    _add_submission_records()
    assert auth_db.issue_refresh_token("route-user@example.com", 7)
    assert auth_db.issue_oauth_exchange_code("route-user@example.com", 60)

    csrf_token = client.get("/api/auth/me").get_json()["csrf_token"]
    response = client.delete(
        "/api/auth/me",
        headers={data_rights_app.CSRF_HEADER: csrf_token},
    )
    assert response.status_code == 200
    assert response.get_json()["anonymized"] is True

    # the identity is gone everywhere it was stored...
    assert auth_db.get_user("route-user@example.com") is None
    with auth_db._Session() as session:
        assert session.query(auth_db.RefreshToken).filter_by(
            user_email="route-user@example.com"
        ).count() == 0
        assert session.query(auth_db.OAuthExchangeCode).filter_by(
            user_email="route-user@example.com"
        ).count() == 0
    with submission_store._Session() as session:
        submissions = session.query(submission_store.Submission).all()
        models = session.query(submission_store.RegisteredModel).all()
    assert len(submissions) == 1, "published submission was destroyed"
    assert len(models) == 1, "registered model was destroyed"
    assert submissions[0].user_email.endswith("@anonymized.invalid")
    assert models[0].owner_email == submissions[0].user_email

    # ...and the session no longer authenticates
    assert client.get("/api/auth/me").get_json()["authenticated"] is False


def test_delete_restores_submission_ownership_when_auth_update_fails(
    data_rights_app,
    monkeypatch,
):
    client = _signed_in(data_rights_app)
    _add_submission_records()
    monkeypatch.setattr(
        data_rights_app,
        "anonymize_user",
        lambda _email, _replacement: (_ for _ in ()).throw(RuntimeError("auth unavailable")),
    )

    csrf_token = client.get("/api/auth/me").get_json()["csrf_token"]
    response = client.delete(
        "/api/auth/me",
        headers={data_rights_app.CSRF_HEADER: csrf_token},
    )

    assert response.status_code == 503
    with submission_store._Session() as session:
        assert session.query(submission_store.Submission).one().user_email == "route-user@example.com"
        assert session.query(submission_store.RegisteredModel).one().owner_email == "route-user@example.com"


def test_delete_requires_authentication(auth_app):
    response = auth_app.app.test_client().delete("/api/auth/me")
    assert response.status_code == 401
