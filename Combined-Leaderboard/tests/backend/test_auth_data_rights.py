"""Data subject rights: export and erasure-by-anonymisation on /api/auth/me."""
import json

import pytest
from sqlalchemy import text

import auth_db

from test_auth_routes import auth_app, PASSWORD, _register, _verify  # noqa: F401


def _signed_in(web):
    client = web.app.test_client()
    payload = _register(client)
    _verify(client, payload["verify_url"])
    client.post("/api/auth/login",
                json={"email": "route-user@example.com", "password": PASSWORD})
    return client


def test_export_returns_account_data_and_no_secrets(auth_app):
    client = _signed_in(auth_app)
    response = client.get("/api/auth/me/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    body = response.get_json()
    assert body["account"]["email"] == "route-user@example.com"
    assert "password_hash" not in json.dumps(body)


def test_export_requires_authentication(auth_app):
    response = auth_app.app.test_client().get("/api/auth/me/export")
    assert response.status_code == 401


def test_delete_anonymises_and_preserves_published_rows(auth_app):
    client = _signed_in(auth_app)
    with auth_db._Session() as session:
        session.execute(text(
            "CREATE TABLE IF NOT EXISTS registered_models "
            "(id INTEGER PRIMARY KEY, owner_email VARCHAR(255), name TEXT)"))
        session.execute(text(
            "INSERT INTO registered_models (owner_email, name) "
            "VALUES ('route-user@example.com', 'm1')"))
        session.commit()

    response = client.delete("/api/auth/me")
    assert response.status_code == 200
    assert response.get_json()["anonymized"] is True

    # the identity is gone everywhere it was stored...
    assert auth_db.get_user("route-user@example.com") is None
    with auth_db._Session() as session:
        remaining = session.execute(text(
            "SELECT COUNT(*) FROM registered_models "
            "WHERE lower(owner_email) = 'route-user@example.com'")).scalar()
        kept = session.execute(text("SELECT COUNT(*) FROM registered_models")).scalar()
    assert remaining == 0, "original email still present after erasure"
    assert kept == 1, "published result was destroyed instead of anonymised"

    # ...and the session no longer authenticates
    assert client.get("/api/auth/me").get_json()["authenticated"] is False


def test_delete_requires_authentication(auth_app):
    response = auth_app.app.test_client().delete("/api/auth/me")
    assert response.status_code == 401
