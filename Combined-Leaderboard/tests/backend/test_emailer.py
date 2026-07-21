import logging

import emailer  # noqa: E402


def _clear_email_provider(monkeypatch):
    for name in (
        "ACS_CONNECTION_STRING",
        "ACS_ENDPOINT",
        "ACS_SENDER_ADDRESS",
        "SMTP_HOST",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_FROM",
    ):
        monkeypatch.delenv(name, raising=False)


def test_unconfigured_production_email_does_not_log_account_token(monkeypatch, caplog):
    _clear_email_provider(monkeypatch)
    monkeypatch.setenv("AUTH_DEV_MODE", "false")
    secret_link = "https://app.example/reset#reset_token=secret-token"

    with caplog.at_level(logging.WARNING):
        result = emailer.send_password_reset_email("user@example.com", secret_link)

    assert result == "error"
    assert "secret-token" not in caplog.text
    assert "message body was omitted" in caplog.text


def test_development_email_logging_is_explicit(monkeypatch, caplog):
    _clear_email_provider(monkeypatch)
    monkeypatch.setenv("AUTH_DEV_MODE", "true")

    with caplog.at_level(logging.WARNING):
        result = emailer.send_password_reset_email(
            "user@example.com",
            "http://localhost:5173/login#reset_token=development-token",
        )

    assert result == "logged"
    assert "development-token" in caplog.text


def test_acs_delivery_waits_for_terminal_success(monkeypatch):
    from azure.communication.email import EmailClient

    monkeypatch.setenv("ACS_CONNECTION_STRING", "endpoint=https://example.invalid/;accesskey=test")
    monkeypatch.setenv("ACS_SENDER_ADDRESS", "sender@example.com")

    class FakePoller:
        def result(self, timeout):
            assert timeout == 30
            return {"id": "operation-1", "status": "Succeeded"}

    class FakeClient:
        def begin_send(self, message):
            assert message["recipients"]["to"][0]["address"] == "user@example.com"
            return FakePoller()

    monkeypatch.setattr(
        EmailClient,
        "from_connection_string",
        lambda _connection_string: FakeClient(),
    )

    assert emailer.send_email("user@example.com", "Subject", "Body") == "sent"


def test_acs_non_success_status_is_reported_as_failure(monkeypatch):
    from azure.communication.email import EmailClient

    monkeypatch.setenv("ACS_CONNECTION_STRING", "endpoint=https://example.invalid/;accesskey=test")
    monkeypatch.setenv("ACS_SENDER_ADDRESS", "sender@example.com")

    class FakePoller:
        def result(self, timeout):
            assert timeout == 30
            return {"id": "operation-1", "status": "Failed"}

    class FakeClient:
        def begin_send(self, _message):
            return FakePoller()

    monkeypatch.setattr(
        EmailClient,
        "from_connection_string",
        lambda _connection_string: FakeClient(),
    )

    assert emailer.send_email("user@example.com", "Subject", "Body") == "error"
