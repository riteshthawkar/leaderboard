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


RECIPIENT = "user@example.com"


def _fake_acs(monkeypatch, begin_send):
    from azure.communication.email import EmailClient

    monkeypatch.setenv("ACS_CONNECTION_STRING", "endpoint=https://example.invalid/;accesskey=test")
    monkeypatch.setenv("ACS_SENDER_ADDRESS", "sender@example.com")

    class FakeClient:
        def begin_send(self, message):
            return begin_send(message)

    monkeypatch.setattr(EmailClient, "from_connection_string", lambda _cs: FakeClient())


def test_unconfigured_email_does_not_log_the_recipient(monkeypatch, caplog):
    _clear_email_provider(monkeypatch)
    monkeypatch.setenv("AUTH_DEV_MODE", "false")
    with caplog.at_level(logging.DEBUG):
        assert emailer.send_email(RECIPIENT, "Subject", "Body") == "error"
    assert RECIPIENT not in caplog.text


def test_development_email_logs_the_link_but_not_the_recipient(monkeypatch, caplog):
    _clear_email_provider(monkeypatch)
    monkeypatch.setenv("AUTH_DEV_MODE", "true")
    with caplog.at_level(logging.DEBUG):
        emailer.send_password_reset_email(
            RECIPIENT, "http://localhost:5173/login#reset_token=development-token"
        )
    assert "development-token" in caplog.text
    assert RECIPIENT not in caplog.text


def test_acs_delivery_logs_do_not_contain_the_recipient(monkeypatch, caplog):
    class Poller:
        def result(self, timeout):
            return {"id": "operation-1", "status": "Succeeded"}

    _fake_acs(monkeypatch, lambda message: Poller())
    with caplog.at_level(logging.DEBUG):
        assert emailer.send_email(RECIPIENT, "Subject", "Body") == "sent"
    assert RECIPIENT not in caplog.text


def test_acs_failure_keeps_the_diagnostic_but_redacts_the_address(monkeypatch, caplog):
    def reject(message):
        raise RuntimeError(f"Recipient {RECIPIENT} was rejected by the provider")

    _fake_acs(monkeypatch, reject)
    with caplog.at_level(logging.DEBUG):
        assert emailer.send_email(RECIPIENT, "Subject", "Body") == "error"
    assert RECIPIENT not in caplog.text
    assert "was rejected by the provider" in caplog.text


def test_smtp_refusal_does_not_leak_the_recipient_through_the_error_text(monkeypatch, caplog):
    # smtplib's own error text names the refused recipient, so logging str(exc) would leak it.
    import smtplib

    _clear_email_provider(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, **kwargs):
            pass

        def login(self, *args):
            pass

        def send_message(self, msg):
            raise smtplib.SMTPRecipientsRefused({RECIPIENT: (550, b"mailbox unavailable")})

    monkeypatch.setattr(emailer.smtplib, "SMTP", FakeSMTP)
    assert RECIPIENT in str(smtplib.SMTPRecipientsRefused({RECIPIENT: (550, b"x")}))
    with caplog.at_level(logging.DEBUG):
        assert emailer.send_email(RECIPIENT, "Subject", "Body") == "error"
    assert RECIPIENT not in caplog.text
