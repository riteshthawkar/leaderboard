"""
Transactional email sending for account verification.

Uses SMTP when configured via environment variables. When SMTP is not
configured (typical in local development), the message is written to the log
instead of being delivered, so the verification flow remains testable without a
mail server.

Providers (checked in order):
  1. Azure Communication Services (ACS) Email — when ACS_CONNECTION_STRING or
     ACS_ENDPOINT is set. This is the Azure-native transactional email service.
  2. Generic SMTP — when SMTP_HOST is set.
  3. Development fallback — logs the message (no delivery).

Environment variables:
    # Azure Communication Services (preferred)
    ACS_CONNECTION_STRING  ACS connection string (endpoint=...;accesskey=...)
    ACS_ENDPOINT           ACS endpoint (https://<res>.communication.azure.com);
                           used with Managed Identity / DefaultAzureCredential
                           when ACS_CONNECTION_STRING is not set
    ACS_SENDER_ADDRESS     Verified sender, e.g. DoNotReply@<domain>.azurecomm.net
    # Generic SMTP (fallback)
    SMTP_HOST        SMTP server hostname (enables real delivery when set)
    SMTP_PORT        SMTP port (default 587)
    SMTP_USERNAME    SMTP auth username (optional)
    SMTP_PASSWORD    SMTP auth password (optional)
    SMTP_USE_TLS     "true"/"false" — STARTTLS (default true)
    SMTP_FROM        From address (default SMTP_USERNAME or no-reply@ms-vista.local)
"""

import logging
import os
import smtplib
import ssl
from html import escape
from email.message import EmailMessage
from typing import Optional

logger = logging.getLogger(__name__)


def _smtp_config():
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"},
        "from_addr": (os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USERNAME", "") or "no-reply@ms-vista.local").strip(),
        "timeout": float(os.getenv("EMAIL_SEND_TIMEOUT_SECONDS", "30")),
    }


def _acs_enabled() -> bool:
    return bool(os.getenv("ACS_CONNECTION_STRING", "").strip() or os.getenv("ACS_ENDPOINT", "").strip())


def _acs_send(to_addr: str, subject: str, text_body: str, html_body: Optional[str]) -> str:
    """Send via Azure Communication Services Email (connection string or Managed Identity)."""
    sender = os.getenv("ACS_SENDER_ADDRESS", "").strip()
    if not sender:
        logger.error("ACS_SENDER_ADDRESS is not set; cannot send email via Azure Communication Services.")
        return "error"
    try:
        from azure.communication.email import EmailClient
    except ImportError:
        logger.error("azure-communication-email is not installed. Run: pip install azure-communication-email")
        return "error"
    conn = os.getenv("ACS_CONNECTION_STRING", "").strip()
    endpoint = os.getenv("ACS_ENDPOINT", "").strip()
    try:
        if conn:
            client = EmailClient.from_connection_string(conn)
        else:
            from azure.identity import DefaultAzureCredential
            client = EmailClient(endpoint, DefaultAzureCredential())
        message = {
            "senderAddress": sender,
            "recipients": {"to": [{"address": to_addr}]},
            "content": {
                "subject": subject,
                "plainText": text_body,
                **({"html": html_body} if html_body else {}),
            },
        }
        poller = client.begin_send(message)
        result = poller.result(
            timeout=float(os.getenv("EMAIL_SEND_TIMEOUT_SECONDS", "30"))
        )
        status = str(result.get("status") if isinstance(result, dict) else "").lower()
        if status != "succeeded":
            logger.error(
                "ACS email to %s did not report terminal success (status=%s)",
                to_addr,
                status or "missing",
            )
            return "error"
        logger.info("ACS email delivered for %s", to_addr)
        return "sent"
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("ACS email to %s failed: %s", to_addr, exc)
        return "error"


def send_email(to_addr: str, subject: str, text_body: str, html_body: Optional[str] = None) -> str:
    """
    Send an email via the first configured provider (ACS, then SMTP, then dev log).

    Returns:
        "sent"   - delivered/queued via ACS or SMTP
        "logged" - no provider configured; body written to the log (dev fallback)
        "error"  - a provider was configured but delivery failed
    """
    # 1. Azure Communication Services (preferred, Azure-native).
    if _acs_enabled():
        return _acs_send(to_addr, subject, text_body, html_body)

    # 2. Generic SMTP.
    cfg = _smtp_config()
    if not cfg:
        if os.getenv("AUTH_DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.warning(
                "Development email for %s was not delivered.\nSubject: %s\n%s",
                to_addr,
                subject,
                text_body,
            )
            return "logged"
        logger.error(
            "No email provider is configured; email to %s was not delivered. "
            "The message body was omitted from logs.",
            to_addr,
        )
        return "error"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_addr
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"]) as server:
            if cfg["use_tls"]:
                server.starttls(context=ssl.create_default_context())
            if cfg["username"]:
                server.login(cfg["username"], cfg["password"])
            server.send_message(msg)
        logger.info("Email sent to %s (subject: %s)", to_addr, subject)
        return "sent"
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Failed to send email to %s: %s", to_addr, exc)
        return "error"


def send_verification_email(to_addr: str, verify_url: str) -> str:
    """Send the account verification link. Returns the send_email() status."""
    subject = "Verify your MS-VISTA account"
    text = (
        "Welcome to MS-VISTA.\n\n"
        "Confirm your email address to activate your account and submit models:\n\n"
        f"{verify_url}\n\n"
        "This link expires in 24 hours. If you did not create this account, you can "
        "safely ignore this email."
    )
    safe_verify_url = escape(verify_url, quote=True)
    html = (
        "<p>Welcome to <strong>MS-VISTA</strong>.</p>"
        "<p>Confirm your email address to activate your account and submit models:</p>"
        f'<p><a href="{safe_verify_url}">Verify my email</a></p>'
        f'<p style="color:#666;font-size:13px">Or paste this link into your browser:<br>{safe_verify_url}</p>'
        "<p style=\"color:#666;font-size:13px\">This link expires in 24 hours. "
        "If you did not create this account, you can ignore this email.</p>"
    )
    return send_email(to_addr, subject, text, html)


def send_password_reset_email(to_addr: str, reset_url: str) -> str:
    """Send the password reset link. Returns the send_email() status."""
    subject = "Reset your MS-VISTA password"
    text = (
        "We received a request to reset your MS-VISTA password.\n\n"
        "Use this link to choose a new password:\n\n"
        f"{reset_url}\n\n"
        "This link expires in 1 hour. If you did not request a password reset, "
        "you can safely ignore this email."
    )
    safe_reset_url = escape(reset_url, quote=True)
    html = (
        "<p>We received a request to reset your <strong>MS-VISTA</strong> password.</p>"
        f'<p><a href="{safe_reset_url}">Reset my password</a></p>'
        f'<p style="color:#666;font-size:13px">Or paste this link into your browser:<br>{safe_reset_url}</p>'
        "<p style=\"color:#666;font-size:13px\">This link expires in 1 hour. "
        "If you did not request a password reset, you can ignore this email.</p>"
    )
    return send_email(to_addr, subject, text, html)
