"""Outbound transactional email via Resend.

Only used for account lifecycle mail (address verification). Deliberately a
thin HTTP client over ``httpx`` rather than the ``resend`` SDK: httpx is
already a dependency, and keeping the call at the HTTP boundary means tests
can mock the transport and still exercise every line of our own logic.

Configuration (both required, else sending is treated as UNCONFIGURED):
  RESEND_API_KEY  -- server-side API key
  EMAIL_FROM      -- verified sender, e.g. "The Shovel <noreply@theshovel.ai>"
                     The domain must be verified in Resend or the send 4xxs.

There is no silent fallback. A caller that cannot send must decide what to do
about it -- returning success for an email that was never sent is the kind of
confident zero this codebase refuses.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SEND_URL = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 20.0


class EmailNotConfigured(RuntimeError):
    """RESEND_API_KEY / EMAIL_FROM missing — nothing was sent."""


class EmailSendFailed(RuntimeError):
    """The provider rejected the message, or was unreachable."""


def _api_key() -> str:
    return os.getenv("RESEND_API_KEY", "").strip()


def from_address() -> str:
    return os.getenv("EMAIL_FROM", "").strip()


def is_configured() -> bool:
    """True when a real send could be attempted. Read at call time."""
    return bool(_api_key() and from_address())


async def send_email(to: str, subject: str, html: str) -> str:
    """Send one message. Returns the provider message id.

    Raises EmailNotConfigured if credentials are absent, EmailSendFailed if the
    provider refused or was unreachable. Never returns normally on failure.
    """
    key = _api_key()
    sender = from_address()
    if not key or not sender:
        raise EmailNotConfigured(
            "RESEND_API_KEY and EMAIL_FROM must both be set to send email"
        )

    payload: dict[str, Any] = {
        "from": sender,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                _SEND_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise EmailSendFailed(f"could not reach the email provider: {exc}") from exc

    if resp.status_code >= 400:
        # The body carries the provider's reason (unverified domain, bad
        # recipient). Truncated, and the API key is never echoed.
        raise EmailSendFailed(
            f"email provider returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return str(resp.json().get("id") or "")
    except ValueError as exc:
        raise EmailSendFailed(
            "email provider returned a non-JSON success body"
        ) from exc


def password_reset_html(reset_url: str) -> str:
    """The body of the password-reset message: one link, shown raw as well."""
    return (
        "<p>Someone asked to reset the password for this account.</p>"
        f'<p><a href="{reset_url}">Choose a new password</a></p>'
        "<p>If the link does not work, paste this into your browser:<br>"
        f"{reset_url}</p>"
        "<p>This link expires in 1 hour and works once. If you did not ask "
        "for a reset, ignore this message: your password stays as it is.</p>"
    )


def verification_html(verify_url: str) -> str:
    """The body of the address-confirmation message.

    Plain and link-forward on purpose: the one action must survive clients
    that strip styling, and the raw URL is shown so a recipient can see where
    the link goes before following it.
    """
    return (
        "<p>Confirm your email address to finish setting up your account.</p>"
        f'<p><a href="{verify_url}">Confirm my email address</a></p>'
        "<p>If the link does not work, paste this into your browser:<br>"
        f"{verify_url}</p>"
        "<p>This link expires in 24 hours. If you did not create an account, "
        "ignore this message and nothing will happen.</p>"
    )
