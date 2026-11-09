"""In-app support: a public form that emails SUPPORT_EMAIL via Resend.

The submit route is public (a signed-out visitor must be able to reach
support) and relies on the app-wide per-caller rate limiter (middleware)
for abuse control. The message is sent to the support address read from
SUPPORT_EMAIL (default support@theshovel.ai); the physical mailbox is an
owner action, this code only wires the sender.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import email as email_svc

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_SUPPORT_EMAIL = "support@theshovel.ai"


def support_address() -> str:
    """The support recipient. Owner-set via SUPPORT_EMAIL; sensible default."""
    return os.getenv("SUPPORT_EMAIL", "").strip() or DEFAULT_SUPPORT_EMAIL


class SupportRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="The help request body")
    contact_email: Optional[str] = Field(
        None, max_length=320, description="Where to reach the visitor back (optional)"
    )


@router.post("/v1/support")
async def submit_support(req: SupportRequest) -> dict:
    """Email the support request to SUPPORT_EMAIL.

    Public by design. Returns 503 when email is not configured or the
    provider refused, so a visitor is never told "sent" when nothing was.
    """
    to = support_address()
    subject = "Support request" + (f" from {req.contact_email}" if req.contact_email else "")
    html = (
        "<p>New support request.</p>"
        f"<p><strong>Contact:</strong> {req.contact_email or '(none given)'}</p>"
        f"<pre>{req.message}</pre>"
    )
    try:
        message_id = await email_svc.send_email(to, subject, html)
    except email_svc.EmailNotConfigured as exc:
        logger.warning("support email not sent: %s", exc)
        raise HTTPException(status_code=503, detail="Support email is not configured") from exc
    except email_svc.EmailSendFailed as exc:
        logger.warning("support email send failed: %s", exc)
        raise HTTPException(status_code=503, detail="Support email could not be sent") from exc
    return {"ok": True, "message_id": message_id}


def support_domain() -> str:
    """The domain part of the support address (for the MX startup check)."""
    addr = support_address()
    return addr.rsplit("@", 1)[-1].strip().lower()


def support_domain_has_mx(domain: str, timeout: float = 3.0) -> bool:
    """True when the domain advertises at least one MX record.

    Uses dnspython (lazy import). A lookup failure is treated as "no MX" so
    the startup check errs toward warning the operator rather than silently
    assuming the mailbox will work.
    """
    try:
        import dns.resolver  # type: ignore
    except ImportError:
        logger.warning("dnspython not installed; cannot check MX for %s", domain)
        return False
    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=timeout)
        return len(answers) > 0
    except Exception as exc:  # noqa: BLE001 — any DNS failure means "not verified"
        logger.warning("MX lookup for %s failed: %s", domain, exc)
        return False


def log_support_mx() -> None:
    """Startup check: warn (never fail) when the support domain has no MX."""
    domain = support_domain()
    if not support_domain_has_mx(domain):
        logger.warning(
            "SUPPORT_EMAIL domain %r has no MX record; support mail will not be delivered",
            domain,
        )
