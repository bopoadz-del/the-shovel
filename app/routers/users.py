"""User account API — register, login, me, email verification. Stream A."""
import logging
import os
import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.core import email as email_service
from app.core import jwt_auth
from app.core import users as users_store
from app.dependencies import require_user

logger = logging.getLogger(__name__)

router = APIRouter()

# A verification link sits in a mailbox for a while, gets forwarded, and is
# sometimes prefetched by the mail client. 24h is long enough to be usable and
# short enough that a leaked old mailbox is not a standing account takeover.
_VERIFY_TTL_SECONDS = 24 * 60 * 60

# Dummy credentials for constant-time login (prevents email enumeration via
# timing side-channel). Computed once at import so no PBKDF2 is run per request.
# The dummy is a random value drawn at startup: it is never written in the
# source and never matches any account's password.
_DUMMY_CREDS = users_store.hash_password(secrets.token_urlsafe(24))
_DUMMY_HASH = _DUMMY_CREDS["hash"]
_DUMMY_SALT = _DUMMY_CREDS["salt"]


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


def _registration_allowed() -> bool:
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
    if env in {"dev", "development", "local", "test", "testing"}:
        return True
    # Allow open registration if the flag is set
    if os.getenv("ALLOW_OPEN_REGISTRATION", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return False


def _public_base_url() -> str:
    """Base URL for links we put in email. Same precedence as exports.py:
    operator override, then Render's own value, then nothing."""
    return (
        os.getenv("PUBLIC_BASE_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or ""
    ).rstrip("/")


def _is_dev_env() -> bool:
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
    return env in {"dev", "development", "local", "test", "testing"}


async def _send_verification(user: dict) -> bool:
    """Mail a verification link. True if the provider accepted it.

    Never raises: the caller has already created the account, and a provider
    blip must not turn a working registration into a permanently unusable
    address (the email is unique-constrained, so a retry would 409). Failures
    are logged and recoverable through /v1/users/resend-verification.
    """
    base = _public_base_url()
    if not base:
        logger.warning(
            "cannot send verification email: no PUBLIC_BASE_URL/RENDER_EXTERNAL_URL, "
            "so the link would be relative and unusable"
        )
        return False
    token = jwt_auth.create_purpose_token(
        user["id"], jwt_auth.EMAIL_VERIFY_TYP, _VERIFY_TTL_SECONDS
    )
    url = f"{base}/v1/users/verify-email?token={token}"
    try:
        await email_service.send_email(
            to=user["email"],
            subject="Confirm your email address",
            html=email_service.verification_html(url),
        )
        return True
    except (email_service.EmailNotConfigured, email_service.EmailSendFailed):
        logger.warning(
            "verification email to %s was not sent", user["email"], exc_info=True
        )
        return False


@router.post("/v1/users/register", status_code=201)
async def register(req: RegisterRequest):
    if not _registration_allowed():
        raise HTTPException(403, "Open registration is disabled")

    # Fail CLOSED rather than create accounts nobody can ever verify. With
    # open registration on a live box and no mail configured, every signup
    # would be a dead end the user cannot recover from on their own.
    # Locally there is no provider, so dev/test auto-verifies instead.
    dev = _is_dev_env()
    if not email_service.is_configured() and not dev:
        raise HTTPException(
            503,
            "Registration is temporarily unavailable: email delivery is not "
            "configured, so the address could not be verified.",
        )

    auto_verified = dev and not email_service.is_configured()
    try:
        user = users_store.create_user(
            str(req.email), req.password, display_name=req.display_name,
            email_verified=auto_verified,
        )
    except ValueError:
        # An existing address gets the same reply as a new one: a
        # registration never reveals whether an account exists (#978).
        return _registration_received(str(req.email))
    if not user.get("email_verified"):
        await _send_verification(user)
    # One reply for every registration -- new address or existing, mail
    # configured or not, sent or not.
    return _registration_received(str(req.email))


#: What every registration answers -- new address or existing one, mail
#: configured or not, sent or not -- so the reply tells a caller nothing about
#: which accounts exist.
_PENDING_MESSAGE = ("If this address can be registered, a confirmation link is on its way. "
                    'If none arrives, use "Resend confirmation email" on the sign-in page.')


def _registration_received(email: str) -> dict:
    return {"email": email, "registration_received": True, "message": _PENDING_MESSAGE}


@router.get("/v1/users/verify-email")
async def verify_email(token: str = ""):
    """Follow-the-link endpoint. Always lands the user back in the app.

    Redirects rather than returning JSON: this URL is opened by a human from
    their mail client, and a raw JSON error page is a dead end. The outcome
    travels as a query flag the SPA renders.
    """
    frontend = os.getenv("FRONTEND_URL", "").rstrip("/") or _public_base_url()

    def _back(outcome: str) -> RedirectResponse:
        return RedirectResponse(f"{frontend}/?verified={outcome}", status_code=302)

    if not token:
        return _back("invalid")
    try:
        payload = jwt_auth.decode_purpose_token(token, jwt_auth.EMAIL_VERIFY_TYP)
    except jwt_auth.InvalidTokenError:
        # Covers expiry, tampering, and a session token replayed as a
        # verification link. The user is told to request a fresh one.
        return _back("invalid")

    if not users_store.mark_email_verified(payload.get("user_id") or ""):
        return _back("invalid")
    return _back("success")


class ResendVerificationRequest(BaseModel):
    email: str


@router.post("/v1/users/resend-verification", status_code=202)
async def resend_verification(req: ResendVerificationRequest):
    """Re-send the link. Always 202, regardless of whether the address exists.

    An honest 404 here would turn this endpoint into an account oracle, which
    is exactly what the constant-time login path above works to avoid.
    """
    user = users_store.get_user_by_email(str(req.email))
    if user and not user.get("email_verified"):
        await _send_verification(user)
    return {"status": "accepted"}


#: A reset link is short-lived: it sits in a mailbox and works once.
_RESET_TTL_SECONDS = 60 * 60
_RESET_INVALID = "This reset link is invalid or has expired. Ask for a new one from the sign-in page."


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


@router.post("/v1/users/forgot-password", status_code=202)
async def forgot_password(req: ForgotPasswordRequest):
    """Mail a reset link. Always 202 with the same body, whether or not the
    address has an account, so the endpoint is never an account oracle."""
    from app.core.offload import off_loop

    user = await off_loop(users_store.get_user_by_email, str(req.email))
    base = os.getenv("FRONTEND_URL", "").rstrip("/") or _public_base_url()
    if user and base:
        token = jwt_auth.create_purpose_token(user["id"], jwt_auth.PASSWORD_RESET_TYP, _RESET_TTL_SECONDS)
        # The link carries the current password's fingerprint: once the
        # password changes, the link no longer works (single use).
        url = f"{base}/?reset={token}.{users_store.password_fingerprint(user)}"
        try:
            await email_service.send_email(to=user["email"], subject="Reset your password",
                                           html=email_service.password_reset_html(url))
        except (email_service.EmailNotConfigured, email_service.EmailSendFailed):
            logger.warning("password reset email was not sent", exc_info=True)
    elif user and not base:
        logger.warning("cannot send password reset email: no FRONTEND_URL/PUBLIC_BASE_URL")
    return {"status": "accepted",
            "message": "If this address has an account, a reset link is on its way."}


@router.post("/v1/users/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Set a new password from a reset link. The link works once: it carries
    the old password's fingerprint, which no longer matches after the reset."""
    from app.core.offload import off_loop

    token, _, fingerprint = (req.token or "").rpartition(".")
    if not token or not fingerprint:
        raise HTTPException(400, _RESET_INVALID)
    try:
        payload = jwt_auth.decode_purpose_token(token, jwt_auth.PASSWORD_RESET_TYP)
    except jwt_auth.InvalidTokenError:
        raise HTTPException(400, _RESET_INVALID)
    user = await off_loop(users_store.get_user_by_id, str(payload.get("user_id") or ""))
    if not user or not secrets.compare_digest(users_store.password_fingerprint(user), fingerprint):
        raise HTTPException(400, _RESET_INVALID)
    # PBKDF2 is CPU work: off the event loop.
    await off_loop(users_store.set_password, user["id"], req.new_password)
    return {"status": "reset", "message": "Your password has been changed. Sign in with the new one."}


@router.post("/v1/users/login")
async def login(req: LoginRequest):
    user = users_store.get_user_by_email(str(req.email))
    if user is None:
        # Unknown email: run a dummy PBKDF2 to match the timing of a real
        # verify_password call — prevents email enumeration via response time.
        users_store.verify_password(req.password, _DUMMY_HASH, _DUMMY_SALT)
        raise HTTPException(401, "Invalid email or password")
    if not users_store.verify_password(
        req.password, user.get("password_hash"), user.get("salt")
    ):
        raise HTTPException(401, "Invalid email or password")
    # Deliberately AFTER the password check. Answering "unverified" before
    # proving the password would turn login into an account oracle, undoing
    # the constant-time work above. A caller who reaches here already knows
    # the credentials, so naming the real reason costs nothing and is the
    # only way they can act on it.
    if not user.get("email_verified"):
        raise HTTPException(
            403,
            "Email address not verified. Check your inbox for the confirmation "
            'link, or use "Resend confirmation email" on the sign-in page.',
        )
    return {
        "token": jwt_auth.create_token(user["id"]),
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "role": user["role"]},
    }


@router.get("/v1/users/me")
async def me(auth: dict = Depends(require_user)):
    user = users_store.get_user_by_id(auth["user_id"])
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "user_id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "role": user["role"],
    }
