"""JWT session tokens — Stream A (User Accounts & Multi-Tenancy).

Pure-Python (PyJWT). SECRET_KEY comes from the env var of the same name;
if unset, a key is generated and persisted to {DATA_DIR}/.secret_key so
tokens survive process restarts. DATA_DIR is read at call time so tests
can relocate it.
"""
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone

import jwt
from jwt import InvalidTokenError  # re-exported for callers

_ALGORITHM = "HS256"

# Session tokens predate the ``typ`` claim and carry none. Anything with a
# different ``typ`` is a single-purpose token and must never authenticate.
SESSION_TYP = "session"
EMAIL_VERIFY_TYP = "email_verify"
#: A password-reset link: single purpose, never a session.
PASSWORD_RESET_TYP = "password_reset"
_lock = threading.Lock()
_cached_secret: str | None = None

logger = logging.getLogger(__name__)


def _default_expiry() -> int:
    return int(os.getenv("JWT_EXPIRY_SECONDS", "86400"))


def _secret_file() -> str:
    data_dir = os.getenv("DATA_DIR", "./data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError:
        import tempfile
        data_dir = tempfile.gettempdir()
        logger.warning(
            "Could not create DATA_DIR %r; signing secret will be stored in "
            "temp dir %r — rotate the secret if this host is shared.",
            os.getenv("DATA_DIR", "./data"),
            data_dir,
        )
    return os.path.join(data_dir, ".secret_key")


def _get_secret() -> str:
    """Resolve the signing secret: env var, else persisted file, else generate."""
    global _cached_secret
    if _cached_secret is not None:
        return _cached_secret
    with _lock:
        if _cached_secret is not None:
            return _cached_secret
        env_secret = os.getenv("SECRET_KEY")
        if env_secret:
            _cached_secret = env_secret
            return _cached_secret
        path = _secret_file()
        disk_secret: str | None = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                disk_secret = f.read().strip() or None  # treat empty file as missing
        if disk_secret:
            _cached_secret = disk_secret
        else:
            _cached_secret = secrets.token_hex(32)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_cached_secret)
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    # No-op on platforms without POSIX permissions (Windows),
                    # so this is EXPECTED there and must not be a warning — it
                    # would fire on every boot. It still gets recorded, because
                    # on a POSIX host it means the signing secret is sitting on
                    # disk more readable than intended.
                    logger.debug(
                        "could not chmod 0600 the JWT secret at %s "
                        "(expected on non-POSIX platforms)", path, exc_info=True,
                    )
            except OSError:
                # The secret could not be PERSISTED. This process falls back to
                # an in-memory secret, which means every token it issues becomes
                # invalid on restart, and a second worker will not accept them.
                # Genuinely worth a warning.
                logger.warning(
                    "could not persist the JWT secret to %s — falling back to "
                    "an in-memory secret; tokens will not survive a restart or "
                    "be valid across workers", path, exc_info=True,
                )
        return _cached_secret


def create_token(user_id: str, expires_in: int | None = None) -> str:
    exp_seconds = _default_expiry() if expires_in is None else expires_in
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=exp_seconds),
    }
    return jwt.encode(payload, _get_secret(), algorithm=_ALGORITHM)


def create_purpose_token(user_id: str, purpose: str, expires_in: int) -> str:
    """Mint a SINGLE-PURPOSE token that can never authenticate a session.

    Email verification links travel through an insecure channel: they sit in
    mailboxes, get forwarded, and leak through referrers. A link that doubled
    as a session token would hand out logins to anyone who saw the URL. The
    ``typ`` claim is what keeps the two apart, and ``decode_token`` refuses
    anything carrying one.
    """
    if not purpose or purpose == SESSION_TYP:
        raise ValueError("purpose must be a non-session token type")
    payload = {
        "user_id": user_id,
        "typ": purpose,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
    }
    return jwt.encode(payload, _get_secret(), algorithm=_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode + verify a SESSION token. Raises InvalidTokenError on any failure.

    Every authenticating caller in the app funnels through here
    (``dependencies.require_user``, ``dependencies`` API-key path, and
    ``main._rate_limit_identity``), so rejecting purpose-scoped tokens at this
    one seam closes the door for all of them.

    Session tokens predate the ``typ`` claim and carry none, so a missing
    ``typ`` is treated as a session; anything else is refused.
    """
    payload = jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
    typ = payload.get("typ")
    if typ is not None and typ != SESSION_TYP:
        raise InvalidTokenError(
            f"token of type {typ!r} cannot be used to authenticate"
        )
    return payload


def decode_purpose_token(token: str, purpose: str) -> dict:
    """Decode a single-purpose token, refusing session tokens.

    The other direction of the same fence: a stolen session token must not be
    replayable as a verification link.
    """
    payload = jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
    if payload.get("typ") != purpose:
        raise InvalidTokenError(
            f"expected a {purpose!r} token, got {payload.get('typ')!r}"
        )
    return payload


def signing_secret() -> str:
    """The app's signing secret, for other HMAC uses (e.g. OAuth state).

    Same resolution order as JWTs: SECRET_KEY env var, else the persisted
    DATA_DIR/.secret_key file — so values signed with it survive restarts
    and are shared by every worker on the same host/disk."""
    return _get_secret()
