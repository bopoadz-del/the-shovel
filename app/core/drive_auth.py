"""Per-user Google Drive OAuth token store + refresh.

Each connected user gets their own token file
(DATA_DIR/google_drive_token_<user>.json). The token — including the refresh
token, a secret — is persisted via file_crypto, so it is encrypted at rest
when DATA_ENCRYPTION_KEY is set. Tokens are NOT shared between users.
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from app.core import file_crypto

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"


class DriveNotConnected(Exception):
    """No Google Drive token is stored for this user."""


class DriveAuthError(Exception):
    """Token exchange or refresh failed."""


def _safe_user(user_id: str) -> str:
    """Filesystem-safe form of a user id for use in the token filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id or "")).strip("_")
    if not cleaned:
        raise DriveAuthError("A user id is required for the Drive token store.")
    return cleaned


def _token_path(user_id: str) -> Path:
    return (
        Path(os.getenv("DATA_DIR", "./data"))
        / f"google_drive_token_{_safe_user(user_id)}.json"
    )


def save_token(user_id: str, token: Dict[str, Any]) -> None:
    path = _token_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_crypto.write_document(str(path), json.dumps(token).encode("utf-8"))


def load_token(user_id: str) -> Optional[Dict[str, Any]]:
    path = _token_path(user_id)
    if not path.exists():
        return None
    try:
        return json.loads(file_crypto.read_document(str(path)).decode("utf-8"))
    except (file_crypto.DecryptionError, ValueError, UnicodeDecodeError) as exc:
        # The token was written with a different DATA_ENCRYPTION_KEY (rotated),
        # or is corrupt, and can no longer be read. Treat the user as not
        # connected and remove the dead file so the UI prompts a fresh Drive
        # connect instead of surfacing a 500 on every admin/drive call.
        logger.warning(
            "Drive token for %s is unreadable (%s); clearing it so the user can reconnect.",
            _safe_user(user_id), type(exc).__name__,
        )
        try:
            path.unlink()
        except OSError:
            logger.warning(
                "swallowed %s in load_token() — continuing",
                "OSError", exc_info=True,
            )
        return None


def clear_token(user_id: str) -> bool:
    path = _token_path(user_id)
    if path.exists():
        path.unlink()
        return True
    return False


_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


async def _revoke_request(token: str) -> bool:
    """POST the token to Google's revoke endpoint. Overridable seam for tests."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(_REVOKE_URL, data={"token": token})
    return resp.status_code == 200


async def revoke_and_clear(user_id: str) -> Dict[str, bool]:
    """Disconnect for real: revoke the grant at Google, then drop the file.

    Deleting only our copy left the grant live at Google -- the app still
    showed as connected in the user's Google account, and the refresh token
    kept working for anyone holding a copy. Revoking the refresh token ends
    the whole grant. A failed revoke still removes our copy, and says so."""
    token = load_token(user_id)
    revoked = False
    secret = (token or {}).get("refresh_token") or (token or {}).get("access_token")
    if secret:
        try:
            revoked = await _revoke_request(secret)
        except httpx.HTTPError:
            logger.warning("Drive revoke failed for %s", _safe_user(user_id), exc_info=True)
    return {"was_connected": clear_token(user_id), "revoked": revoked}


async def _refresh_request(refresh_token: str) -> Dict[str, Any]:
    """POST the refresh grant to Google. Overridable seam for tests."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(_TOKEN_URL, data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        })
    if resp.status_code != 200:
        # Surface Google's actual reason — usually one of:
        #   invalid_grant + "Token has been expired or revoked"  → testing-mode
        #       7-day expiry, user revoked, password change, or 6mo unused
        #   invalid_client                                       → wrong creds
        # Helps the operator distinguish "publish OAuth consent screen" from
        # "rotate credentials" without log-diving.
        google_reason = ""
        try:
            body = resp.json()
            google_reason = (
                body.get("error_description")
                or body.get("error")
                or ""
            )
        except Exception:
            google_reason = (resp.text or "")[:200]
        raise DriveAuthError(
            f"Token refresh failed (HTTP {resp.status_code}: {google_reason})"
            if google_reason
            else f"Token refresh failed (HTTP {resp.status_code})"
        )
    return resp.json()


async def get_access_token(user_id: str) -> str:
    """Return a valid access token for ``user_id``, refreshing if expired."""
    token = load_token(user_id)
    if not token:
        raise DriveNotConnected("Google Drive is not connected.")
    if token.get("expiry", 0) > time.time() + 60:
        return token["access_token"]
    refresh_token = token.get("refresh_token")
    if not refresh_token:
        raise DriveAuthError("No refresh token stored — reconnect Google Drive.")
    data = await _refresh_request(refresh_token)
    token["access_token"] = data["access_token"]
    token["expiry"] = time.time() + int(data.get("expires_in", 3600))
    save_token(user_id, token)
    return token["access_token"]
