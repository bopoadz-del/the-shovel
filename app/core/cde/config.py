"""Feature flag + credential detection for the Aconex / CDE client.

Default OFF. Production stays on the project flag until Oracle credentials exist.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Oracle Construction and Engineering Lobby (commercial). Early Access is
# constructionandengineering-ea.oraclecloud.com — override via ACONEX_LOBBY_URL.
DEFAULT_LOBBY_URL = "https://constructionandengineering.oraclecloud.com"
# Single resource server for all commercial Aconex instances + EA1.
DEFAULT_API_BASE = "https://api.aconex.com"


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def aconex_enabled() -> bool:
    """Real Aconex HTTP client. Default off until credentials exist."""
    return _truthy("ACONEX_ENABLED")


def requested_adapter() -> str:
    """Explicit adapter override: '', 'fake', or 'aconex'."""
    return os.getenv("CDE_ADAPTER", "").strip().lower()


def lobby_url() -> str:
    return (os.getenv("ACONEX_LOBBY_URL") or DEFAULT_LOBBY_URL).rstrip("/")


def api_base() -> str:
    return (os.getenv("ACONEX_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def access_token() -> str:
    return os.getenv("ACONEX_ACCESS_TOKEN", "").strip()


def client_id() -> str:
    return os.getenv("ACONEX_CLIENT_ID", "").strip()


def client_secret() -> str:
    return os.getenv("ACONEX_CLIENT_SECRET", "").strip()


def aconex_user_id() -> str:
    return os.getenv("ACONEX_USER_ID", "").strip()


def aconex_user_site() -> str:
    return os.getenv("ACONEX_USER_SITE", "").strip()


def default_cde_project_id() -> str:
    return os.getenv("ACONEX_PROJECT_ID", "").strip()


def has_oauth_credentials() -> bool:
    """Enough to talk to Lobby / the resource server without inventing a register."""
    if access_token():
        return True
    return bool(client_id() and client_secret())


def oauth_client_ready() -> bool:
    """Feature flag ON and credentials present — the real client may be constructed."""
    return aconex_enabled() and has_oauth_credentials()


def event_poll_enabled() -> bool:
    """Optional CDE mail+register poll loop. Default off — fail closed."""
    return _truthy("CDE_EVENT_POLL_ENABLED")


def connector_mode(aconex_connected: bool) -> str:
    """Honest GET /connectors mode: flag vs oauth vs not_configured.

    ``aconex_connected`` remains "this project has a CDE feed" (real OAuth
    or explicitly managed outside). Mode says *how*.
    """
    if not aconex_connected:
        return "not_configured"
    if oauth_client_ready():
        return "oauth"
    return "flag"


def connector_note(mode: str) -> str:
    if mode == "oauth":
        return (
            "Oracle Aconex OAuth client is enabled. Aconex is the system of "
            "record; The Fork caches documents for RAG only."
        )
    if mode == "flag":
        return (
            "CDE feed marked connected without a live OAuth client "
            "(managed outside, or ACONEX_ENABLED is off). Full Oracle client "
            "needs Lobby credentials."
        )
    return (
        "No CDE feed on this project. POST the connector flag for an "
        "externally managed feed, or enable ACONEX_ENABLED with Oracle "
        "Lobby OAuth credentials."
    )
