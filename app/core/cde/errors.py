"""CDE connector errors. Fail closed — never pretend a local register is Aconex."""

from __future__ import annotations


class CdeError(Exception):
    """A CDE operation failed. The CDE remains the system of record."""


class CdePayloadError(CdeError):
    """Caller sent a CDE-shaped row that cannot be accepted (no live id)."""


class CdeNotConfiguredError(CdeError):
    """Real CDE client is not configured. Do not fall back to a Fork register."""

    def __init__(
        self,
        message: str = (
            "Aconex / CDE client is not configured. Set ACONEX_ENABLED=true and "
            "provide Oracle Lobby OAuth credentials (ACONEX_CLIENT_ID + "
            "ACONEX_CLIENT_SECRET, or ACONEX_ACCESS_TOKEN). The Fork does not "
            "keep an RFI / submittal / transmittal register of its own."
        ),
    ) -> None:
        super().__init__(message)
