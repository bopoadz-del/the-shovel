"""SSRF guard — validate that an outbound URL points at a public host.

Blocks that fetch or POST to caller-supplied URLs (``web``, ``webhook``) use
this so a caller cannot make the server reach loopback, link-local (cloud
metadata at 169.254.169.254), or private-network addresses.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class UnsafeURLError(ValueError):
    """Raised when a URL is not safe to request (bad scheme or private host)."""


def _is_public_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        logger.warning("not a public IP (unparseable): %r", ip_str, exc_info=True)
        return False
    # is_global is False for private, loopback, link-local and reserved ranges.
    return ip.is_global and not ip.is_multicast


def validate_public_url(url: str) -> str:
    """Return the URL stripped if it is safe to request, else raise.

    Safe means: an ``http``/``https`` scheme, a resolvable hostname, and every
    resolved IP address is a public (global) unicast address.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("A URL is required")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"Only http/https URLs are allowed (got {parsed.scheme or 'none'!r})"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeURLError(f"Cannot resolve host {host!r}: {e}")
    if not infos:
        raise UnsafeURLError(f"Cannot resolve host {host!r}")
    for info in infos:
        ip_str = info[4][0]
        if not _is_public_ip(ip_str):
            raise UnsafeURLError(
                f"URL host {host!r} resolves to a non-public address ({ip_str})"
            )
    return url
