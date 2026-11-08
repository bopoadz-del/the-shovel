"""Client address for anonymous rate-limit buckets behind a reverse proxy.

Forwarded headers are attacker-controlled unless the immediate peer is a
proxy we have been told to trust. Trust is enabled only when both of these
are set to a usable value:

* ``TRUSTED_PROXY_COUNT`` — how many right-hand ``X-Forwarded-For`` hops the
  proxy tier may have appended (a positive integer).
* ``TRUSTED_PROXY_CIDRS`` — comma-separated networks of those proxies. The
  socket peer must fall in one of them. A default route (``0.0.0.0/0`` or
  ``::/0``) is rejected. One invalid entry disables trust for the whole list.

If either is missing, zero, negative, or unparseable, forwarding is ignored
and the socket peer is the bucket. That is the closed failure: a forged
header cannot mint a fresh bucket, and an unconfigured deployment does not
pretend the load balancer's address is a distinct client.

When trust is on, the walk starts at the right of ``X-Forwarded-For`` and
moves left only across addresses that themselves sit in the proxy networks,
and never further than ``TRUSTED_PROXY_COUNT`` hops. The first address that
is not a trusted proxy is the client. A malformed token in that window, or a
window that never leaves the proxy networks, keeps the socket peer.
``Forwarded`` and ``X-Real-IP`` are not read: a proxy that only appends
``X-Forwarded-For`` does not overwrite those, so honouring them would trust
the caller.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_BOUND: ContextVar[bool] = ContextVar("rate_limit_client_bound", default=False)
_PEER: ContextVar[str | None] = ContextVar("rate_limit_client_peer", default=None)
_FORWARDED: ContextVar[str | None] = ContextVar(
    "rate_limit_client_forwarded", default=None
)
_PATH: ContextVar[str] = ContextVar("rate_limit_client_path", default="")

# Credential-establishing routes. A Redis outage must not open these.
# Matched by exact path or as a path prefix, after a trailing slash is removed.
_count_warned = False
_network_warned = False

_CREDENTIAL_ROUTE_PREFIXES: tuple[str, ...] = (
    "/v1/users/login",
    "/v1/users/register",
    "/v1/users/forgot-password",
    "/v1/users/reset-password",
    "/v1/users/resend-verification",
    "/v1/users/verify-email",
    "/v1/auth",
)


def trusted_proxy_count() -> int:
    """Hops of ``X-Forwarded-For`` the proxy tier is allowed to append.

    ``0`` means forwarding is off, including when the variable is unset,
    blank, negative, or not an integer.
    """
    global _count_warned
    raw = (os.getenv("TRUSTED_PROXY_COUNT") or "").strip()
    if raw:
        try:
            count = int(raw, 10)
        except ValueError:
            if not _count_warned:
                _count_warned = True
                logger.warning(
                    "TRUSTED_PROXY_COUNT is not an integer; forwarding stays off"
                )
            return 0
        if count > 0:
            return count
    return 0


def trusted_proxy_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Proxy networks, or an empty tuple when trust must stay off.

    An empty or unset variable, a token that is not a network, or a
    default route anywhere in the list yields no networks at all.
    """
    global _network_warned
    raw = (os.getenv("TRUSTED_PROXY_CIDRS") or "").strip()
    if raw:
        nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for part in raw.split(","):
            piece = part.strip()
            if not piece:
                continue
            try:
                net = ipaddress.ip_network(piece, strict=False)
            except ValueError:
                if not _network_warned:
                    _network_warned = True
                    logger.warning(
                        "TRUSTED_PROXY_CIDRS has an entry that is not a network; "
                        "forwarding stays off"
                    )
                return ()
            if net.prefixlen == 0:
                return ()
            nets.append(net)
        if nets:
            return tuple(nets)
    return ()


def _parse_ip(token: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = (token or "").strip().strip('"').strip("'")
    if text.startswith("[") and "]" in text:
        text = text[1:text.find("]")]
    elif text.count(":") == 1 and "." in text:
        host, sep, port = text.rpartition(":")
        if sep and port.isdigit():
            text = host
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        logger.debug("forwarded address is not an IP; that hop is unusable")
        return None
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        return mapped
    return addr


def _contains(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address | None,
    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> bool:
    for net in networks:
        if addr is not None and addr.version == net.version and addr in net:
            return True
    return False


def forwarded_client_host(peer: str | None, forwarded_for: str | None) -> str:
    """Address that should key an anonymous rate-limit bucket.

    ``peer`` is the socket address. ``forwarded_for`` is the raw
    ``X-Forwarded-For`` value, or ``None`` when the header is absent.
    """
    label = (peer or "").strip() or "unknown"
    count = trusted_proxy_count()
    networks = trusted_proxy_networks()
    if count >= 1 and networks:
        peer_ip = _parse_ip(label)
        if peer_ip is not None and _contains(peer_ip, networks) and forwarded_for:
            parts = [_parse_ip(part) for part in str(forwarded_for).split(",")]
            chosen: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
            for addr in reversed(parts[-count:]):
                if addr is None:
                    chosen = None
                    break
                if _contains(addr, networks):
                    continue
                chosen = addr
                break
            if chosen is not None:
                return str(chosen)
    return label


def path_is_credential_route(path: str) -> bool:
    """True for routes that accept or mint credentials."""
    raw = path if isinstance(path, str) else ""
    raw = raw.split("?", 1)[0].rstrip("/") or "/"
    for prefix in _CREDENTIAL_ROUTE_PREFIXES:
        if raw == prefix or raw.startswith(prefix + "/"):
            return True
    return False


def forwarding_context_bound() -> bool:
    """True while a request is inside ``ClientAddressMiddleware``."""
    if _BOUND.get():
        return True
    return False


def bound_peer() -> str | None:
    """Socket peer captured for the current request, if one is bound."""
    if _BOUND.get():
        return _PEER.get()
    return None


def bound_forwarded_for() -> str | None:
    """``X-Forwarded-For`` captured for the current request, if one is bound."""
    if _BOUND.get():
        return _FORWARDED.get()
    return None


def credential_route_fail_closed() -> bool:
    """True when the current request is a credential route and a context is bound."""
    if forwarding_context_bound():
        return path_is_credential_route(_PATH.get())
    return False


def _header(scope: dict, name: str) -> str | None:
    target = name.lower().encode("latin-1")
    found: list[str] = []
    for key, value in scope.get("headers") or ():
        if key.lower() == target:
            found.append(value.decode("latin-1"))
    if found:
        return ", ".join(found)
    return None


class ClientAddressMiddleware:
    """Bind the socket peer, forwarded header, and path for the rate limiter.

    Registered outside the rate-limit middleware and inside CORS, so the
    limiter can read the binding and a rejected request still carries CORS
    headers. Non-HTTP scopes are forwarded untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        peer = client[0] if client else None
        if not isinstance(peer, str) or not peer.strip():
            peer = None
        tokens = (
            (_BOUND, _BOUND.set(True)),
            (_PEER, _PEER.set(peer)),
            (_FORWARDED, _FORWARDED.set(_header(scope, "x-forwarded-for"))),
            (_PATH, _PATH.set(scope.get("path") or "")),
        )
        try:
            await self.app(scope, receive, send)
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
