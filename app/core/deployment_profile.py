"""The deployment profile switch and the on-premise boot check.

One setting, ``DEPLOYMENT_PROFILE``, names the profile:

* ``hosted`` (unset, ``hosted`` or ``cloud``): today's behaviour; nothing here
  runs a check.
* ``onprem`` (``onprem``, ``on-prem``, ``on_premise`` ...): the service must
  not call outside the box. At boot :func:`enforce_deployment_profile` refuses
  to start while any configured setting would, and names every offender.

Any other value is a refusal to start: a mistyped profile must not quietly
run as hosted.

What counts as "calling outside the box" is decided by rules, not by a list
of services:

* model provider: the provider each role names (``LLM_PROVIDER``,
  ``LLM_FALLBACK_PROVIDER``) is resolved through the runtime's own provider
  table. A name the runtime does not support is refused (the runtime would
  route it to its default cloud provider); a supported provider is refused
  when its endpoint is not inside the box. With no provider named and no key
  for the default one, no model is configured and nothing is called: the
  model-free path answers what it can.
* embedder: a model-hub id needs ``HF_HUB_OFFLINE`` and
  ``TRANSFORMERS_OFFLINE`` truthy (the weights pre-seeded); a local
  directory or ``fake`` needs neither.
* credentials: a non-empty setting named ``*_API_KEY``, ``*_ACCESS_TOKEN`` or
  ``*_DSN`` is a credential for a remote service.
* endpoints: a non-empty ``*_URL`` setting with a host must point inside the
  box: loopback, private or link-local addresses, a single-label (compose
  service) name, or a name that resolves only to such addresses. The
  service's own inbound addresses are not dialled and are not checked.

Network isolation itself is the deployment's job; this check only proves the
configuration does not need egress. It never prints a setting's value.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PROFILE_SETTING = "DEPLOYMENT_PROFILE"
HOSTED = "hosted"
ONPREM = "onprem"

_PROFILE_VALUES = {
    "": HOSTED,
    "hosted": HOSTED,
    "cloud": HOSTED,
    "onprem": ONPREM,
    "on-prem": ONPREM,
    "on_prem": ONPREM,
    "on-premise": ONPREM,
    "on_premise": ONPREM,
}

_PROVIDER_ROLES = ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER")
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_ACCESS_TOKEN", "_DSN")
_ENDPOINT_SUFFIX = "_URL"
# Where users reach this service; the service never dials them.
_INBOUND_ADDRESS_SETTINGS = frozenset({"FRONTEND_URL", "PUBLIC_BASE_URL", "RENDER_EXTERNAL_URL"})
_OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUTHY = {"1", "true", "yes", "on"}


class DeploymentProfileError(RuntimeError):
    """The profile is unknown, or the on-premise box would call outside."""

    def __init__(self, message: str, offenders: list[Offender] | None = None):
        super().__init__(message)
        self.offenders = list(offenders or [])


@dataclass(frozen=True)
class Offender:
    setting: str
    reason: str


def _process_env() -> Mapping[str, str]:
    return os.environ


def _resolve_host(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def deployment_profile(env: Mapping[str, str] | None = None) -> str:
    """``hosted`` or ``onprem``; an unknown value raises."""
    env = _process_env() if env is None else env
    raw = (env.get(PROFILE_SETTING) or "").strip().lower()
    if raw not in _PROFILE_VALUES:
        raise DeploymentProfileError(
            f"{PROFILE_SETTING} names an unknown profile; use one of: "
            f"{HOSTED}, {ONPREM}."
        )
    return _PROFILE_VALUES[raw]


def _value(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _as_ip(text: str):
    """The address a literal names, or None when the text is a host name."""
    literal = text.split("%", 1)[0]
    if ":" not in literal and not (literal.count(".") == 3 and literal.replace(".", "").isdigit()):
        return None
    try:
        return ipaddress.ip_address(literal)
    except ValueError:
        logger.debug("on-prem check: %r is not an address literal", literal)
        return None


def _address_is_inside(ip) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _host_is_inside(host: str, resolve: Callable[[str], list[str]]) -> bool:
    host = host.strip("[]").lower()
    if not host:
        return True
    ip = _as_ip(host)
    if ip is not None:
        return _address_is_inside(ip)
    if host == "localhost" or "." not in host:
        return True
    try:
        addresses = [_as_ip(a) for a in resolve(host)]
    except (OSError, ValueError) as exc:
        logger.warning("on-prem check: %s did not resolve (%s); treated as outside.",
                       host, type(exc).__name__)
        addresses = []
    return bool(addresses) and all(a is not None and _address_is_inside(a) for a in addresses)


def _url_host(url: str) -> str | None:
    """The host a URL dials, or None when the value is not a URL.

    Raises ValueError for a value that looks like a URL but cannot be read."""
    parts = urlsplit(url)
    return parts.hostname if parts.scheme and parts.netloc else None


def _provider_offenders(env, resolve) -> Iterable[Offender]:
    from app.agents.runtime import _llm_config

    for role in _PROVIDER_ROLES:
        named = _value(env, role).lower()
        if not named and role != "LLM_PROVIDER":
            continue
        cfg = _llm_config(provider=named)
        if named and cfg.get("provider") != named:
            yield Offender(
                role,
                f"'{named}' is not a supported provider; the runtime would route it "
                f"to '{cfg.get('provider')}', which is outside the box.",
            )
            continue
        key = cfg.get("env_key")
        if not named and key and not _value(env, key):
            continue  # no model configured: nothing is called
        host = _url_host(cfg.get("url") or "")
        if host and not _host_is_inside(host, resolve):
            yield Offender(role, f"provider '{cfg.get('provider')}' calls {host}, outside the box.")


def _embedder_offenders(env) -> Iterable[Offender]:
    from app.core.rag.embeddings import DEFAULT_EMBEDDING_MODEL

    name = _value(env, "RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if name == "fake" or os.path.isdir(name):
        return
    missing = [f for f in _OFFLINE_FLAGS if _value(env, f).lower() not in _TRUTHY]
    if missing:
        yield Offender(
            "RAG_EMBEDDING_MODEL",
            f"the embedder '{name}' would be fetched from the model hub; set "
            f"{' and '.join(missing)}=1 with the weights pre-seeded, or point "
            "RAG_EMBEDDING_MODEL at a local directory.",
        )


def check_onprem_ready(
    env: Mapping[str, str] | None = None,
    resolve: Callable[[str], list[str]] | None = None,
) -> list[Offender]:
    """Every setting that would make an on-premise box call outside."""
    env = _process_env() if env is None else env
    resolve = resolve or _resolve_host
    found: dict[str, Offender] = {}

    def add(offender: Offender) -> None:
        found.setdefault(offender.setting, offender)

    for offender in _provider_offenders(env, resolve):
        add(offender)
    for offender in _embedder_offenders(env):
        add(offender)
    for name in sorted(env):
        if not _value(env, name):
            continue
        if name.endswith(_CREDENTIAL_SUFFIXES):
            add(Offender(name, "is a credential for a remote service; unset it."))
        elif name.endswith(_ENDPOINT_SUFFIX) and name not in _INBOUND_ADDRESS_SETTINGS:
            try:
                host = _url_host(_value(env, name))
            except ValueError:
                add(Offender(name, "cannot be read as a URL, so it is not shown to be inside the box."))
                continue
            if host and not _host_is_inside(host, resolve):
                add(Offender(name, f"points at {host}, which is not shown to be inside the box."))
    return [found[k] for k in sorted(found)]


def assert_onprem_ready(
    env: Mapping[str, str] | None = None,
    resolve: Callable[[str], list[str]] | None = None,
) -> list[Offender]:
    """Raise naming every offender under the on-premise profile; hosted is a no-op."""
    env = _process_env() if env is None else env
    if deployment_profile(env) != ONPREM:
        return []
    offenders = check_onprem_ready(env, resolve)
    if offenders:
        lines = "\n".join(f"  - {o.setting}: {o.reason}" for o in offenders)
        raise DeploymentProfileError(
            f"{PROFILE_SETTING}={ONPREM} refuses to start; these settings would "
            f"call outside the box:\n{lines}",
            offenders,
        )
    return []


def enforce_deployment_profile() -> None:
    """The boot hook: read the process environment and refuse or return."""
    assert_onprem_ready(_process_env(), _resolve_host)
