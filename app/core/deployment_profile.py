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
* credentials: a non-empty setting whose name contains KEY, TOKEN, SECRET,
  DSN, PASSWORD, PASSWD or CREDENTIAL anywhere, or has PASSPHRASE, AUTH,
  BEARER, PWD or SIGNATURE as a whole underscore-separated word, is a
  credential for a remote service, whatever the rest of the name says.
  Exceptions, all stated here: the box's own local secrets (generated on the
  box by the installer, never sent anywhere) on a short allow list; the
  shell's own working-directory setting (PWD, exactly that name) when it
  holds an absolute path; and a plain whole number under a name that has a count
  word (MAX, MIN, LIMIT, BUDGET, COUNT, TOKENS) as a whole underscore-
  separated word and whose only credential word is TOKEN: a token budget,
  not a secret (ADMIN_TOKEN or ACCOUNT_TOKEN hold MIN or COUNT only inside
  another word, so they stay credentials). A number under a KEY, SECRET,
  PASSWORD, PASSWD, CREDENTIAL or DSN name is a secret.
* endpoints: judged by the VALUE, not the name. Every non-empty setting is
  read, and a host found ANYWHERE in it counts: URLs anywhere in the text; a
  value that is JSON is walked string by string; the rest is split on
  separators, ``=`` and quotes, and each token is read as ``[//][user@]host
  [:port][/path][?query][#fragment]`` (percent escapes decoded, non-ASCII
  names converted with IDNA) where host is an address literal (IPv4, IPv6
  bracketed or not, or a decimal/hex IPv4 in a host position, or a bare
  decimal/hex number that fits IPv4 under a name with no count word) or a
  dotted name (labels may hold an underscore; the last label alphabetic or
  punycode). A token that is a path (starts with one ``/``, ``./``, ``~`` or
  holds a backslash) is not a host; a bare file name such as ``name.ext``
  reads as a host, so write it as ``./name.ext``. A setting whose value is a
  path to a file that holds an endpoint is NOT read: the file's contents are
  never checked. Every host must point inside the box: loopback, private or
  link-local addresses, a single-label (compose service) name, or a name
  that resolves only to such addresses. The service's own inbound addresses
  are not dialled and are not checked.
* fail closed: a value with a token that is none of the known-safe shapes
  (a host, a path, a number, or plain ASCII words joined by ``/`` and ``:``
  in which no piece is a host or address) is refused with its setting name.
  The pattern rules above cannot be complete; what they cannot read is not
  shown to stay inside, so it does not pass.

Network isolation itself is the deployment's job; this check only proves the
configuration does not need egress. It never prints a setting's value.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

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
# A name containing any of these words names a secret.
_CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "DSN", "PASSWORD", "PASSWD", "CREDENTIAL")
# A name having any of these as a whole underscore-separated word names a
# secret (as substrings they would catch AUTHOR or SIGNATURES_DIR's kin).
_CREDENTIAL_WHOLE_WORDS = frozenset({"PASSPHRASE", "AUTH", "BEARER", "PWD", "SIGNATURE"})
# Set by the shell to the working directory: not a secret when the value is
# an absolute path. Exactly this name; DB_PWD is a credential.
_SHELL_WORKING_DIR = "PWD"
# The box's own secrets: generated on the box (deploy/onprem/install.sh), used
# only by this service and its local database, never sent anywhere. GPG_KEY is
# set by the python base image: the public fingerprint of its release-signing
# key, not a secret.
_LOCAL_ONLY_SECRETS = frozenset({
    "SECRET_KEY",
    "DATA_ENCRYPTION_KEY",
    "CEREBRUM_MASTER_KEY",
    "POSTGRES_PASSWORD",
    "BOOTSTRAP_USER_PASSWORD",
    "GPG_KEY",
})
# A plain whole number is a count or a limit only under a name that says so.
_COUNT = re.compile(r"\d{1,9}")
# Whole underscore-separated words only: ADMIN or ACCOUNT is not MIN or COUNT.
_COUNT_WORDS = frozenset({"MAX", "MIN", "LIMIT", "BUDGET", "COUNT", "TOKENS"})
# The only credential word a count's name may carry (MAX_TOKENS); never
# KEY, SECRET, PASSWORD, PASSWD, CREDENTIAL or DSN.
_COUNTABLE_WORDS = frozenset({"TOKEN"})
# A dotted host name: the last label alphabetic or punycode (so "3.11" or
# "qwen2.5" is not a host).
_LABEL = r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?"
_DOTTED_NAME = re.compile(r"(?:" + _LABEL + r"\.)+(?:[a-z]{2,63}|xn--[a-z0-9-]{1,59})\.?", re.IGNORECASE)
# A URL anywhere in a value.
_URL_IN_TEXT = re.compile(r"[a-z][a-z0-9+.-]*://[^\s,;\"'<>`]+", re.IGNORECASE)
# What separates tokens in a value: list separators, key=value, quotes and braces.
_TOKEN_SEPARATORS = re.compile(r"[\s,;=\"'`(){}<>|]+")
_BRACKETED_V6 = re.compile(r"\[(?P<host>[0-9a-f:.%]+)\](?::\d{1,5})?", re.IGNORECASE)
_PORT = re.compile(r"\d{1,5}")
# A bare number a resolver reads as an IPv4 address (decimal or 0x hex).
_BARE_NUMBER = re.compile(r"\d+|0x[0-9a-f]+", re.IGNORECASE)
# One piece of a plain value: ASCII letters, digits and a few marks, no
# separator a host or URL needs.
_PLAIN_PIECE = re.compile(r"[A-Za-z0-9_.+\-*~!$^&%@]*")
_PLAIN_SPLIT = re.compile(r"[/:?#@\[\]]")
# Where users reach this service; the service never dials them.
_INBOUND_ADDRESS_SETTINGS = frozenset({"FRONTEND_URL", "PUBLIC_BASE_URL", "RENDER_EXTERNAL_URL"})
_OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUTHY = {"1", "true", "yes", "on"}


class _Unclassified(ValueError):
    """A value holding a part that is none of the known-safe shapes."""


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
    try:
        return ipaddress.ip_address(literal)
    except ValueError:
        logger.debug("on-prem check: %r is not an address literal", literal)
        return None


def _address_is_inside(ip) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _numeric_ipv4(text: str):
    """The IPv4 address a numeric host names as the system resolver reads it
    (decimal ``167772162``, hex ``0x0a000002``, octal or short dotted forms
    such as ``10.2`` or ``0x7f.1``), or None."""
    parts = text.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    nums = []
    for part in parts:
        try:
            if part[:2].lower() == "0x":
                nums.append(int(part, 16))
            elif len(part) > 1 and part.startswith("0"):
                nums.append(int(part[1:], 8))
            elif part.isdigit():
                nums.append(int(part))
            else:
                return None
        except ValueError:
            return None
    *head, last = nums
    if any(n > 255 for n in head) or last >= 256 ** (4 - len(head)):
        return None
    value = last
    for k, n in enumerate(head):
        value += n << (8 * (3 - k))
    return ipaddress.IPv4Address(value)


def _ascii_name(text: str) -> str:
    """The text with a non-ASCII name converted as a resolver would (IDNA).

    Raises _Unclassified when it cannot be converted."""
    if text.isascii():
        return text
    try:
        return text.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise _Unclassified("a non-ASCII part is not a readable name") from exc


def _host_is_inside(host: str, resolve: Callable[[str], list[str]]) -> bool:
    try:
        host = _ascii_name(host.strip("[]")).rstrip(".").lower()
    except _Unclassified:
        return False
    ip = _as_ip(host)
    if ip is None:
        ip = _numeric_ipv4(host)
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


def _name_words(name: str) -> set[str]:
    """The whole words of a setting name, split on underscores (and any other
    non-alphanumeric mark)."""
    return {w for w in re.split(r"[^A-Z0-9]+", name.upper()) if w}


def _is_count_name(name: str) -> bool:
    return bool(_name_words(name) & _COUNT_WORDS)


def _is_credential(name: str, value: str) -> bool:
    upper = name.upper()
    words = {word for word in _CREDENTIAL_WORDS if word in upper}
    words |= _name_words(name) & _CREDENTIAL_WHOLE_WORDS
    if not words or upper in _LOCAL_ONLY_SECRETS:
        return False
    if upper == _SHELL_WORKING_DIR and value.startswith("/"):
        return False
    is_count = (
        _COUNT.fullmatch(value) is not None
        and words <= _COUNTABLE_WORDS
        and _is_count_name(name)
    )
    return not is_count


def _strings_in_json(node) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, int) and not isinstance(node, bool):
        yield str(node)  # a whole number may be a decimal IPv4
    elif isinstance(node, dict):
        for key, item in node.items():
            yield str(key)
            yield from _strings_in_json(item)
    elif isinstance(node, list):
        for item in node:
            yield from _strings_in_json(item)


def _is_plain(token: str) -> bool:
    """A known-safe non-host shape: ASCII words (a number, a version, a model
    tag, a relative name) joined by ``/ : ? # @ [ ]``, no piece of which is a
    host name or an address."""
    for piece in _PLAIN_SPLIT.split(token):
        piece = _ascii_name(piece)
        if not _PLAIN_PIECE.fullmatch(piece):
            return False
        if piece and (_DOTTED_NAME.fullmatch(piece) or _as_ip(piece) is not None):
            return False  # a host where no host reading put it
    return True


def _token_host(token: str, bare_numbers_are_hosts: bool = False) -> str | None:
    """The host one ``[//][user@]host[:port][/path][?q][#f]`` token dials, or
    None for a path or a plain value.

    Raises _Unclassified for a token that is none of the known-safe shapes."""
    if not token:
        return None
    if "%" in token:
        token = unquote(token)  # an encoded dot or slash, read as a resolver reads it
    scheme_relative = token.startswith("//")
    if scheme_relative:
        token = token[2:]
    elif token.startswith(("/", "./", "../", "~")) or "\\" in token:
        return None  # a path, not a host
    in_host_position = scheme_relative or "@" in token
    rest = re.split(r"[?#]", token.rsplit("@", 1)[-1], maxsplit=1)[0]
    v6 = _BRACKETED_V6.match(rest)
    if v6:
        return v6.group("host")
    if _as_ip(rest) is not None:
        return rest  # an address literal, IPv6 unbracketed included
    head: str | None = rest.split("/", 1)[0]
    if ":" in head:
        head, _, port = head.rpartition(":")
        if ":" in head or not _PORT.fullmatch(port):
            head = None  # not host:port (a version tag or a time)
        else:
            in_host_position = True
    if head:
        head = _ascii_name(head)
        if _as_ip(head) is not None or _DOTTED_NAME.fullmatch(head):
            return head
        bare = bare_numbers_are_hosts and head == rest and _BARE_NUMBER.fullmatch(head)
        if (in_host_position or bare) and _numeric_ipv4(head) is not None:
            return head
    if not _is_plain(token):
        raise _Unclassified("a part is not a known-safe shape")
    return None


def _dialled_hosts(value: str, bare_numbers_are_hosts: bool = False) -> list[str]:
    """Every host the value would dial, wherever it sits in the value.

    Raises ValueError for a part that looks like a URL but cannot be read,
    and _Unclassified for a part that is none of the known-safe shapes."""
    text = value.strip()
    if text[:1] in "{[":
        try:
            parsed = json.loads(text)
        except ValueError:
            logger.debug("on-prem check: a value starting with a bracket is not JSON")
        else:
            return [
                h for s in _strings_in_json(parsed)
                for h in _dialled_hosts(s, bare_numbers_are_hosts)
            ]
    hosts: list[str] = []
    for url in _URL_IN_TEXT.findall(text):
        host = _url_host(url)
        if host:
            hosts.append(host)
    rest = _URL_IN_TEXT.sub(" ", text)
    for token in _TOKEN_SEPARATORS.split(rest):
        host = _token_host(token, bare_numbers_are_hosts)
        if host:
            hosts.append(host)
    return hosts


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
        if _is_credential(name, _value(env, name)):
            add(Offender(name, "is a credential for a remote service; unset it."))
        elif name not in _INBOUND_ADDRESS_SETTINGS:
            try:
                hosts = _dialled_hosts(_value(env, name), not _is_count_name(name))
            except _Unclassified:
                add(Offender(
                    name,
                    "holds a part that is not a known-safe shape (a host, a path, a number "
                    "or plain words), so it is not shown to stay inside the box; refused.",
                ))
                continue
            except ValueError:
                add(Offender(name, "cannot be read as a URL, so it is not shown to be inside the box."))
                continue
            outside = [h for h in hosts if not _host_is_inside(h, resolve)]
            if outside:
                add(Offender(name, f"points at {outside[0]}, which is not shown to be inside the box."))
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
