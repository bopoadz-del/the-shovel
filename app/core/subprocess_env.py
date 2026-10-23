"""Scrubbed environment for subprocesses (security fix — audit §6.1).

Subprocesses spawned by the app — file converters (antiword, catdoc),
the code/bash sandbox block, hardware probes — previously inherited the FULL
parent environment, including ``SECRET_KEY``, ``DATABASE_URL`` (with the DB
password), the LLM/provider API keys, ``DATA_ENCRYPTION_KEY``, cloud tokens,
and the master key. The bash sandbox could therefore ``echo $SECRET_KEY`` and
exfiltrate it to output.

This builds a copy of ``os.environ`` with every secret-bearing variable removed
while keeping functional system env (PATH, HOME, LANG, LC_*, TMPDIR,
LD_LIBRARY_PATH, TESSDATA_PREFIX, PYTHON*, NODE_*, …) intact.

**Denylist, not allowlist — deliberate.** A denylist can only ever REMOVE a
variable that a working subprocess never needed (secrets are not read by a file
converter or a code sandbox), so it cannot break a subprocess that was already
working. Over-stripping a non-secret *config* var is harmless because these
subprocesses do not read the app's configuration from the environment. On a
live deployment that "cannot regress a working path" property matters more than
the marginal tightness of an allowlist.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

# Exact secret names (belt). Connection strings that can carry credentials
# (DATABASE_URL, REDIS_URL) and named secrets that don't match a substring
# pattern (GDRIVE_SERVICE_ACCOUNT_JSON) are pinned here.
_FORBIDDEN_EXACT = frozenset({
    "SECRET_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "DATA_ENCRYPTION_KEY",
    "CEREBRUM_MASTER_KEY",
    "SENTRY_DSN",
    "GDRIVE_SERVICE_ACCOUNT_JSON",
    "BOOTSTRAP_USER_PASSWORD",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_ACCESS_TOKEN",
    "ONEDRIVE_ACCESS_TOKEN",
})

# Substring patterns (suspenders): any env var whose NAME contains one of these
# (case-insensitive) is treated as a secret and stripped. Covers *_API_KEY,
# *_TOKEN, *_SECRET, *_PASSWORD, R2_*ACCESS_KEY*, CEREBRUM_API_KEY_*,
# GITHUB_TOKEN, RENDER_API_KEY, service-account and signing keys, etc.
_FORBIDDEN_SUBSTRINGS = (
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "_TOKEN",
    "TOKEN_",
    "SERVICE_ACCOUNT",
    "_DSN",
    "MASTER_KEY",
    "SIGNING_KEY",
    "ENCRYPTION_KEY",
)


def is_secret_env_name(name: str) -> bool:
    """True if an env var NAME denotes a secret that must not reach a subprocess."""
    up = name.upper()
    if up in _FORBIDDEN_EXACT:
        return True
    return any(sub in up for sub in _FORBIDDEN_SUBSTRINGS)


def scrubbed_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of ``os.environ`` with secret-bearing vars removed.

    ``extra`` values are applied last (so an explicit functional override like
    ``PYTHONDONTWRITEBYTECODE`` always wins). Pass this as ``env=`` to every
    ``subprocess.run`` / ``asyncio.create_subprocess_*`` call.
    """
    env = {k: v for k, v in os.environ.items() if not is_secret_env_name(k)}
    if extra:
        env.update(extra)
    return env
