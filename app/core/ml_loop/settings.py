"""The ML loop block's own settings object.

Everything the block needs from its host arrives here: the on/off switch,
the build sha it stamps on every row, text caps, and the ONE database
session factory. Nothing in ``app.core.ml_loop`` reads host configuration
anywhere else, so a Store deployment configures the block by building one
of these and nothing more.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

#: Env keys the host may use to inject the shipped commit. An unset env is
#: an honest ``None``; the block never invents a hash.
BUILD_SHA_ENV_KEYS = ("GIT_SHA", "SOURCE_VERSION")

#: Env switch for writes. Absent means ON: writes still need a session
#: factory, so an unconfigured host never writes anything.
ENABLED_ENV_KEY = "ML_LOOP_ENABLED"

_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def _env_build_sha() -> str | None:
    for key in BUILD_SHA_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


@dataclass(frozen=True)
class MLLoopSettings:
    """Configuration for one ML loop block instance.

    ``session_factory`` is a zero-argument callable returning a SQLAlchemy
    ``Session`` (usable as a context manager). ``None`` means the store is
    not configured: every write operation reports ``not_configured`` and
    stores nothing.
    """

    enabled: bool = True
    build_sha: str | None = None
    session_factory: Callable[[], Any] | None = None
    #: Cap on any free text the block stores (question, answer, message).
    max_text_chars: int = 8000
    #: Minimum evidence rows before ``propose`` emits a proposal.
    min_proposal_support: int = 3
    #: A route whose refused-turn share reaches this rate (with enough
    #: support) is proposed for a retrieval review.
    refusal_review_rate: float = 0.5
    #: Default look-back window for ``propose`` and ``shadow_report``.
    default_window_days: int = 7

    @classmethod
    def from_env(cls, session_factory: Callable[[], Any] | None = None) -> MLLoopSettings:
        """Settings from the process environment plus a host-supplied factory."""
        flag = os.getenv(ENABLED_ENV_KEY, "").strip().lower()
        return cls(
            enabled=flag not in _FALSE_WORDS,
            build_sha=_env_build_sha(),
            session_factory=session_factory,
        )

    def with_session_factory(self, factory: Callable[[], Any] | None) -> MLLoopSettings:
        return replace(self, session_factory=factory)

    @property
    def store_configured(self) -> bool:
        return self.session_factory is not None

    @property
    def writes_enabled(self) -> bool:
        return self.enabled and self.store_configured
