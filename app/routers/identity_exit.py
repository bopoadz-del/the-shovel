"""The one exit every streamed frame of a project turn passes through.

Order 2210Z / 0735Z item 3a: a project's identity and its document file
names never leave in a streamed answer, a citation or a Sources line.
``redact_frames`` wraps the whole SSE stream of a turn, outermost, so every
frame any producer writes -- tokens, ``end`` (its content and sources), errors,
and the watchdog's own fallbacks -- is redacted by the project's one rule
(app.core.project_identity).

Token text is redacted on a rolling buffer: the stream holds back the tail
that could still be the start of a name, so a name split across two tokens
("Harbour" + " Tower") is caught. Held text is released before any other
frame, and at the end of the stream.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

from app.core import project_identity

_LOG = logging.getLogger(__name__)


def _payload(raw: str) -> Optional[dict]:
    for line in (raw or "").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            body = line[len("data:"):].strip()
            try:
                value = json.loads(body)
            except (TypeError, ValueError):
                # Not one of our JSON frames (a [DONE] or a comment): redacted as text.
                _LOG.debug("identity_exit: non-JSON frame passed as text")
                return None
            return value if isinstance(value, dict) else None
    return None


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def _safe_cut(redactor: project_identity.Redactor, text: str) -> int:
    """How much of ``text`` can be released now: all but the last
    ``redactor.span`` characters, moved back to the start of any name that
    crosses that point."""
    cut = max(0, len(text) - redactor.span)
    for rx in (redactor._title_re, redactor._member_re):
        if rx is None:
            continue
        for m in rx.finditer(text):
            if m.start() < cut < m.end():
                cut = m.start()
    return cut


async def redact_frames(frames: AsyncIterator[str], redactor: Optional[project_identity.Redactor]
                        ) -> AsyncIterator[str]:
    """Every frame of the stream, redacted by ``redactor`` (None: unchanged)."""
    if redactor is None:
        async for raw in frames:
            yield raw
        return
    held = ""

    def release(final: bool) -> Optional[str]:
        nonlocal held
        if not held:
            return None
        cut = len(held) if final else _safe_cut(redactor, held)
        if cut <= 0:
            return None
        out, held = redactor.text(held[:cut]), held[cut:]
        return _frame({"type": "token", "content": out}) if out else None

    async for raw in frames:
        payload = _payload(raw)
        if payload is not None and payload.get("type") == "token" and isinstance(payload.get("content"), str):
            held += payload["content"]
            frame = release(final=False)
            if frame:
                yield frame
            continue
        frame = release(final=True)
        if frame:
            yield frame
        if payload is not None:
            yield _frame(redactor.deep(payload))
        else:
            yield redactor.text(raw)
    frame = release(final=True)
    if frame:
        yield frame
