"""Hat activation on EVERY chat stream, not just the agent's.

Why this exists
---------------
#604 emitted ``hat_signals`` from ``Agent.chat_stream``. That fixed the floor
scorer for turns the agent answers -- and left it reading 0 for every turn the
agent never sees. Live on 51b8e70, ``project_id=master_corpus``:

    "Draft a variation order under FIDIC clause"  ->  hat_signals: 0

The raw ``start`` said ``"mode": "predefined"``. ``app/routers/chat.py`` has
three stream paths -- the agent, the heavy-reasoning stream, and
``_stream_from_predefined`` -- and the last two build their own ``start`` and
``end`` at the router without ever entering the code #604 changed.

Patching each path in turn is how the third one got missed. So this is one
wrapper at the outermost layer, applied where the response is built: every
path that exists is covered, and so is the next one somebody adds.

Idempotent by construction
--------------------------
The agent path already emits its own ``hat_signals`` right after its
``start``. This wrapper therefore injects only when the first frame that is
NOT ``start`` / ``route`` / ``hat_signals`` arrives and none has been seen --
so an agent turn passes through untouched and is never double-counted.

Telemetry must not break chat
-----------------------------
Any failure in here -- a frame that is not JSON, a bug in hat scoring -- passes
the frame through unchanged. Losing one turn's scores is the right price for a
defect in scoring. Losing the answer is not.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

_LOG = logging.getLogger(__name__)

#: Frames that may legitimately precede ``hat_signals``.
_PREAMBLE = frozenset({"start", "route", "hat_signals", "heartbeat"})

_DATA_PREFIX = "data:"


def _decode(frame: Any) -> Optional[dict[str, Any]]:
    """The JSON object inside an SSE ``data:`` frame, or None."""
    if not isinstance(frame, str) or not frame.startswith(_DATA_PREFIX):
        return None
    try:
        payload = json.loads(frame[len(_DATA_PREFIX):].strip())
    except (ValueError, TypeError):
        _LOG.debug("hat_frames: non-JSON SSE frame; passing through", exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _encode(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _hat_event(message: str) -> Optional[dict[str, Any]]:
    """The ``hat_signals`` payload, or None when hats are off or scoring fails."""
    try:
        from app.agents.runtime import hat_scores_sse

        return hat_scores_sse(message or "")
    except Exception:  # noqa: BLE001 - telemetry must never take down chat
        _LOG.exception("hat_frames: hat scoring failed; stream continues without it")
        return None


async def with_hat_signals(
    frames: AsyncIterator[str], message: str
) -> AsyncIterator[str]:
    """Pass frames through, guaranteeing hat activation is on the stream.

    Emits one ``hat_signals`` frame before the first substantive frame if the
    inner stream did not emit its own, and stamps ``hat_signals`` /
    ``hat_selected`` onto the terminal ``end`` when they are missing.
    """
    hat_evt = _hat_event(message)
    if not hat_evt:
        # Hats off (the default) or scoring failed: a pure pass-through.
        async for frame in frames:
            yield frame
        return

    seen = False
    async for frame in frames:
        try:
            event = _decode(frame)
            kind = event.get("type") if event else None

            if kind == "hat_signals":
                seen = True
            elif event is not None and kind not in _PREAMBLE and not seen:
                # First substantive frame and nobody has reported activation.
                yield _encode(hat_evt)
                seen = True

            if kind == "end" and event is not None and not event.get("hat_signals"):
                stamped = dict(event)
                stamped["hat_signals"] = list(hat_evt.get("hat_signals") or [])
                stamped["hat_selected"] = hat_evt.get("selected")
                frame = _encode(stamped)
        except Exception:  # noqa: BLE001 - never let telemetry eat a frame
            _LOG.exception("hat_frames: frame passed through unmodified")
        yield frame
