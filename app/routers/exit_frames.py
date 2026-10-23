"""The exit check on EVERY chat stream, not just the agent's.

``app/routers/chat.py`` and ``app/routers/agents.py`` answer a turn on several
paths (the agent, heavy reasoning, predefined dispatch), and the last two
build their own ``end`` at the router. One wrapper where the response is
built covers every path that exists and the next one somebody adds, the same
way ``hat_frames`` does for hat activation.

The client replaces the streamed bubble with ``end.content`` (or the
accumulated tokens when there is none), so the checked text is put on the
``end`` frame: what stays on screen and what the Sources list shows is what
:mod:`app.agents.answer_exit` let through.

Any failure passes the frame through unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.routers.hat_frames import _decode, _encode

_LOG = logging.getLogger(__name__)


async def with_answer_exit(frames: AsyncIterator[str]) -> AsyncIterator[str]:
    from app.agents import answer_exit
    from app.core.offload import off_loop

    own = answer_exit.open_turn()
    streamed: list[str] = []
    async for frame in frames:
        try:
            event = _decode(frame)
            kind = event.get("type") if event else None
            if kind == "token" and isinstance(event.get("content"), str):
                streamed.append(event["content"])
            elif kind == "end" and event is not None:
                # The agent opens its own record in this same context; read
                # whichever is current so its evidence resolves citations.
                turn = answer_exit.current_turn() or own
                frame = _encode(await off_loop(answer_exit.check_end_event, event, "".join(streamed), turn))
        except Exception:  # noqa: BLE001 - the exit check must never eat a frame
            _LOG.exception("exit_frames: frame passed through unmodified")
        yield frame
