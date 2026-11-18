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

On a COMPUTED turn (a tool result stated a figure) the tool speaks first:
each new tool statement is emitted as its own token frame the moment the
result is on the turn, in call order, an identical statement once, and from
then on the model's own tokens are held, not streamed, until the exit check
on ``end`` has read them (:func:`answer_exit.hold_stated_figures`). The
user never watches the model state a figure the check then takes back.

Any failure passes the frame through unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.routers.hat_frames import _decode, _encode

_LOG = logging.getLogger(__name__)


async def with_answer_exit(frames: AsyncIterator[str], question: str = "") -> AsyncIterator[str]:
    """``question`` is the user's turn as asked; the answer contract reads it
    to tell which evidence values answer the asked thing. The checked answer's
    structured citations are stored on the assistant message this turn stored
    (W2-a), so a document produced from that answer lists its sources from them."""
    from app.agents import answer_exit
    from app.core import agent_memory
    from app.core.offload import off_loop

    own = answer_exit.open_turn(question)
    # The holder lives for this request's task; the next turn gets its own.
    stored, _ = agent_memory.watch_stored_answer()
    streamed: list[str] = []
    led: list[str] = []
    seen_results = 0
    async for frame in frames:
        try:
            event = _decode(frame)
            kind = event.get("type") if event else None
            turn = answer_exit.current_turn() or own
            if len(turn.computed_results) != seen_results:
                seen_results = len(turn.computed_results)
                for lead in await off_loop(answer_exit.computed_leads, turn):
                    if lead not in led:
                        led.append(lead)
                        yield _encode({"type": "token", "content": lead + "\n\n",
                                       "request_id": (event or {}).get("request_id")})
            if kind == "token" and isinstance(event.get("content"), str):
                streamed.append(event["content"])
                if led:
                    continue  # held: the exit check on ``end`` reads it first
            elif kind == "end" and event is not None:
                # The agent opens its own record in this same context; the
                # current one (read above) resolves its evidence's citations.
                # The router knows the asked text even when the agent's own
                # record was opened without it.
                turn.question = turn.question or own.question
                checked = await off_loop(answer_exit.check_end_event, event, "".join(streamed), turn)
                frame = _encode(checked)
                if stored.get("message_id"):
                    await off_loop(_store_citations, stored["message_id"], checked)
        except Exception:  # noqa: BLE001 - the exit check must never eat a frame
            _LOG.exception("exit_frames: frame passed through unmodified")
        yield frame


def _store_citations(message_id: str, event: dict) -> bool:
    """Never raises: storing the citations must not break the answer."""
    try:
        from app.agents import answer_exit
        from app.core import agent_memory

        return agent_memory.attach_citations(message_id, answer_exit.citations_to_store(event))
    except Exception:  # noqa: BLE001 -- bookkeeping only
        _LOG.exception("exit_frames: citations not stored")
        return False
